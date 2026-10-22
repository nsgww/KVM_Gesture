#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KVM 手势识别主程序（板端）。

用法
----
    # 摄像头 0，事件输出到 stdout
    python3 kvm_gesture.py --source 0

    # 指定输出文件 + 显示预览（需板子接显示器）
    python3 kvm_gesture.py --source 0 --out /tmp/events.jsonl --preview

    # 网页预览（无显示器时用）：浏览器打开 http://板子IP:8082/
    python3 kvm_gesture.py --source rtsp://... --web-port 8082

    # 用视频文件回放调参
    python3 kvm_gesture.py --source test.mp4 --out /tmp/ev.jsonl

    # 只看事件，不落文件
    python3 kvm_gesture.py --source 0 --print-only

事件协议（每行一个 JSON）
--------------------------
    {"action":"MOVE","x":0.42,"y":0.55,"scale":1.0,"ts":1712345678901}

退出：Ctrl-C，或在 preview 窗口按 q。
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gesture_core import GestureRecognizer, GestureConfig, draw_hud  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PALM = os.path.join(HERE, "models", "palm_detection_mediapipe_2023feb.onnx")
DEFAULT_HAND = os.path.join(HERE, "models", "handpose_estimation_mediapipe_2023feb.onnx")

_running = True


def _sigint(signum, frame):
    global _running
    _running = False


class _MjpegHub:
    """发布最新一帧 JPEG，供 --web-port 的网页预览（WebSocket/MJPEG）消费。"""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, last_seq: int, timeout: float = 1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._seq, self._jpeg


# 预览页：WebSocket 收 JPEG 帧，canvas 解码即绘、只画最新帧。
# Chrome 的 <img> MJPEG 管线有 ~0.9s 缓冲/节流，websocket+canvas 与摄像头
# 自带播放器同思路，端到端可压到 ~0.2s。
_WEB_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVM Gesture Preview</title>
<style>
html,body{margin:0;height:100%;background:#000;overflow:hidden}
#c{position:absolute;inset:0;margin:auto;max-width:100%;max-height:100%}
#s{position:fixed;top:8px;left:8px;width:10px;height:10px;border-radius:50%;background:#e33}
#s.on{background:#3c5}
#v{position:fixed;right:8px;bottom:6px;color:#666;font:12px sans-serif}
</style></head><body>
<canvas id="c"></canvas><div id="s"></div><div id="v">v2 · ws</div>
<script>
var c = document.getElementById('c'), ctx = c.getContext('2d'),
    dot = document.getElementById('s'), latest = null, busy = false;
async function pump() {
  if (busy) return;
  busy = true;
  while (latest) {
    var blob = latest; latest = null;
    try {
      if (window.createImageBitmap) {
        var bmp = await createImageBitmap(blob);
        if (c.width !== bmp.width || c.height !== bmp.height) { c.width = bmp.width; c.height = bmp.height; }
        ctx.drawImage(bmp, 0, 0);
        bmp.close();
      } else {
        var url = URL.createObjectURL(blob), img = new Image();
        img.src = url; await img.decode();
        if (c.width !== img.width || c.height !== img.height) { c.width = img.width; c.height = img.height; }
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
      }
    } catch (e) {}
  }
  busy = false;
}
function connect() {
  var ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'blob';
  ws.onopen = function () { dot.classList.add('on'); };
  ws.onmessage = function (ev) { latest = ev.data; pump(); };
  ws.onclose = function () { dot.classList.remove('on'); setTimeout(connect, 500); };
  ws.onerror = function () { ws.close(); };
}
connect();
</script></body></html>
""".encode("utf-8")

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(key: str) -> str:
    digest = hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _ws_pack(payload: bytes) -> bytes:
    """打包一个 WebSocket 二进制帧（FIN + binary，服务端发送不掩码）。"""
    n = len(payload)
    if n < 126:
        head = bytes((0x82, n))
    elif n < 65536:
        head = bytes((0x82, 126)) + n.to_bytes(2, "big")
    else:
        head = bytes((0x82, 127)) + n.to_bytes(8, "big")
    return head + payload


def _make_web_handler(hub: _MjpegHub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_WEB_HTML)))
                # 禁止缓存：旧版页面（MJPEG <img>）在 Chrome 里有 ~0.9s 缓冲，
                # 拿到旧页面会误判成"延迟 1 秒"，这里强制每次取新版本。
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                self.wfile.write(_WEB_HTML)
                return
            if self.path.split("?")[0] == "/ws":
                key = self.headers.get("Sec-WebSocket-Key")
                if not key or self.headers.get("Upgrade", "").lower() != "websocket":
                    self.send_error(400)
                    return
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", _ws_accept_key(key))
                self.end_headers()
                self.close_connection = True
                try:
                    self.connection.setsockopt(socket.IPPROTO_TCP,
                                               socket.TCP_NODELAY, 1)
                except OSError:
                    pass
                seq = -1
                while _running:
                    prev = seq
                    seq, jpeg = hub.wait_frame(seq)
                    if jpeg is None or seq == prev:
                        continue
                    try:
                        self.wfile.write(_ws_pack(jpeg))
                    except (BrokenPipeError, ConnectionResetError,
                            ConnectionAbortedError, OSError):
                        break
                return
            if self.path.startswith("/stream.mjpeg"):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seq = -1
                while _running:
                    seq, jpeg = hub.wait_frame(seq)
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: %d\r\n\r\n" % len(jpeg)
                                         + jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError,
                            ConnectionAbortedError, OSError):
                        break
                return
            self.send_error(404)

    return Handler


class _StreamReader:
    """后台线程持续排空视频流，只保留最新一帧。

    识别处理速度低于码流帧率时，若主循环逐帧顺序读，处理不完的帧会在
    FFmpeg/socket 缓冲区积压——TCP 不会主动丢包，延迟随使用时间持续增长。
    后台线程始终尽力读、只留最新帧，主循环拿到的一直是"现在"的画面。

    同时统计"读帧耗时"：实时流每帧应等 ~frame_interval（25fps ≈ 40ms）；
    若连续多帧 read() 都在 12ms 内返回，说明在追积压的旧帧，延迟正在累积。
    backlog 即该比例，供主循环告警。
    """

    def __init__(self, cap, web_pub=None):
        self._cap = cap
        self._web_pub = web_pub
        self._stop = False
        self._cond = threading.Condition()
        self._frame = None
        self._ok = True
        self.seq = 0
        self._read_ms = collections.deque(maxlen=40)
        self.backlog = 0.0          # 最近 40 帧中"读得过快"的比例
        self.read_median = 0.0      # 最近读帧耗时中位数（ms）
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
            dt = (time.time() - t0) * 1000.0
            with self._cond:
                if ok and frame is not None:
                    self._frame = frame
                    self.seq += 1
                    self._ok = True
                    self._read_ms.append(dt)
                    vals = sorted(self._read_ms)
                    self.read_median = vals[len(vals) // 2]
                    fast = sum(1 for v in vals if v < 12.0)
                    self.backlog = fast / len(vals)
                else:
                    self._ok = False
                self._cond.notify_all()
            # 预览与推理主循环解耦：采集到帧立刻交给编码线程
            if ok and frame is not None and self._web_pub is not None:
                self._web_pub.submit(frame)

    def latest(self):
        """(ok, frame, seq)。seq 不变 = 还没有更新的帧。"""
        with self._cond:
            return self._ok, self._frame, self.seq

    def stop(self):
        self._stop = True
        self._th.join(timeout=8.0)   # read 可能被 FFmpeg 阻塞，等它带超时退出


def _publish_web(hub: _MjpegHub, frame, width: int,
                 fps: float = None, found: bool = False) -> None:
    """缩放 ->（可选）叠状态文字 -> JPEG -> 发布。文字画在缩放/拷贝后的图上，不污染推理输入。"""
    out = frame
    if width and out.shape[1] > width:
        h, w = out.shape[:2]
        out = cv2.resize(out, (width, max(1, int(h * width / w))),
                         interpolation=cv2.INTER_AREA)
    if fps is not None:
        if out is frame:
            out = frame.copy()
        cv2.putText(out, f"fps {fps:4.1f}  hand {'Y' if found else 'N'}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        hub.publish(buf.tobytes())


class _WebPublisher:
    """独立编码线程：采集到帧立即缩放+JPEG 编码+发布，与推理主循环解耦。
    板端 CPU 推理一轮 ~140ms（约 7fps），预览若挂在主循环上，每帧平均
    多等半个循环周期；解耦后预览链路只剩 采集->编码->网络->浏览器绘制，
    推理快慢不再影响预览延迟和帧率。"""

    def __init__(self, hub: _MjpegHub, width: int):
        self._hub = hub
        self._width = width
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._stop = False
        self.fps = 0.0          # 主循环每轮更新，OSD 用
        self.found = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def submit(self, frame) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify()

    def _loop(self) -> None:
        last = 0
        while not self._stop:
            with self._cond:
                if self._seq == last:
                    self._cond.wait(1.0)
                frame, last = self._frame, self._seq
            if frame is not None:
                _publish_web(self._hub, frame, self._width, self.fps, self.found)

    def stop(self) -> None:
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._th.join(timeout=2.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KVM 隔空手势识别（识别侧）")
    p.add_argument("--source", default="0",
                   help="摄像头编号（0/1/2...）或视频文件路径")
    p.add_argument("--palm-model", default=DEFAULT_PALM)
    p.add_argument("--handpose-model", default=DEFAULT_HAND)
    p.add_argument("--out", default="", help="事件输出文件（JSONL）；留空则不落盘")
    p.add_argument("--print-events", action="store_true", default=True,
                   help="事件打印到 stdout（默认开）")
    p.add_argument("--quiet", action="store_true", help="不打印事件，只落文件")
    p.add_argument("--preview", action="store_true", help="显示预览窗口（需要图形环境）")
    p.add_argument("--web-port", type=int, default=0,
                   help="开启低延迟网页预览的端口（0=关闭），浏览器访问 http://板子IP:端口/（/stream.mjpeg 保留给 VLC 等播放器）")
    p.add_argument("--web-width", type=int, default=960,
                   help="网页预览编码宽度，等比缩放（降低 JPEG 编码耗时），0=原始分辨率")
    p.add_argument("--web-overlay", action="store_true",
                   help="网页预览叠加骨骼/HUD（需等推理完成，预览延迟略高）")
    p.add_argument("--max-hands", type=int, default=1, help="最大跟踪手数")
    p.add_argument("--palm-threshold", type=float, default=0.4)
    p.add_argument("--hand-conf", type=float, default=0.5)
    p.add_argument("--control-margin", type=float, default=0.10,
                   help="控制区内缩比例，边缘留白")
    p.add_argument("--no-mirror", action="store_true", help="不镜像 X 坐标")
    p.add_argument("--smooth", type=float, default=0.35, help="平滑系数 0.1~1，越小越稳")
    p.add_argument("--fps", type=float, default=0, help="摄像头帧率上限，0=不设置")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--backend", type=int, default=0, help="cv.dnn backendId")
    p.add_argument("--target", type=int, default=0, help="cv.dnn targetId")
    p.add_argument("--palm-every", type=int, default=2,
                   help="固定模式下每N帧做一次 palm 检测（按需模式下作兜底倍数）")
    p.add_argument("--no-palm-ondemand", action="store_true",
                   help="关闭按需 palm 检测。默认开启：只在丢手时重检，帧率约翻倍，"
                        "延迟从 ~140ms/帧 降到 ~75ms/帧")
    p.add_argument("--max-seconds", type=float, default=0, help="运行多少秒后自动退出，0=不限")
    p.add_argument("--reopen-after", type=float, default=5.0,
                   help="视频流多少秒无新帧后自动重开，0=禁用")
    p.add_argument("--rtsp-transport", default="tcp", choices=("tcp", "udp"),
                   help="RTSP 传输协议。tcp 稳定但遇丢包会堆积；udp 延迟更低但可能花屏")
    p.add_argument("--reader", default="ffmpeg", choices=("ffmpeg", "opencv"),
                   help="流读取方式。ffmpeg=裸帧管道（绕过 OpenCV 内部队列，延迟最低，默认）；"
                        "opencv=VideoCapture（兼容性好，但多压 2~4 帧）")
    p.add_argument("--rtsp-buffer", type=int, default=524288,
                   help="RTSP 接收缓冲字节数（默认 512KB）。调小延迟更低，过小会掉帧")
    p.add_argument("--rtsp-max-delay", type=int, default=0,
                   help="FFmpeg max_delay 微秒（默认 0=不等待重排序；摄像头带 B 帧时可设 200000）")
    p.add_argument("--lag-warn", type=float, default=5.0,
                   help="积压告警间隔秒（0=关闭）：连续读帧过快说明解码跟不上，延迟在累积")
    p.add_argument("--stats", action="store_true", help="退出时打印性能统计")
    return p


class FfmpegPipeCapture:
    """绕过 OpenCV 的 VideoCapture 队列，直接用 ffmpeg 管道读裸帧。

    为什么：cv2.VideoCapture 内部有 packet/frame 队列，即使设了
    CAP_PROP_BUFFERSIZE 和 max_delay，实测仍会多压 2~4 帧（10fps 下 = 200~400ms）。
    改成 `ffmpeg -fflags nobuffer -flags low_delay ... -f rawvideo -` 后，
    解码出一帧就立刻从管道吐出，延迟由摄像头本身决定。

    提供与 cv2.VideoCapture 相同的 read()/release() 接口，便于复用现有逻辑。
    """

    def __init__(self, url: str, width: int, height: int, transport: str = "tcp"):
        self.url = url
        self.w = int(width)
        self.h = int(height)
        self.transport = transport
        self._proc = None
        self._open()

    def _open(self):
        # 这些参数都要放在 -i 前面，才会作用在 RTSP 解复用/解码上：
        # max_delay / reorder_queue_size 默认是自动，RTP 重排队列能压住大约几百毫秒；
        # filter_threads 默认按 CPU 核数开，yuv→bgr 会拉起几十个缩放线程，在 8 核上
        # 只会互相抢，不会更快。fps_mode=passthrough 避免按固定帧率再攒一截。
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-threads", "1",
            "-filter_threads", "1",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-probesize", "32768", "-analyzeduration", "0", "-fpsprobesize", "0",
            "-max_delay", "0", "-reorder_queue_size", "0",
            "-rtsp_transport", self.transport,
            "-i", self.url, "-an",
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self._buf = bytearray()      # 未成帧的残余字节

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def read(self):
        """凑够一帧就返回：不预读、不攒整帧缓冲，帧到即用。"""
        n = self.w * self.h * 3
        try:
            while len(self._buf) < n:
                chunk = self._proc.stdout.read(n - len(self._buf))
                if not chunk:
                    return False, None
                self._buf.extend(chunk)
        except Exception:
            return False, None
        frame = np.frombuffer(bytes(self._buf[:n]), np.uint8).reshape(self.h, self.w, 3).copy()
        del self._buf[:n]
        return True, frame

    def release(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None


def _probe_size(url: str, transport: str = "tcp"):
    """用 ffprobe 取流分辨率，供裸帧管道使用。失败返回 (0,0)。"""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-rtsp_transport", transport,
             "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", url], timeout=15)
        parts = out.decode().strip().split(",")
        return int(parts[0]), int(parts[1])
    except Exception as e:
        print(f"[warn] ffprobe 取分辨率失败: {e}", flush=True)
        return 0, 0


def open_source(src: str, args):
    if src.startswith(("rtsp://", "http://", "https://")):
        kind = f"stream:{src.split('@')[-1]}"   # 日志里不打印账号密码
        if getattr(args, "reader", "ffmpeg") == "ffmpeg":
            w, h = _probe_size(src, args.rtsp_transport)
            if w and h:
                cap = FfmpegPipeCapture(src, w, h, args.rtsp_transport)
                if cap.isOpened():
                    print(f"[init] reader=ffmpeg-pipe  {w}x{h}  {args.rtsp_transport}", flush=True)
                    return cap, kind
                cap.release()
            print("[warn] ffmpeg-pipe 不可用，回退 OpenCV VideoCapture", flush=True)
        # 回退路径：OpenCV VideoCapture + FFmpeg 低缓冲参数
        opts = (f"rtsp_transport;{args.rtsp_transport}"
                "|timeout;5000000|rw_timeout;5000000"
                f"|max_delay;{args.rtsp_max_delay}"
                f"|buffer_size;{args.rtsp_buffer}")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[init] reader=opencv  ffmpeg opts: {opts}", flush=True)
    elif os.path.exists(src):
        cap = cv2.VideoCapture(src)
        kind = f"file:{src}"
    else:
        cap = cv2.VideoCapture(int(src))
        kind = f"camera:{src}"
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源 {src}")
    if hasattr(cap, "set"):          # FfmpegPipeCapture 无 set()
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap, kind


def main() -> int:
    args = build_parser().parse_args()
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    cfg = GestureConfig(
        control_margin=args.control_margin,
        mirror_x=not args.no_mirror,
        smooth_alpha=args.smooth,
    )
    print(f"[init] loading models ...", flush=True)
    rec = GestureRecognizer(args.palm_model, args.handpose_model, cfg,
                            palm_threshold=args.palm_threshold,
                            hand_conf=args.hand_conf,
                            max_hands=args.max_hands,
                            backend_id=args.backend, target_id=args.target,
                            palm_detect_every=args.palm_every,
                            palm_ondemand=not args.no_palm_ondemand)
    print(f"[init] models ready", flush=True)

    cap, kind = open_source(args.source, args)
    print(f"[init] video source: {kind}", flush=True)

    web_hub = None
    web_srv = None
    if args.web_port:
        web_hub = _MjpegHub()
        web_srv = ThreadingHTTPServer(("0.0.0.0", args.web_port), _make_web_handler(web_hub))
        threading.Thread(target=web_srv.serve_forever, daemon=True).start()
        print(f"[init] web preview: http://0.0.0.0:{args.web_port}/", flush=True)

    out_f = open(args.out, "a", encoding="utf-8", buffering=1) if args.out else None
    if out_f:
        print(f"[init] events -> {args.out}", flush=True)

    start = time.time()
    t_prev = start
    frames = 0
    fps_smooth = 0.0
    last_found = False
    ev_total = 0
    hand_frames = 0
    action_count = {}

    # 网络流用后台读帧线程（只留最新帧，防缓冲区积压延迟）；离线文件仍逐帧顺序处理
    is_file = kind.startswith("file")
    web_pub = None
    if web_hub is not None and not args.web_overlay:
        web_pub = _WebPublisher(web_hub, args.web_width)
    reader = None if is_file else _StreamReader(cap, web_pub)
    last_seq = 0

    print("[run] press Ctrl-C to stop", flush=True)
    last_frame = time.time()
    last_lag_warn = last_frame
    try:
        while _running:
            if reader is not None:
                ok, frame, seq = reader.latest()
                fresh = seq != last_seq
            else:
                ok, frame = cap.read()
                fresh = True
            if not ok or frame is None or not fresh:
                if is_file and (not ok or frame is None):
                    break
                # 流卡死看门狗：超过 --reopen-after 秒无新帧则重开视频源。
                # RTSP/HEVC 丢参考帧后 FFmpeg 可能恢复不了，重开是唯一可靠手段。
                if args.reopen_after > 0 and time.time() - last_frame > args.reopen_after:
                    print("[warn] 视频流超时，重开视频源 ...", flush=True)
                    for e in rec.engine.update(None, 1, 1):   # 补 RELEASE，防按键卡死
                        line = json.dumps(e, ensure_ascii=False, separators=(",", ":"))
                        if out_f:
                            out_f.write(line + "\n")
                        if not args.quiet:
                            print(line, flush=True)
                    if reader is not None:
                        reader.stop()
                    cap.release()
                    while _running:
                        try:
                            cap, kind = open_source(args.source, args)
                            print("[warn] 视频源已重开", flush=True)
                            break
                        except Exception as exc:
                            print(f"[warn] 重开失败: {exc}，2s 后重试", flush=True)
                            time.sleep(2)
                    if reader is not None:
                        reader = _StreamReader(cap, web_pub)
                        last_seq = 0
                    last_frame = time.time()
                time.sleep(0.02 if reader is not None else 0.05)
                continue

            if reader is not None:
                last_seq = seq
            last_frame = time.time()

            # 预览先行：推理前就把当前帧发出去，识别耗时不进预览链路。
            # 需要骨骼/HUD 叠加时用 --web-overlay（预览要等推理，+~0.1s）
            # 网络流的预览由 _WebPublisher 独立线程随采集即发，不走这里
            if web_hub is not None and not args.web_overlay and reader is None:
                _publish_web(web_hub, frame, args.web_width, fps_smooth, last_found)

            events, vis, found = rec.process(frame)
            last_found = found
            frames += 1
            hand_frames += int(found)
            ev_total += len(events)

            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps_smooth = inst if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst
            if web_pub is not None:
                web_pub.fps = fps_smooth
                web_pub.found = last_found

            for e in events:
                action_count[e["action"]] = action_count.get(e["action"], 0) + 1
                line = json.dumps(e, ensure_ascii=False, separators=(",", ":"))
                if out_f:
                    out_f.write(line + "\n")
                if not args.quiet:
                    print(line, flush=True)

            if args.stats and frames % 30 == 0:
                el = now - start
                rstat = ("" if reader is None else
                         f" read={reader.read_median:4.0f}ms backlog={reader.backlog*100:3.0f}%")
                print(f"[stat] {frames} frames  {el:6.1f}s  fps={fps_smooth:5.1f} "
                      f"hand={hand_frames/max(frames,1)*100:4.1f}%  events={ev_total}{rstat}",
                      flush=True)

            # 积压告警：解码跟不上码流时，读帧会"越来越快"（在追旧帧），
            # 延迟随使用时间增长。这里把它显性化，避免只看到"画面慢"却不知原因。
            if (reader is not None and args.lag_warn > 0
                    and now - last_lag_warn > args.lag_warn):
                last_lag_warn = now
                if reader.backlog > 0.5 and reader.read_median < 15.0:
                    print(f"[lag] 解码跟不上码流：read中位 {reader.read_median:.0f}ms "
                          f"（实时应≈40ms），近40帧中 {reader.backlog*100:.0f}% 在读旧帧。"
                          f"延迟在累积——降低摄像头帧率/分辨率，或改用硬解。", flush=True)

            if args.preview or (web_hub is not None and args.web_overlay):
                vis_frame = draw_hud(vis, events, fps_smooth, found,
                                     getattr(rec.engine, "last_pinch", None),
                                     getattr(rec.engine, "last_pinch_base", None),
                                     getattr(rec.engine, "last_tips", None),
                                     getattr(rec.engine, "last_tips_base", None))
                if args.preview:
                    cv2.imshow("KVM Gesture", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if web_hub is not None and args.web_overlay:
                    _publish_web(web_hub, vis_frame, args.web_width)

            if args.max_seconds and time.time() - start > args.max_seconds:
                break
    finally:
        if reader is not None:
            reader.stop()
        if web_pub is not None:
            web_pub.stop()
        cap.release()
        if out_f:
            out_f.close()
        if web_srv is not None:
            web_srv.shutdown()
        if args.preview:
            cv2.destroyAllWindows()

    el = time.time() - start
    print(f"\n[done] frames={frames} elapsed={el:.1f}s "
          f"avg_fps={frames/max(el,1e-6):.1f} hand_ratio={hand_frames/max(frames,1)*100:.1f}% "
          f"events={ev_total}", flush=True)
    if args.stats and action_count:
        print("[done] action breakdown: " +
              ", ".join(f"{k}={v}" for k, v in sorted(action_count.items())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
