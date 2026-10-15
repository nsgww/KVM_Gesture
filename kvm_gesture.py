#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KVM 手势识别主程序（板端）。

用法
----
    # 摄像头 0，事件输出到 stdout
    python3 kvm_gesture.py --source 0

    # 指定输出文件 + 显示预览（需板子接显示器）
    python3 kvm_gesture.py --source 0 --out /tmp/events.jsonl --preview

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
import json
import os
import signal
import sys
import time

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
                   help="每N帧做一次 palm 检测，中间帧复用（板端默认2，实测提速1.5倍）")
    p.add_argument("--max-seconds", type=float, default=0, help="运行多少秒后自动退出，0=不限")
    p.add_argument("--reopen-after", type=float, default=5.0,
                   help="视频流多少秒无新帧后自动重开，0=禁用")
    p.add_argument("--stats", action="store_true", help="退出时打印性能统计")
    return p


def open_source(src: str, args):
    if src.startswith(("rtsp://", "http://", "https://")):
        # 网络摄像头（RTSP/HTTP 流）。FFMPEG 后端 + 最小缓冲降低延迟。
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                              "rtsp_transport;tcp|timeout;5000000|rw_timeout;5000000"
                              "|max_delay;200000|buffer_size;2048000")
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        kind = f"stream:{src.split('@')[-1]}"   # 日志里不打印账号密码
    elif os.path.exists(src):
        cap = cv2.VideoCapture(src)
        kind = f"file:{src}"
    else:
        cap = cv2.VideoCapture(int(src))
        kind = f"camera:{src}"
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源 {src}")
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
                            palm_detect_every=args.palm_every)
    print(f"[init] models ready", flush=True)

    cap, kind = open_source(args.source, args)
    print(f"[init] video source: {kind}", flush=True)

    out_f = open(args.out, "a", encoding="utf-8", buffering=1) if args.out else None
    if out_f:
        print(f"[init] events -> {args.out}", flush=True)

    start = time.time()
    t_prev = start
    frames = 0
    fps_smooth = 0.0
    ev_total = 0
    hand_frames = 0
    action_count = {}

    print("[run] press Ctrl-C to stop", flush=True)
    last_frame = time.time()
    try:
        while _running:
            ok, frame = cap.read()
            if not ok or frame is None:
                if kind.startswith("file"):
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
                    cap.release()
                    while _running:
                        try:
                            cap, kind = open_source(args.source, args)
                            print("[warn] 视频源已重开", flush=True)
                            break
                        except Exception as exc:
                            print(f"[warn] 重开失败: {exc}，2s 后重试", flush=True)
                            time.sleep(2)
                    last_frame = time.time()
                time.sleep(0.05)
                continue

            last_frame = time.time()

            events, vis, found = rec.process(frame)
            frames += 1
            hand_frames += int(found)
            ev_total += len(events)

            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps_smooth = inst if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst

            for e in events:
                action_count[e["action"]] = action_count.get(e["action"], 0) + 1
                line = json.dumps(e, ensure_ascii=False, separators=(",", ":"))
                if out_f:
                    out_f.write(line + "\n")
                if not args.quiet:
                    print(line, flush=True)

            if args.stats and frames % 30 == 0:
                el = now - start
                print(f"[stat] {frames} frames  {el:6.1f}s  fps={fps_smooth:5.1f} "
                      f"hand={hand_frames/max(frames,1)*100:4.1f}%  events={ev_total}",
                      flush=True)

            if args.preview:
                cv2.imshow("KVM Gesture", draw_hud(vis, events, fps_smooth, found))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_seconds and time.time() - start > args.max_seconds:
                break
    finally:
        cap.release()
        if out_f:
            out_f.close()
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