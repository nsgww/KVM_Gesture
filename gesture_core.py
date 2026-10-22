# -*- coding: utf-8 -*-
"""手势识别核心（板端版）：palm 检测 + hand landmark 关键点 + 手势状态机。

为什么不用 MediaPipe
--------------------
板子为 SpacemiT K3 / riscv64（Bianbu 4.0.1），MediaPipe 官方无 riscv64 wheel，
无法 pip 安装。本模块改用 **OpenCV 官方 opencv_zoo 发布的 MediaPipe 同款 ONNX 模型**：

    palm_detection_mediapipe_2023feb.onnx        手掌检测 (1x192x192x3)
    handpose_estimation_mediapipe_2023feb.onnx   21 关键点   (1x224x224x3)

推理直接复用官方参考实现 mp_palmdet.py / mp_handpose.py（预处理、后处理逐行对齐），
保证关键点精度与官方 demo 一致，不做二次实现以避免踩坑。
依赖仅 numpy + opencv-python，onnxruntime 为可选 NPU 加速后端。

事件协议
--------
默认输出到 stdout（每行一个 JSON），可直接被注入侧消费：
    {"action":"MOVE","x":0.42,"y":0.55,"scale":1.0,"ts":1712345678901}

action 取值：
    MOVE        指针移动（始终跟食指尖，不看手指直不直）
    GRAB        拇指尖相对这只手自己的基线变近并保持 → 底层按下
    RELEASE     拇指食指回到基线 / 手丢失 → 底层抬起
    CLICK       食指尖与中指尖相对基线变近，或指针停顿；一次一次触发
    SWIPE_UP/DOWN/LEFT/RIGHT   手腕在窗口内快速位移（抓取中不触发）
    ZOOM_IN/ZOOM_OUT          握拳后手上移=放大，下移=缩小；握拳期间指针停住
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "models")
for _p in (_MODELS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mp_palmdet import MPPalmDet          # noqa: E402  官方实现
from mp_handpose import MPHandPose        # noqa: E402  官方实现

# ---------------------------------------------------------------------------
# 关键点索引（MediaPipe Hands 21 点拓扑）
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP = 4, 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

ACTION_MOVE = "MOVE"
ACTION_GRAB = "GRAB"
ACTION_RELEASE = "RELEASE"
ACTION_CLICK = "CLICK"
ACTION_SWIPE_UP = "SWIPE_UP"
ACTION_SWIPE_DOWN = "SWIPE_DOWN"
ACTION_SWIPE_LEFT = "SWIPE_LEFT"
ACTION_SWIPE_RIGHT = "SWIPE_RIGHT"
ACTION_ZOOM_IN = "ZOOM_IN"
ACTION_ZOOM_OUT = "ZOOM_OUT"


# ---------------------------------------------------------------------------
# 手势状态机
# ---------------------------------------------------------------------------
@dataclass
class GestureConfig:
    control_margin: float = 0.10     # 控制区内缩比例（边缘留白，防出画丢事件）
    mirror_x: bool = True            # 前置摄像头成镜像，符合人直觉
    smooth_alpha: float = 0.45       # 指数平滑：越小越稳越滞后
    pair_window: float = 1.0         # 张开时基线跟上的时间常数（秒）
    pair_ratio: float = 0.72         # 低于基线的这个比例算变近
    pair_min_drop: float = 0.08      # 同时至少靠近一个掌长的这个比例
    pair_hold: float = 0.05          # 变近或回到张开需稳住的时间（秒）
    pair_rearm: float = 0.90         # 回到基线的这个比例才算张开
    pair_arm: float = 0.35           # 看见这只手这么久之后才开始比基线
    grab_still: float = 0.12         # 触发抓取前手腕需基本不动（挥手不误抓）
    click_grab_block: float = 0.45   # 点击之后这段时间内不触发抓取（点击手
                                     # 势常带拇指收拢，否则点完立刻又按住）
    release_dwell_block: float = 0.8 # 松开抓取后这段时间内禁停顿点击
                                     # （松手时指针一停就被 dwell 补一枪误触）
    dwell_time: float = 0.60         # 指针停住多久算一次点击（兜底路径，
                                     # 主动点击走指尖触碰，这里不用太快）
    click_touch_px: float = 60.0     # 两指尖距离小于该像素值=点击（按 1000px
                                     # 画面宽度折算，对齐经典 MediaPipe 方案）
    click_release_ratio: float = 1.4 # 张到触发距离的这个倍数才解除点击锁定
    grab_touch_px: float = 40.0      # 拇指尖-食指尖距离小于该像素值=抓取按下
    grab_release_ratio: float = 1.6  # 分开到这个倍数才抬起（连续 2 帧防抖）
    dwell_radius: float = 0.045      # 停顿期间允许的指针抖动（归一化）
    dwell_suppress: float = 1.0      # 指尖点击之后这段时间不再用停顿补点
    swipe_min_disp: float = 0.18     # 归一化位移阈值
    swipe_window: float = 0.42       # 秒
    swipe_cooldown: float = 0.60     # 秒
    swipe_reverse_block: float = 1.2 # 挥手后屏蔽反方向挥手的时长（防回位误判）
    zoom_window: float = 0.6         # 握拳后看这段时间里的上下位移（秒）
    zoom_min_move: float = 0.12      # 画面高度比例，超过才缩放
    zoom_cooldown: float = 0.8       # 两次缩放的最小间隔（秒）
    lost_grace: float = 0.6          # 手消失宽限期（秒），短暂丢失不松开


class _ClosingGate:
    """一只手自己的指尖距离。张开时基线慢慢跟上，往近处收时基线停住。"""

    def __init__(self):
        self.raw: List[float] = []
        self.base: Optional[float] = None
        self.value: Optional[float] = None
        self.fast: Optional[float] = None
        self.seen_since: Optional[float] = None
        self.close_since: Optional[float] = None
        self.open_since: Optional[float] = None
        self.latched = False
        self.last_ts: Optional[float] = None

    def reset(self):
        self.__init__()

    def update(self, ts: float, dist: float, cfg: GestureConfig) -> float:
        self.raw.append(dist)
        if len(self.raw) > 3:
            self.raw.pop(0)
        d = sorted(self.raw)[len(self.raw) // 2]
        self.value = d
        # 合拢判定用最近 2 帧的较小值：中值滤波会把 1~2 帧的快速合拢抹掉，
        # 8fps 下等于"要慢吞吞并拢才认"。张开判定仍用中值，保持防抖。
        self.fast = min(self.raw[-2:])
        if self.seen_since is None:
            self.seen_since = ts
        dt = 0.0 if self.last_ts is None else ts - self.last_ts
        if dt > 0.2:
            # 丢帧的空档不能算进「稳住了」
            self.close_since = None
            self.open_since = None
        self.last_ts = ts
        if self.base is None:
            self.base = d
        elif not self.latched and d >= self.base * cfg.pair_rearm:
            # 锁存（已触发）期间冻结基线：否则长时间并拢会让基线坍缩到并拢
            # 距离，"张开"判定恒成立，每帧解锁又触发，变成连击
            alpha = min(1.0, max(dt, 0.0) / cfg.pair_window)
            self.base = self.base + alpha * (d - self.base)
        return d

    def armed(self, ts: float, cfg: GestureConfig) -> bool:
        return (self.seen_since is not None and self.base is not None
                and ts - self.seen_since >= cfg.pair_arm)

    def is_close(self, d: float, cfg: GestureConfig) -> bool:
        return (self.base is not None
                and d < self.base * cfg.pair_ratio
                and (self.base - d) > cfg.pair_min_drop)

    def is_open(self, d: float, cfg: GestureConfig) -> bool:
        return self.base is not None and d >= self.base * cfg.pair_rearm

    def held_close(self, ts: float, d: float, cfg: GestureConfig) -> bool:
        if not self.armed(ts, cfg) or not self.is_close(d, cfg):
            self.close_since = None
            return False
        if self.close_since is None:
            self.close_since = ts
        return ts - self.close_since >= cfg.pair_hold

    def held_open(self, ts: float, d: float, cfg: GestureConfig) -> bool:
        if not self.is_open(d, cfg):
            self.open_since = None
            return False
        if self.open_since is None:
            self.open_since = ts
        return ts - self.open_since >= cfg.pair_hold


class GestureEngine:
    """关键点 → 事件流。逐帧调用，非线程安全。"""

    def __init__(self, cfg: Optional[GestureConfig] = None):
        self.cfg = cfg or GestureConfig()
        self.smooth: Optional[np.ndarray] = None
        self.grabbing = False
        self.absent_since: Optional[float] = None
        self.track: List[dict] = []          # 食指尖：指针、停顿、缩放时是否还在动
        self.wrist_track: List[dict] = []    # 手腕：挥手
        self._fist_track: List[dict] = []    # 握拳期间的掌心纵向位置
        self.last_swipe_ts = 0.0
        self.last_swipe_dir = ""
        self.last_zoom_ts = 0.0
        self.last_pinch: Optional[float] = None
        self.last_pinch_base: Optional[float] = None
        self.last_tips: Optional[float] = None
        self.last_tips_base: Optional[float] = None
        self.last_fist = False
        self._pinch = _ClosingGate()         # 拇指尖–食指尖
        self._tips = _ClosingGate()          # 食指尖–中指尖
        self._dwell_latched = False
        self._dwell_block_until = 0.0
        self._grab_block_until = 0.0
        self._click_latched = False
        self._pinch_open_frames = 0
        self._pinch_touching = False
        self._tips_hist: List[dict] = []

    # -- 几何 -------------------------------------------------------------
    @staticmethod
    def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
        return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))

    @staticmethod
    def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
        return float(math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2))

    def _extended(self, pts: np.ndarray, tip: int, pip: int) -> bool:
        """指尖比第二关节更远离手腕 = 手指伸直。方向无关（按到手腕距离比）。"""
        return self._dist2(pts[tip], pts[WRIST]) > self._dist2(pts[pip], pts[WRIST]) * 1.12

    def _is_fist(self, pts: np.ndarray) -> bool:
        """四指尖都收回到比第二关节更靠近手腕。用于缩放，不用于点击。"""
        return not (self._extended(pts, INDEX_TIP, INDEX_PIP)
                    or self._extended(pts, MIDDLE_TIP, MIDDLE_PIP)
                    or self._extended(pts, RING_TIP, RING_PIP)
                    or self._extended(pts, PINKY_TIP, PINKY_PIP))

    # -- 主入口 -----------------------------------------------------------
    def update(self, hand_result: Optional[np.ndarray], w: int, h: int,
               ts: Optional[float] = None) -> List[dict]:
        ts = ts if ts is not None else time.time()

        if hand_result is None:
            # 宽限期：短暂丢失（检测抖动）静默等待，超时才按手离开处理
            if self.absent_since is None:
                self.absent_since = ts
            if ts - self.absent_since >= self.cfg.lost_grace:
                self.last_pinch = None
                self.last_pinch_base = None
                self.last_tips = None
                self.last_tips_base = None
                return self._on_hand_lost(ts)
            return []
        self.absent_since = None

        pts = hand_result[4:4 + 63].reshape(21, 3)
        events: List[dict] = []
        cfg = self.cfg
        # 握拳是缩放：指针停在握上之前的位置，整手上移放大、下移缩小。
        # 这只手不再去比指尖距离，避免握拳被当成拖拽或点击。
        if self._is_fist(pts):
            self.last_fist = True
            if self.grabbing:
                self.grabbing = False
                self._pinch.latched = False
                self._pinch.close_since = None
                events.append(self._event(ACTION_RELEASE, self.smooth, ts))
            ref_y = float((pts[WRIST][1] + pts[MIDDLE_MCP][1]) / 2) / max(h, 1)
            self._fist_track.append({"t": ts, "y": ref_y})
            self._trim(self._fist_track, ts, cfg.zoom_window)
            if (ts - self.last_zoom_ts > cfg.zoom_cooldown
                    and len(self._fist_track) >= 2):
                dy = self._fist_track[-1]["y"] - self._fist_track[0]["y"]
                if abs(dy) >= cfg.zoom_min_move:
                    act = ACTION_ZOOM_IN if dy < 0 else ACTION_ZOOM_OUT
                    events.append(self._event(act, self.smooth, ts, extra={"dy": dy}))
                    self.last_zoom_ts = ts
                    self._fist_track = self._fist_track[-1:]
            return events
        self.last_fist = False
        self._fist_track = []

        scale = self._dist3(pts[WRIST], pts[MIDDLE_MCP]) or 1e-6
        pinch_d = self._pinch.update(
            ts, self._dist3(pts[THUMB_TIP], pts[INDEX_TIP]) / scale, self.cfg)
        tips_d = self._tips.update(
            ts, self._dist3(pts[INDEX_TIP], pts[MIDDLE_TIP]) / scale, self.cfg)
        # 点击判定用的原始 2D 像素距离（参考经典 MediaPipe 虚拟鼠标方案：
        # 两指伸直 + 指尖距离 < 40px@1000px 宽 → 点击），不做基线/滤波
        tips_px = self._dist2(pts[INDEX_TIP], pts[MIDDLE_TIP])
        touch_px = cfg.click_touch_px * w / 1000.0
        release_px = touch_px * cfg.click_release_ratio
        # 抓取同样是绝对像素判定：拇指尖-食指尖直接碰上才算，碰中指不算
        pinch_px = self._dist2(pts[THUMB_TIP], pts[INDEX_TIP])
        grab_touch = cfg.grab_touch_px * w / 1000.0
        grab_release = grab_touch * cfg.grab_release_ratio
        self._pinch_touching = pinch_px < grab_touch
        self.last_pinch = pinch_d
        self.last_pinch_base = self._pinch.base
        self.last_tips = tips_d
        self.last_tips_base = self._tips.base
        self._tips_hist.append({"t": ts, "d": tips_d})
        self._trim(self._tips_hist, ts, self.cfg.dwell_time + 0.2)

        # 1) MOVE —— 手张开时跟食指尖。握拳时上面已经返回，指针保持不动
        pointer = self._norm_point(pts[INDEX_TIP], w, h)
        wrist = self._norm_point(pts[WRIST], w, h)
        keep = max(cfg.swipe_window, cfg.dwell_time, cfg.zoom_window) + 0.4
        self.track.append({"t": ts, "x": float(pointer[0]), "y": float(pointer[1])})
        self.wrist_track.append({"t": ts, "x": float(wrist[0]), "y": float(wrist[1])})
        self._trim(self.track, ts, keep)
        self._trim(self.wrist_track, ts, keep)
        self.smooth = (pointer if self.smooth is None else
                       cfg.smooth_alpha * pointer + (1 - cfg.smooth_alpha) * self.smooth)
        events.append(self._event(ACTION_MOVE, self.smooth, ts))

        # 2) GRAB / RELEASE —— 拇指尖与食指尖直接碰上=按下，分开=抬起
        #    （绝对像素距离，与点击同一思路；碰中指不算，只认拇指-食指）。
        #    两指合拢（点击手势）期间、以及点击后的冷却窗内不判抓取：
        #    并拢食指中指时拇指常跟着收拢，否则 CLICK 后必跟一次 GRAB，
        #    主机会把点击吃成"按下不放"。
        if (not self.grabbing
                and ts >= self._grab_block_until
                and not self._click_latched
                and tips_px >= touch_px
                and pinch_px < grab_touch
                and self._quiet(self.wrist_track, ts, 0.25, cfg.grab_still, 0.15)):
            self.grabbing = True
            self._pinch_open_frames = 0
            events.append(self._event(ACTION_GRAB, self.smooth, ts,
                                      extra={"pinch": pinch_px}))
        elif self.grabbing:
            # 连续 2 帧分开才抬起：拖动中关键点抖动一帧不至于掉键
            if pinch_px > grab_release:
                self._pinch_open_frames += 1
            else:
                self._pinch_open_frames = 0
            if self._pinch_open_frames >= 2:
                self.grabbing = False
                events.append(self._event(ACTION_RELEASE, self.smooth, ts))
                # 松手后 dwell 重新计时：否则拖拽结束的瞬间指针一停，
                # 停顿检测会立刻把"停住"误判成一次点击（松手即点误触）
                self._dwell_block_until = ts + cfg.release_dwell_block

        # 3) CLICK —— 经典 MediaPipe 虚拟鼠标方案：
        #    食指中指都伸直（指尖比第二关节更靠上）且两指尖距离够近 → 点击。
        #    锁存一次，张开后解除，不连击。抓取按下期间不发点击。
        # 伸直判定留 4% 画面高度的余量：捏合时手指自然微弯也算伸直；
        # 拳头里指尖比第二关节低 70+ 像素，余量不会把它误判成伸直
        up_slack = h * 0.04
        idx_up = pts[INDEX_TIP][1] < pts[INDEX_PIP][1] + up_slack
        mid_up = pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1] + up_slack
        if self.grabbing:
            pass
        elif (not self._click_latched and idx_up and mid_up
              and tips_px < touch_px):
            self._click_latched = True
            self._dwell_block_until = ts + cfg.dwell_suppress
            self._grab_block_until = ts + cfg.click_grab_block
            pos = self._lagged_pos(ts, 0.20)
            events.append(self._event(
                ACTION_CLICK, pos if pos is not None else self.smooth, ts,
                extra={"how": "touch"}))
        elif self._click_latched and tips_px > release_px:
            self._click_latched = False

        # 4) 停顿点击 —— 手指并拢张不开、距离下不去时，指针停住也算一次
        if self._dwell_ready(ts):
            self._dwell_latched = True
            events.append(self._event(ACTION_CLICK, self.smooth, ts, extra={"how": "dwell"}))
        elif self._dwell_latched and not self._quiet(
                self.track, ts, 0.25, cfg.dwell_radius, 0.0):
            self._dwell_latched = False

        # 5) SWIPE —— 看手腕。历史不够不触发。抓取中不触发
        if (not self.grabbing and ts - self.last_swipe_ts > cfg.swipe_cooldown):
            recent = [p for p in self.wrist_track if ts - p["t"] <= cfg.swipe_window]
            if (len(recent) >= 3
                    and recent[-1]["t"] - recent[0]["t"] >= 0.15
                    and self._at_rest_before(ts)):
                dx = recent[-1]["x"] - recent[0]["x"]
                dy = recent[-1]["y"] - recent[0]["y"]
                if abs(dx) > cfg.swipe_min_disp or abs(dy) > cfg.swipe_min_disp:
                    act = ((ACTION_SWIPE_RIGHT if dx > 0 else ACTION_SWIPE_LEFT)
                           if abs(dx) > abs(dy)
                           else (ACTION_SWIPE_DOWN if dy > 0 else ACTION_SWIPE_UP))
                    opposite = {
                        ACTION_SWIPE_UP: ACTION_SWIPE_DOWN, ACTION_SWIPE_DOWN: ACTION_SWIPE_UP,
                        ACTION_SWIPE_LEFT: ACTION_SWIPE_RIGHT, ACTION_SWIPE_RIGHT: ACTION_SWIPE_LEFT}
                    blocked = (act == opposite.get(self.last_swipe_dir)
                               and ts - self.last_swipe_ts < cfg.swipe_reverse_block)
                    self.wrist_track = self.wrist_track[-1:]
                    if not blocked:
                        events.append(self._event(act, self.smooth, ts, extra={"dx": dx, "dy": dy}))
                        self.last_swipe_ts = ts
                        self.last_swipe_dir = act
        return events

    # -- 内部 -------------------------------------------------------------
    def _dwell_ready(self, ts: float) -> bool:
        cfg = self.cfg
        # 停顿点击默认关闭：手停住就补一枪是误触主因（日志里误触 CLICK 几乎全是 dwell）
        if not getattr(cfg, "dwell_enabled", False):
            return False
        if self.grabbing or self._dwell_latched or self._click_latched:
            return False
        # 拇指正在捏合（抓取手势进行中）时停顿点击不抢答
        if self._pinch_touching:
            return False
        if ts < self._dwell_block_until:
            return False
        pts = [p for p in self.track if p["t"] >= self._dwell_block_until and ts - p["t"] <= cfg.dwell_time]
        if len(pts) < 2 or pts[-1]["t"] - pts[0]["t"] < cfg.dwell_time * 0.9:
            return False
        dx, dy = self._track_disp(pts)
        if dx >= cfg.dwell_radius or dy >= cfg.dwell_radius:
            return False
        recent = [p["d"] for p in self._tips_hist if p["t"] >= pts[0]["t"]]
        if len(recent) >= 2 and max(recent) - min(recent) > 0.10:
            return False
        return True

    def _at_rest_before(self, ts) -> bool:
        """挥动窗口之前 0.25s 内手腕是否基本静止。历史不够则不挥。"""
        t1 = ts - self.cfg.swipe_window
        pre = [p for p in self.wrist_track if t1 - 0.25 <= p["t"] <= t1]
        if len(pre) < 2:
            return False
        dx, dy = self._track_disp(pre)
        return dx < 0.06 and dy < 0.06

    def _quiet(self, track, ts, window, limit, min_span) -> bool:
        recent = [p for p in track if ts - p["t"] <= window]
        if len(recent) < 2 or recent[-1]["t"] - recent[0]["t"] < min_span:
            return False
        dx, dy = self._track_disp(recent)
        return dx < limit and dy < limit

    @staticmethod
    def _trim(hist, ts, keep):
        while len(hist) > 2 and ts - hist[0]["t"] > keep:
            hist.pop(0)

    @staticmethod
    def _track_disp(pts) -> Tuple[float, float]:
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        return max(xs) - min(xs), max(ys) - min(ys)

    def _lagged_pos(self, ts, lag: float) -> Optional[np.ndarray]:
        """ts-lag 时刻的指针位置，避开两指靠近时食指尖的跳动。"""
        if not self.track:
            return None
        ref = [p for p in self.track if p["t"] <= ts - lag]
        p = ref[-1] if ref else self.track[0]
        return np.array([p["x"], p["y"]], dtype=np.float32)

    def _norm_point(self, pt, w, h):
        m = self.cfg.control_margin
        nx = float(np.clip((float(pt[0]) / w - m) / (1 - 2 * m), 0.0, 1.0))
        ny = float(np.clip((float(pt[1]) / h - m) / (1 - 2 * m), 0.0, 1.0))
        if self.cfg.mirror_x:
            nx = 1.0 - nx
        return np.array([nx, ny], dtype=np.float32)

    def _on_hand_lost(self, ts) -> List[dict]:
        """手离开画面必须补 RELEASE，否则底层会卡在按下状态。"""
        events: List[dict] = []
        if self.grabbing:
            self.grabbing = False
            events.append(self._event(ACTION_RELEASE, self.smooth, ts,
                                      extra={"reason": "hand_lost"}))
        self.smooth = None
        self.track = []
        self.wrist_track = []
        self._fist_track = []
        self._tips_hist = []
        self._pinch.reset()
        self._tips.reset()
        self._dwell_latched = False
        self._click_latched = False
        self._pinch_open_frames = 0
        self._pinch_touching = False
        self.absent_since = None
        return events

    def _event(self, action, xy, ts, extra=None) -> dict:
        ev = {
            "action": action,
            "x": round(float(xy[0]), 4) if xy is not None else None,
            "y": round(float(xy[1]), 4) if xy is not None else None,
            "scale": 1.0,
            "ts": int(ts * 1000),
        }
        if extra:
            for k, v in extra.items():
                ev[k] = round(float(v), 4) if isinstance(v, float) else v
        return ev


# ---------------------------------------------------------------------------
# 识别器：官方检测/关键点模型 + 手势状态机
# ---------------------------------------------------------------------------
class GestureRecognizer:
    """palm 检测 + 关键点 + 手势状态机。

    性能策略（板端实测 palm 检测占 2/3 耗时）：
      * palm_detect_every=N —— palm 检测降频，中间帧复用上一帧 palm box。
        手在画面里的移动速度远低于检测频率，2~3 帧复用对精度无感知，
        帧率可从 5.3 提升到 8~9 fps。
      * 只在关键帧跑 palm 检测，非关键帧直接用缓存的 palm 结果做关键点回归。
    """

    def __init__(self, palm_model: Optional[str] = None,
                 handpose_model: Optional[str] = None,
                 cfg: Optional[GestureConfig] = None,
                 palm_threshold: float = 0.4, hand_conf: float = 0.5,
                 max_hands: int = 1, backend_id: int = 0, target_id: int = 0,
                 palm_detect_every: int = 1, palm_ondemand: bool = False):
        palm_model = palm_model or os.path.join(_MODELS, "palm_detection_mediapipe_2023feb.onnx")
        handpose_model = handpose_model or os.path.join(_MODELS, "handpose_estimation_mediapipe_2023feb.onnx")
        for p in (palm_model, handpose_model):
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少模型文件: {p}")
        self.palm_det = MPPalmDet(palm_model, scoreThreshold=palm_threshold,
                                  backendId=backend_id, targetId=target_id)
        self.hand_pose = MPHandPose(handpose_model, confThreshold=hand_conf,
                                    backendId=backend_id, targetId=target_id)
        self.engine = GestureEngine(cfg)
        self.max_hands = max_hands
        self.palm_detect_every = max(1, int(palm_detect_every))
        self.palm_ondemand = bool(palm_ondemand)
        self._frame_idx = 0
        self._cached_palms: Optional[np.ndarray] = None
        self._need_palm = True          # 手丢失后强制重新检测

    def _palms(self, frame: np.ndarray) -> np.ndarray:
        """palm 检测调度。

        按需模式（palm_ondemand）：只在"手丢了"或每隔 N 帧做一次兜底检测，
        其余帧完全复用上一次的 palm box——省掉 121ms 的检测开销，
        帧率由 handpose 单项决定（约 12~14fps）。
        固定频率模式：每 N 帧检测一次。
        """
        if self.palm_ondemand:
            need = (self._need_palm or self._cached_palms is None
                    or len(self._cached_palms) == 0
                    or self._frame_idx % max(self.palm_detect_every * 4, 20) == 0)
        else:
            need = (self._frame_idx % self.palm_detect_every == 0
                    or self._cached_palms is None or len(self._cached_palms) == 0)
        if need:
            self._cached_palms = self.palm_det.infer(frame)
            self._need_palm = False
        self._frame_idx += 1
        return self._cached_palms if self._cached_palms is not None else np.empty((0, 19))

    def process(self, frame_bgr: np.ndarray) -> Tuple[List[dict], np.ndarray, bool]:
        """返回 (事件列表, 标注帧, 是否检测到手)。"""
        h, w = frame_bgr.shape[:2]
        palms = self._palms(frame_bgr)
        events: List[dict] = []
        vis = frame_bgr.copy()
        hand_found = False

        if len(palms):
            order = np.argsort(-palms[:, 18])[: self.max_hands]
            for idx in order:
                res = self.hand_pose.infer(frame_bgr, palms[idx])
                if res is None:
                    continue
                hand_found = True
                events.extend(self.engine.update(res, w, h))
                self._draw(vis, res)
        if not hand_found:
            events.extend(self.engine.update(None, w, h))
            # 按需模式：关键点没检出说明 palm box 可能已失效，下一帧强制重检
            if self.palm_ondemand:
                self._need_palm = True
        return events, vis, hand_found

    @staticmethod
    def _draw(vis, res):
        pts = res[4:4 + 63].reshape(21, 3)
        xy = pts[:, :2].astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(vis, tuple(xy[a]), tuple(xy[b]), (255, 190, 110), 1)
        for p in xy:
            cv2.circle(vis, tuple(p), 3, (255, 255, 255), -1)   # 全部 21 个关节
        cv2.circle(vis, tuple(xy[INDEX_TIP]), 7, (0, 230, 0), -1)
        cv2.circle(vis, tuple(xy[THUMB_TIP]), 6, (0, 210, 255), -1)
        cv2.circle(vis, tuple(xy[MIDDLE_TIP]), 6, (255, 0, 255), -1)


def draw_hud(vis: np.ndarray, events: List[dict], fps: float, hand_found: bool,
             pinch: Optional[float] = None, pinch_base: Optional[float] = None,
             tips: Optional[float] = None, tips_base: Optional[float] = None):
    cv2.putText(vis, f"fps {fps:4.1f}  hand {'Y' if hand_found else 'N'}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if tips is not None and tips_base:
        close = tips < tips_base * 0.70
        cv2.putText(vis, f"tips {tips:.2f}/{tips_base:.2f}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if close else (255, 255, 255), 2)
    if pinch is not None and pinch_base:
        close = pinch < pinch_base * 0.70
        cv2.putText(vis, f"pinch {pinch:.2f}/{pinch_base:.2f}",
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if close else (255, 255, 255), 2)
    if events:
        e = events[-1]
        pos = "-" if e["x"] is None else f"x={e['x']:.3f} y={e['y']:.3f}"
        cv2.putText(vis, f"{e['action']}  {pos}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis
    dwell_enabled: bool = False      # 停顿点击总开关：默认关，只认食指+中指相碰的主动点击
