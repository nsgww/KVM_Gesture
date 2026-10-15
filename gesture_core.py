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
    MOVE        指针移动（食指为指针，位置=控制区归一化坐标）
    GRAB        捏合开始（拇指+食指）→ 底层按下
    RELEASE     捏合结束 / 手丢失     → 底层抬起
    CLICK       短促捏合(<0.35s)
    SWIPE_UP/DOWN/LEFT/RIGHT   窗口内快速位移
    ZOOM_IN/ZOOM_OUT          捏合距离变化（双手缩放在此扩展）
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
    pinch_on: float = 0.65           # 捏合进入阈值（拇指尖-食指尖 / 掌宽）
    pinch_off: float = 0.80          # 捏合退出阈值（迟滞，防抖动误触）
    grab_debounce: int = 2           # 捏合/张开需连续帧数确认，防单帧抖动误触
    grab_still: float = 0.12         # 触发抓取前 0.25s 内手需基本不动（挥手不误抓）
    tap_max_time: float = 0.5        # 短于此视为点击
    swipe_min_disp: float = 0.18     # 归一化位移阈值
    swipe_window: float = 0.42       # 秒
    swipe_cooldown: float = 0.60     # 秒
    swipe_reverse_block: float = 1.2 # 挥手后屏蔽反方向挥手的时长（防回位误判）
    zoom_min_delta: float = 0.05     # 每帧有效变化下限（中值滤波后）
    zoom_max_delta: float = 0.15     # 单帧超过此值=快速张合（松开动作），不计入缩放
    zoom_accum: float = 0.25         # 同向累计变化达到此值才触发一次缩放
    zoom_cooldown: float = 0.6
    lost_grace: float = 0.6          # 手消失宽限期（秒），短暂丢失不松开


class GestureEngine:
    """关键点 → 事件流。逐帧调用，非线程安全。"""

    def __init__(self, cfg: Optional[GestureConfig] = None):
        self.cfg = cfg or GestureConfig()
        self.smooth: Optional[np.ndarray] = None
        self.grabbing = False
        self.pinch_latched = False
        self.pinch_run = 0
        self.open_run = 0
        self.pinch_hist: List[float] = []
        self.absent_since: Optional[float] = None
        self.grab_start_ts = 0.0
        self.track: List[dict] = []
        self.last_swipe_ts = 0.0
        self.last_swipe_dir = ""     # 上次挥手方向，用于反向屏蔽
        self.last_zoom_ts = 0.0
        self.prev_pinch_dist: Optional[float] = None
        self._zoom_acc = 0.0

    # -- 几何 -------------------------------------------------------------
    @staticmethod
    def _dist(a: Sequence[float], b: Sequence[float]) -> float:
        return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))

    def _finger_state(self, pts: np.ndarray) -> dict:
        scale = self._dist(pts[WRIST], pts[MIDDLE_MCP]) or 1e-6

        def extended(tip, pip):
            return self._dist(pts[tip], pts[WRIST]) > self._dist(pts[pip], pts[WRIST]) * 1.12

        return {
            "thumb": self._dist(pts[THUMB_TIP], pts[INDEX_MCP]) > scale * 0.9,
            "index": extended(INDEX_TIP, INDEX_PIP),
            "middle": extended(MIDDLE_TIP, MIDDLE_PIP),
            "ring": extended(RING_TIP, RING_PIP),
            "pinky": extended(PINKY_TIP, PINKY_PIP),
            "scale": scale,
        }

    # -- 主入口 -----------------------------------------------------------
    def update(self, hand_result: Optional[np.ndarray], w: int, h: int,
               ts: Optional[float] = None) -> List[dict]:
        ts = ts if ts is not None else time.time()

        if hand_result is None:
            # 宽限期：短暂丢失（检测抖动）静默等待，超时才按手离开处理
            if self.absent_since is None:
                self.absent_since = ts
            if ts - self.absent_since >= self.cfg.lost_grace:
                return self._on_hand_lost(ts)
            return []
        self.absent_since = None

        pts = hand_result[4:4 + 63].reshape(21, 3)
        fingers = self._finger_state(pts)
        # 中值滤波：单帧关键点跳变（拇指尖误检）会被滤掉，防 ZOOM/GRAB 误触
        raw_pinch = self._dist(pts[THUMB_TIP], pts[INDEX_TIP]) / fingers["scale"]
        self.pinch_hist.append(raw_pinch)
        if len(self.pinch_hist) > 3:
            self.pinch_hist.pop(0)
        pinch_dist = sorted(self.pinch_hist)[len(self.pinch_hist) // 2]

        events: List[dict] = []

        # 1) MOVE —— 食指为指针
        pointer = self._pointer_target(pts, fingers, w, h)
        if pointer is not None:
            self.track.append({"t": ts, "x": float(pointer[0]), "y": float(pointer[1])})
            keep = self.cfg.swipe_window + 0.3     # 多留一段，供"挥前静止"判断
            while len(self.track) > 2 and ts - self.track[0]["t"] > keep:
                self.track.pop(0)
            self.smooth = (pointer if self.smooth is None else
                           self.cfg.smooth_alpha * pointer
                           + (1 - self.cfg.smooth_alpha) * self.smooth)
            events.append(self._event(ACTION_MOVE, self.smooth, ts))

        # 2) GRAB / RELEASE / CLICK（迟滞 + 连续帧确认 + 短按判定）
        if pinch_dist < self.cfg.pinch_on:
            self.pinch_latched = False
        elif pinch_dist > self.cfg.pinch_off:
            self.pinch_latched = True

        if not self.pinch_latched:
            self.pinch_run += 1
            self.open_run = 0
        else:
            self.open_run += 1
            self.pinch_run = 0

        if not self.grabbing and self.pinch_run >= self.cfg.grab_debounce \
                and self._still_enough(ts):
            self.grabbing = True
            self.grab_start_ts = ts
            events.append(self._event(ACTION_GRAB, self.smooth, ts,
                                      extra={"pinch": pinch_dist}))
        elif self.grabbing and self.open_run >= self.cfg.grab_debounce:
            self.grabbing = False
            if ts - self.grab_start_ts < self.cfg.tap_max_time:
                events.append(self._event(ACTION_CLICK, self.smooth, ts))
            events.append(self._event(ACTION_RELEASE, self.smooth, ts))

        # 4) SWIPE（抓取中不触发；要求挥动前静止，防回位动作误判）
        if (not self.grabbing and len(self.track) >= 2
                and ts - self.last_swipe_ts > self.cfg.swipe_cooldown):
            recent = [p for p in self.track if ts - p["t"] <= self.cfg.swipe_window]
            dx = recent[-1]["x"] - recent[0]["x"] if len(recent) >= 2 else 0.0
            dy = recent[-1]["y"] - recent[0]["y"] if len(recent) >= 2 else 0.0
            if (len(recent) >= 2 and self._at_rest_before(ts)
                    and (abs(dx) > self.cfg.swipe_min_disp
                         or abs(dy) > self.cfg.swipe_min_disp)):
                act = (ACTION_SWIPE_RIGHT if dx > 0 else ACTION_SWIPE_LEFT) if abs(dx) > abs(dy) \
                    else (ACTION_SWIPE_DOWN if dy > 0 else ACTION_SWIPE_UP)
                # 反向屏蔽：挥手后短时间内反方向动作视为回位，不上报
                opposite = {ACTION_SWIPE_UP: ACTION_SWIPE_DOWN, ACTION_SWIPE_DOWN: ACTION_SWIPE_UP,
                            ACTION_SWIPE_LEFT: ACTION_SWIPE_RIGHT, ACTION_SWIPE_RIGHT: ACTION_SWIPE_LEFT}
                if (act == opposite.get(self.last_swipe_dir)
                        and ts - self.last_swipe_ts < self.cfg.swipe_reverse_block):
                    self.track = self.track[-1:]
                    return events
                events.append(self._event(act, self.smooth, ts, extra={"dx": dx, "dy": dy}))
                self.last_swipe_ts = ts
                self.last_swipe_dir = act
                self.track = self.track[-1:]

        # 5) ZOOM（仅抓取中且手基本静止：捏住后张合手指才算缩放）
        if self.grabbing and self._stationary(ts):
            events.extend(self._zoom_events(ts, pinch_dist))
        else:
            self.prev_pinch_dist = pinch_dist
        return events

    # -- 内部 -------------------------------------------------------------
    def _at_rest_before(self, ts) -> bool:
        """挥动窗口之前 0.25s 内手是否基本静止（位移 < 0.06）。"""
        t1 = ts - self.cfg.swipe_window
        pre = [p for p in self.track if t1 - 0.25 <= p["t"] <= t1]
        if len(pre) < 2:
            return True                       # 历史不足（刚抬手）不拦
        dx, dy = self._track_disp(pre)
        return dx < 0.06 and dy < 0.06

    def _still_enough(self, ts) -> bool:
        """最近 0.25s 位移 < grab_still 才允许触发新抓取（挥手中不误抓）。"""
        recent = [p for p in self.track if ts - p["t"] <= 0.25]
        if len(recent) < 2:
            return True                       # 刚抬手历史不足，不拦
        dx, dy = self._track_disp(recent)
        return dx < self.cfg.grab_still and dy < self.cfg.grab_still

    def _stationary(self, ts) -> bool:
        """最近 0.3s 手是否基本不动（位移 < 0.05）。"""
        recent = [p for p in self.track if ts - p["t"] <= 0.3]
        if len(recent) < 2:
            return False
        dx, dy = self._track_disp(recent)
        return dx < 0.05 and dy < 0.05

    @staticmethod
    def _track_disp(pts) -> Tuple[float, float]:
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        return max(xs) - min(xs), max(ys) - min(ys)
    def _pointer_target(self, pts, fingers, w, h):
        m = self.cfg.control_margin
        tip = pts[INDEX_TIP] if fingers["index"] else (pts[MIDDLE_MCP] + pts[WRIST]) / 2
        nx = float(np.clip((tip[0] / w - m) / (1 - 2 * m), 0.0, 1.0))
        ny = float(np.clip((tip[1] / h - m) / (1 - 2 * m), 0.0, 1.0))
        if self.cfg.mirror_x:
            nx = 1.0 - nx
        return np.array([nx, ny], dtype=np.float32)

    def _zoom_events(self, ts, pinch_dist) -> List[dict]:
        events: List[dict] = []
        if self.prev_pinch_dist is None:
            self.prev_pinch_dist = pinch_dist
            return events
        delta = pinch_dist - self.prev_pinch_dist
        self.prev_pinch_dist = pinch_dist
        if abs(delta) > self.cfg.zoom_max_delta:
            self._zoom_acc = 0.0          # 快速张合（松开/捏死），清零不计
            return events
        if abs(delta) < self.cfg.zoom_min_delta:
            return events                 # 静止帧：保留累计，不清零
        if delta * self._zoom_acc < 0:
            self._zoom_acc = 0.0          # 换向清零
        self._zoom_acc += delta
        if ts - self.last_zoom_ts < self.cfg.zoom_cooldown:
            return events
        if abs(self._zoom_acc) < self.cfg.zoom_accum:
            return events
        act = ACTION_ZOOM_IN if self._zoom_acc > 0 else ACTION_ZOOM_OUT
        events.append(self._event(act, self.smooth, ts, extra={"delta": self._zoom_acc}))
        self._zoom_acc = 0.0
        self.last_zoom_ts = ts
        return events

    def _on_hand_lost(self, ts) -> List[dict]:
        """手离开画面必须补 RELEASE，否则底层会卡在按下状态。"""
        events: List[dict] = []
        if self.grabbing:
            self.grabbing = False
            if ts - self.grab_start_ts < self.cfg.tap_max_time:
                events.append(self._event(ACTION_CLICK, self.smooth, ts))
            events.append(self._event(ACTION_RELEASE, self.smooth, ts,
                                      extra={"reason": "hand_lost"}))
        self.smooth = None
        self.track = []
        self.prev_pinch_dist = None
        self.pinch_hist = []
        self.pinch_run = 0
        self.open_run = 0
        self.pinch_latched = False
        self.absent_since = None
        self._zoom_acc = 0.0
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
                 palm_detect_every: int = 1):
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
        self._frame_idx = 0
        self._cached_palms: Optional[np.ndarray] = None

    def _palms(self, frame: np.ndarray) -> np.ndarray:
        """按频率跑 palm 检测，其余帧复用缓存结果。"""
        need = (self._frame_idx % self.palm_detect_every == 0
                or self._cached_palms is None or len(self._cached_palms) == 0)
        if need:
            self._cached_palms = self.palm_det.infer(frame)
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
        return events, vis, hand_found

    @staticmethod
    def _draw(vis, res):
        pts = res[4:4 + 63].reshape(21, 3)
        xy = pts[:, :2].astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(vis, tuple(xy[a]), tuple(xy[b]), (255, 190, 110), 1)
        cv2.circle(vis, tuple(xy[INDEX_TIP]), 7, (0, 230, 0), -1)
        cv2.circle(vis, tuple(xy[THUMB_TIP]), 6, (0, 210, 255), -1)


def draw_hud(vis: np.ndarray, events: List[dict], fps: float, hand_found: bool):
    cv2.putText(vis, f"fps {fps:4.1f}  hand {'Y' if hand_found else 'N'}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if events:
        e = events[-1]
        pos = "-" if e["x"] is None else f"x={e['x']:.3f} y={e['y']:.3f}"
        cv2.putText(vis, f"{e['action']}  {pos}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis