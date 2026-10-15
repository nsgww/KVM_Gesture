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
    CLICK       短促捏合(<0.35s)，或 CSDN 方案的食指+中指并拢
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
    smooth_alpha: float = 0.35       # 指数平滑：越小越稳越滞后
    pinch_on: float = 0.55           # 捏合进入阈值（拇指尖-食指尖 / 掌宽）
    pinch_off: float = 0.70# 捏合退出阈值（迟滞，防抖动误触）
    tap_max_time: float = 0.35       # 短于此视为点击
    swipe_min_disp: float = 0.18# 归一化位移阈值
    swipe_window: float = 0.42       # 秒
    swipe_cooldown: float = 0.60     # 秒
    two_finger_on: float = 0.60      # CSDN 方案：食指+中指并拢阈值
    two_finger_off: float = 0.80
    zoom_min_delta: float = 0.02
    zoom_cooldown: float = 0.25


class GestureEngine:
    """关键点 → 事件流。逐帧调用，非线程安全。"""

    def __init__(self, cfg: Optional[GestureConfig] = None):
        self.cfg = cfg or GestureConfig()
        self.smooth: Optional[np.ndarray] = None
        self.grabbing = False
        self.pinch_latched = False
        self.two_finger_down = False
        self.grab_start_ts = 0.0
        self.track: List[dict] = []
        self.last_swipe_ts = 0.0
        self.last_zoom_ts = 0.0
        self.prev_pinch_dist: Optional[float] = None

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
            return self._on_hand_lost(ts)

        pts = hand_result[4:4 + 63].reshape(21, 3)
        fingers = self._finger_state(pts)
        pinch_dist = self._dist(pts[THUMB_TIP], pts[INDEX_TIP]) / fingers["scale"]
        two_dist = self._dist(pts[INDEX_TIP], pts[MIDDLE_TIP]) / fingers["scale"]

        events: List[dict] = []

        # 1) MOVE —— 食指为指针
        pointer = self._pointer_target(pts, fingers, w, h)
        if pointer is not None:
            self.track.append({"t": ts, "x": float(pointer[0]), "y": float(pointer[1])})
            while len(self.track) > 2 and ts - self.track[0]["t"] > self.cfg.swipe_window:
                self.track.pop(0)
            self.smooth = (pointer if self.smooth is None else
                           self.cfg.smooth_alpha * pointer
                           + (1 - self.cfg.smooth_alpha) * self.smooth)
            events.append(self._event(ACTION_MOVE, self.smooth, ts))

        # 2) GRAB / RELEASE / CLICK（迟滞 + 短按判定）
        if pinch_dist < self.cfg.pinch_on:
            self.pinch_latched = False
        elif pinch_dist > self.cfg.pinch_off:
            self.pinch_latched = True

        if not self.pinch_latched and not self.grabbing:
            self.grabbing = True
            self.grab_start_ts = ts
            events.append(self._event(ACTION_GRAB, self.smooth, ts,
                                      extra={"pinch": pinch_dist}))
        elif self.pinch_latched and self.grabbing:
            self.grabbing = False
            if ts - self.grab_start_ts < self.cfg.tap_max_time:
                events.append(self._event(ACTION_CLICK, self.smooth, ts))
            events.append(self._event(ACTION_RELEASE, self.smooth, ts))

        # 3) CSDN 方案：食指+中指同时伸出且并拢 → 点击
        two_ext = fingers["index"] and fingers["middle"]
        if two_ext and two_dist < self.cfg.two_finger_on:
            self.two_finger_down = True
        if (not two_ext or two_dist > self.cfg.two_finger_off) and self.two_finger_down:
            self.two_finger_down = False
            events.append(self._event(ACTION_CLICK, self.smooth, ts,
                                      extra={"gesture": "two_finger"}))

        # 4) SWIPE
        if len(self.track) >= 2 and ts - self.last_swipe_ts > self.cfg.swipe_cooldown:
            dx = self.track[-1]["x"] - self.track[0]["x"]
            dy = self.track[-1]["y"] - self.track[0]["y"]
            if abs(dx) > self.cfg.swipe_min_disp or abs(dy) > self.cfg.swipe_min_disp:
                act = (ACTION_SWIPE_RIGHT if dx > 0 else ACTION_SWIPE_LEFT) if abs(dx) > abs(dy) \
                    else (ACTION_SWIPE_DOWN if dy > 0 else ACTION_SWIPE_UP)
                events.append(self._event(act, self.smooth, ts, extra={"dx": dx, "dy": dy}))
                self.last_swipe_ts = ts
                self.track = self.track[-1:]

        # 5) ZOOM（基于捏合距离变化；双手缩放在此基础上扩展）
        events.extend(self._zoom_events(ts, pinch_dist))
        return events

    # -- 内部 -------------------------------------------------------------
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
        if ts - self.last_zoom_ts < self.cfg.zoom_cooldown:
            return events
        if abs(delta) < self.cfg.zoom_min_delta:
            return events
        self.last_zoom_ts = ts
        act = ACTION_ZOOM_IN if delta > 0 else ACTION_ZOOM_OUT
        events.append(self._event(act, self.smooth, ts, extra={"delta": delta}))
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