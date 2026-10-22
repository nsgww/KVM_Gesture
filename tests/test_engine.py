# -*- coding: utf-8 -*-
"""手势状态机：相对基线，不看手指直不直。不加载模型。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gesture_core import (  # noqa: E402
    ACTION_CLICK, ACTION_GRAB, ACTION_MOVE, ACTION_RELEASE,
    ACTION_SWIPE_LEFT, ACTION_ZOOM_IN, GestureConfig, GestureEngine,
)
from hid_inject import Injector, HidWriter  # noqa: E402

W, H = 640, 480
DT = 0.05


def pack(wrist, mcp, index, middle, thumb):
    pts = np.zeros((21, 3), np.float32)
    pts[0] = wrist
    pts[4] = thumb
    pts[8] = index
    pts[9] = mcp
    pts[12] = middle
    # 第二关节放在手腕和指尖之间，张开的手不会被当成握拳
    pts[6] = wrist + 0.55 * (index - wrist)
    pts[10] = wrist + 0.55 * (middle - wrist)
    up = np.array([0.0, -170.0, 0.0], np.float32)
    pts[16] = wrist + up + np.array([28.0, 0.0, 0.0], np.float32)
    pts[14] = wrist + 0.55 * up + np.array([28.0, 0.0, 0.0], np.float32)
    pts[20] = wrist + up + np.array([-28.0, 0.0, 0.0], np.float32)
    pts[18] = wrist + 0.55 * up + np.array([-28.0, 0.0, 0.0], np.float32)
    buf = np.zeros(67, np.float32)
    buf[4:67] = pts.reshape(-1)
    return buf


def fist_at(y):
    """四指收回，掌心在画面纵坐标 y。"""
    wrist = np.array([220.0, y, 0.0], np.float32)
    pts = np.zeros((21, 3), np.float32)
    pts[0] = wrist
    pts[9] = wrist + np.array([0.0, -36.0, 0.0], np.float32)
    pts[4] = wrist + np.array([16.0, -8.0, 0.0], np.float32)
    for pip_i, tip_i, dx in ((6, 8, 16.0), (10, 12, -16.0), (14, 16, 34.0), (18, 20, -34.0)):
        pts[pip_i] = wrist + np.array([dx, -88.0, 0.0], np.float32)
        pts[tip_i] = wrist + np.array([dx * 0.15, -10.0, 0.0], np.float32)
    buf = np.zeros(67, np.float32)
    buf[4:67] = pts.reshape(-1)
    return buf


def pose(scale=100.0, index=(320.0, 240.0, 0.0), middle_off=(50.0, 0.0, 0.0),
         thumb=(80.0, 300.0, 0.0), wrist=(200.0, 300.0, 0.0)):
    wrist = np.array(wrist, np.float32)
    mcp = wrist + np.array([0.0, -scale, 0.0], np.float32)
    index = np.array(index, np.float32)
    middle = index + np.array(middle_off, np.float32)
    return pack(wrist, mcp, index, middle, np.array(thumb, np.float32))


def feed(eng, n, **kw):
    ev = []
    t = kw.pop("t0", 0.0)
    for _ in range(n):
        ev.extend(eng.update(pose(**kw), W, H, t))
        t += DT
    return ev, t


def actions(ev, name):
    return [e for e in ev if e["action"] == name]


def test_pointer_always_moves():
    eng = GestureEngine()
    # 指尖在图像上离手腕很近（像指向镜头），仍然每帧 MOVE
    ev, _ = feed(eng, 8, index=(205.0, 295.0, 80.0), middle_off=(55.0, 0.0, 40.0))
    assert len(actions(ev, ACTION_MOVE)) == 8
    assert not actions(ev, ACTION_CLICK)


def test_tips_touch_click():
    eng = GestureEngine()
    ev, t = feed(eng, 9, middle_off=(50.0, 0.0, 0.0))          # 0.45s 张开，基线约 0.5
    ev2, _ = feed(eng, 8, t0=t, middle_off=(4.0, 0.0, 0.0))    # 靠近，不要求伸直
    clicks = actions(ev + ev2, ACTION_CLICK)
    assert len(clicks) == 1, clicks
    assert clicks[0]["how"] == "touch"
    assert not actions(ev + ev2, ACTION_GRAB)


def test_click_requires_fingers_up():
    """指尖贴着但手指没伸直（指向下方），不触发点击。"""
    eng = GestureEngine()
    # z 不再参与点击判定：指尖 2D 距离 4px 但手指朝下，不触发
    ev, _ = feed(eng, 8, index=(320.0, 360.0, 0.0), middle_off=(4.0, 0.0, 0.0))
    assert not actions(ev, ACTION_CLICK)


def test_spike_touch_clicks_once():
    eng = GestureEngine()
    ev = []
    t = 0.0
    for i in range(20):
        off = (4.0, 0.0, 0.0) if i == 12 else (50.0, 0.0, 0.0)
        # 指针每帧挪一点，停顿点击不会抢答；单帧贴近点一次（CSDN 方案即触即发）
        ev.extend(eng.update(pose(index=(300.0 + i * 8.0, 240.0, 0.0), middle_off=off), W, H, t))
        t += DT
    assert len(actions(ev, ACTION_CLICK)) == 1


def test_already_closed_clicks_immediately():
    eng = GestureEngine()
    # 手一入画两指就贴着（12px < 阈值 25.6px）：立即点一次，不用等停顿
    ev, _ = feed(eng, 20, middle_off=(12.0, 0.0, 0.0))
    clicks = actions(ev, ACTION_CLICK)
    assert len(clicks) == 1, clicks
    assert clicks[0]["how"] == "touch"


def test_dwell_not_repeated_until_pointer_moves():
    eng = GestureEngine()
    ev, t = feed(eng, 20, middle_off=(12.0, 0.0, 0.0))
    assert len(actions(ev, ACTION_CLICK)) == 1
    ev2, _ = feed(eng, 10, t0=t, middle_off=(12.0, 0.0, 0.0))
    assert not actions(ev2, ACTION_CLICK)


def test_grab_then_release_blocks_click():
    eng = GestureEngine()
    ev, t = feed(eng, 10)
    # 拇指捏合、食指中指张开：只按下，不点击
    ev2, t = feed(eng, 8, t0=t, thumb=(324.0, 240.0, 0.0))
    assert len(actions(ev + ev2, ACTION_GRAB)) == 1
    assert not actions(ev + ev2, ACTION_CLICK)
    ev3, _ = feed(eng, 8, t0=t, middle_off=(50.0, 0.0, 0.0), thumb=(80.0, 300.0, 0.0))
    rel = actions(ev3, ACTION_RELEASE)
    assert len(rel) == 1
    assert "reason" not in rel[0]


def test_hand_lost_releases():
    eng = GestureEngine()
    ev, t = feed(eng, 10)
    ev2, t = feed(eng, 8, t0=t, thumb=(324.0, 240.0, 0.0))
    assert actions(ev2, ACTION_GRAB)
    ev3 = eng.update(None, W, H, t)
    ev4 = eng.update(None, W, H, t + 0.7)
    rel = actions(ev3 + ev4, ACTION_RELEASE)
    assert len(rel) == 1 and rel[0]["reason"] == "hand_lost"


def test_swipe_needs_history_and_uses_wrist():
    eng = GestureEngine()
    ev = []
    t = 0.0
    for _ in range(12):
        ev.extend(eng.update(pose(), W, H, t))
        t += DT
    for i in range(8):
        x = 200.0 + (500.0 - 200.0) * (i / 7.0)
        ev.extend(eng.update(pose(wrist=(x, 300.0, 0.0)), W, H, t))
        t += DT
    sw = [e for e in ev if e["action"].startswith("SWIPE")]
    assert len(sw) == 1, sw
    assert sw[0]["action"] == ACTION_SWIPE_LEFT

    short = GestureEngine()
    a = short.update(pose(wrist=(200.0, 300.0, 0.0)), W, H, 0.0)
    b = short.update(pose(wrist=(520.0, 300.0, 0.0)), W, H, 0.05)
    assert not [e for e in a + b if e["action"].startswith("SWIPE")]


def test_zoom_when_fist_moves_up():
    eng = GestureEngine()
    ev = []
    t = 5.0
    ev.extend(eng.update(pose(), W, H, t))   # 先有一个张开的指针位置
    t += DT
    y0, y1 = 300.0, 190.0
    for i in range(12):
        y = y0 + (y1 - y0) * (i / 11.0)
        ev.extend(eng.update(fist_at(y), W, H, t))
        t += DT
    zooms = [e for e in ev if e["action"] == ACTION_ZOOM_IN]
    assert len(zooms) == 1, [e["action"] for e in ev if e["action"] != ACTION_MOVE]
    assert not [e for e in ev if e["action"] == ACTION_GRAB]
    # 握拳期间不再跟食指
    assert len(actions(ev, ACTION_MOVE)) == 1

    eng2 = GestureEngine()
    ev = []
    t = 5.0
    for i in range(12):
        y = 300.0 - 110.0 * (i / 11.0)
        ev.extend(eng2.update(pose(wrist=(200.0, y, 0.0)), W, H, t))
        t += DT
    assert not [e for e in ev if str(e["action"]).startswith("ZOOM")]


def test_hid_click_does_not_release_grab():
    hid = HidWriter("/dev/hidg0", dry_run=True)
    inj = Injector(hid)
    inj.grabbing = True
    inj.buttons = 1
    inj.handle({"action": "CLICK", "x": 0.4, "y": 0.4, "ts": int(time.time() * 1000)})
    assert inj.grabbing and inj.buttons == 1
    inj.grabbing = False
    inj.buttons = 0
    inj.handle({"action": "CLICK", "x": 0.4, "y": 0.4, "ts": int(time.time() * 1000)})
    assert inj.buttons == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("ALL PASSED", len(tests))
