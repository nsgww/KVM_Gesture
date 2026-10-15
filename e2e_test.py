# -*- coding: utf-8 -*-
"""端到端验证：真实手部图片 → palm → 21关键点 → 手势事件流。

同时输出可视化结果图，便于人工核对关键点是否正确。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gesture_core import GestureRecognizer, GestureConfig, draw_hud

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "testdata_pointing_up.jpg")
OUT = os.path.join(ROOT, "out_e2e.png")

rec = GestureRecognizer(cfg=GestureConfig(), palm_threshold=0.4, hand_conf=0.5)

img = cv2.imread(IMG)
print("image:", img.shape)
h, w = img.shape[:2]

palms = rec.palm_det.infer(img)
print("palms:", len(palms))
if len(palms):
    print("  best palm score:", round(float(palms[0][18]), 4),
          " box:", np.round(palms[0][:4], 1).tolist())

# 逐帧推进（同一张图重复喂，验证状态机与事件输出稳定性）
events = []
for i in range(12):
    ev, vis, found = rec.process(img)
    events.extend(ev)
print("hand found:", found)

acts = {}
for e in events:
    acts[e["action"]] = acts.get(e["action"], 0) + 1
print("event counts:", acts)
print("sample events:")
for e in events[:6]:
    print("   ", json.dumps(e, ensure_ascii=False))
print("last  event:", json.dumps(events[-1], ensure_ascii=False) if events else "NONE")

# 关键点合理性检查
if found:
    res = rec.hand_pose.infer(img, palms[0])
    pts = res[4:67].reshape(21, 3)
    wrist, idx_tip, thumb_tip = pts[0], pts[8], pts[4]
    palm_w = np.linalg.norm(pts[9] - pts[0])
    pinch = np.linalg.norm(idx_tip - thumb_tip) / max(palm_w, 1e-6)
    print("wrist:", np.round(wrist[:2], 1), " index_tip:", np.round(idx_tip[:2], 1))
    print("hand scale(px):", round(float(palm_w), 1), " pinch ratio:", round(float(pinch), 3))
    checks = []
    checks.append(("关键点在画面内", bool(0 <= wrist[0] <= w and 0 <= wrist[1] <= h)))
    checks.append(("手尺寸合理(40-400px)", 40 < palm_w < 400))
    checks.append(("置信度>0.5", float(res[131]) > 0.5))
    for name, ok in checks:
        print(f"   [{'OK' if ok else 'FAIL'}] {name}")

cv2.imwrite(OUT, draw_hud(vis, events[-6:], 30.0, found))
print("visualization ->", OUT)
print("== E2E DONE ==")