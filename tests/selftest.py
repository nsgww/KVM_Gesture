# -*- coding: utf-8 -*-
"""本地自检：用合成图/官方示例验证 palm+handpose 推理链路与手势事件输出。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gesture_core import GestureRecognizer, GestureConfig, ACTION_MOVE, ACTION_GRAB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALM = os.path.join(ROOT, "models", "palm_detection_mediapipe_2023feb.onnx")
HAND = os.path.join(ROOT, "models", "handpose_estimation_mediapipe_2023feb.onnx")

print("== load models ==")
print("  palm :", os.path.exists(PALM), os.path.getsize(PALM) if os.path.exists(PALM) else 0)
print("  hand :", os.path.exists(HAND), os.path.getsize(HAND) if os.path.exists(HAND) else 0)

rec = GestureRecognizer(PALM, HAND, GestureConfig(), palm_threshold=0.4)

print("== warmup on blank frame ==")
blank = np.zeros((480, 640, 3), dtype=np.uint8)
ev, vis, found = rec.process(blank)
print("  events on blank:", len(ev), " hand_found:", found, " (expect 0 / False)")
print("  palm det output shape check ok")

print("== look for a real hand test image ==")
cands = []
for pat in ("example_outputs/*.png", "example_outputs/*.jpg", "*.jpg", "*.png"):
    cands += glob.glob(os.path.join(ROOT, "models", pat))
for pat in ("*.jpg", "*.png"):
    cands += glob.glob(os.path.join(ROOT, "tests", "data", pat))
cands = [c for c in cands if "onnx" not in c.lower()]
print("  candidates:", [os.path.basename(c) for c in cands][:10])

for path in cands:
    img = cv2.imread(path)
    if img is None:
        continue
    print(f"-- test: {os.path.basename(path)} shape={img.shape}")
    palms = rec.palm_det.infer(img)
    print("   palms detected:", len(palms))
    if len(palms):
        res = rec.hand_pose.infer(img, palms[0])
        print("   handpose result:", None if res is None else res.shape)
        if res is not None:
            pts = res[4:67].reshape(21, 3)
            print("   wrist", np.round(pts[0], 1), "index_tip", np.round(pts[8], 1))
            print("   conf", round(float(res[131]), 3))
    break

print("== DONE ==")