#!/usr/bin/env bash
# KVM 手势识别 —— 板端部署脚本（riscv64 / SpacemiT K3 / Bianbu）
#
# 用法：在板子上执行
#   bash deploy.sh
#
# 说明：板子为 riscv64 架构，MediaPipe 无官方 wheel，
#      因此使用 OpenCV 官方 opencv_zoo 的 MediaPipe 同款 ONNX 模型，
#      推理仅依赖 numpy + opencv-python（板子已预装）。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "==> 目录: $DIR"
echo "==> Python: $(python3 -V)"

echo "==> 检查依赖"
python3 - <<'PY'
import importlib, sys
missing = []
for m in ("numpy", "cv2"):
    try:
        mod = importlib.import_module(m)
        print(f"    {m} {getattr(mod,'__version__','?')}")
    except ImportError:
        missing.append(m)
if missing:
    print("缺少模块:", missing)
    print("请先安装: sudo apt install -y python3-numpy python3-opencv")
    sys.exit(1)
PY

echo "==> 检查模型"
for m in models/palm_detection_mediapipe_2023feb.onnx \
         models/handpose_estimation_mediapipe_2023feb.onnx; do
    if [ ! -f "$m" ]; then
        echo "缺少模型: $m"
        echo "请先上传 models/ 目录，或从以下地址下载："
        echo "  https://hf-mirror.com/opencv/opencv_zoo/resolve/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx"
        echo "  https://hf-mirror.com/opencv/opencv_zoo/resolve/main/models/hand_pose_estimation/handpose_estimation_mediapipe_2023feb.onnx"
        exit 1
    fi
    echo "    $(basename "$m") $(stat -c%s "$m") bytes"
done

echo "==> 自检（推理链路）"
python3 selftest.py

echo
echo "部署完成。运行："
echo "  python3 kvm_gesture.py --source 0 --out /tmp/events.jsonl --stats"
echo "  # 有图形界面时加 --preview 看骨架叠加"