# KVM_Gesture —— 隔空手势识别

## 这是什么

板端一体化程序：**摄像头 → 手部 21 关键点 → 手势状态机 → 事件流 → USB HID 注入**，
通过板子的 USB OTG 口模拟键鼠直接控制会议屏，不依赖外部设备。

- 识别侧（`kvm_gesture.py` + `gesture_core.py`）：输出「动作 + 坐标」事件流（JSONL）
- 注入侧（`hid_inject.py` + `setup_hid_gadget.sh`）：事件流翻译成 HID 报告写 `/dev/hidg0`

两侧可独立部署：只跑识别侧时事件流可落文件/管道，供其他注入方消费。

## 为什么不用 MediaPipe

开发板为 **SpacemiT K3 / riscv64**（Bianbu 4.0.1，Python 3.14.4）。
MediaPipe 官方没有 riscv64 wheel，无法 `pip install mediapipe`。

改用 **OpenCV 官方 opencv_zoo 发布的 MediaPipe 同款 ONNX 模型**，推理直接复用官方
参考实现 `mp_palmdet.py` / `mp_handpose.py`（预处理与后处理逐行对齐官方 demo，
不做二次实现以避免精度踩坑）：

| 模型 | 用途 | 输入 |
|---|---|---|
| `palm_detection_mediapipe_2023feb.onnx` | 手掌检测 | 1×192×192×3 |
| `handpose_estimation_mediapipe_2023feb.onnx` | 21 关键点回归 | 1×224×224×3 |

依赖仅 **numpy + opencv-python**（板子已预装），`onnxruntime` 为可选加速后端。

## 模型下载

ONNX 模型不入库（`.gitignore` 排除），从 OpenCV opencv_zoo 官方仓库下载到 `models/`：

```bash
cd models
wget https://github.com/opencv/opencv_zoo/raw/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx
wget https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx
# 可选：int8 量化版（更快；下载后用 --handpose-model 指定加载）
wget https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb_int8.onnx
```

模型主页（含 demo 与说明）：
- https://github.com/opencv/opencv_zoo/tree/main/models/palm_detection_mediapipe
- https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe

`models/mp_palmdet.py`、`models/mp_handpose.py` 为官方参考实现，已随仓库提供，无需下载。

## 目录结构

```
/opt/KVM_Gesture/
├── kvm_gesture.py          主程序（入口）
├── gesture_core.py         关键点推理封装 + 手势状态机
├── hid_inject.py           HID 注入（事件流 → /dev/hidg0）
├── deploy.sh               部署检查脚本
├── install_service.sh      安装 systemd 服务
├── setup_hid_gadget.sh     配置 USB HID gadget（ADB+键鼠复合设备）
├── tests/
│   ├── selftest.py         推理链路自检（无需摄像头）
│   ├── e2e_test.py         端到端验证（图片 → 事件流，产物 out_e2e.png 不入库）
│   └── data/               测试素材（testdata_pointing_up.jpg、synth_hand.mp4）
└── models/
    ├── palm_detection_mediapipe_2023feb.onnx        # 需下载，见上节
    ├── handpose_estimation_mediapipe_2023feb.onnx   # 需下载，见上节
    ├── mp_palmdet.py       官方参考实现（推理用）
    └── mp_handpose.py      官方参考实现（推理用）
```

## 快速开始

```bash
cd /opt/KVM_Gesture

# 1. 自检（不需要摄像头）
python3 tests/selftest.py

# 2. 实时运行，事件输出到文件
python3 kvm_gesture.py --source 0 --out /tmp/events.jsonl --stats

# 3. 有图形界面时叠加骨架预览
python3 kvm_gesture.py --source 0 --out /tmp/events.jsonl --preview

# 4. 用视频文件回放调参
python3 kvm_gesture.py --source test.mp4 --out /tmp/ev.jsonl --stats
```

退出：`Ctrl-C`，或 preview 窗口按 `q`。

## 事件协议

每行一个 JSON，写到 stdout 和 `--out` 指定文件：

```json
{"action":"MOVE","x":0.4367,"y":0.1185,"scale":1.0,"ts":1792036007023}
```

| 字段 | 含义 |
|---|---|
| `action` | 动作，见下表 |
| `x`, `y` | 控制区归一化坐标 0~1（已镜像、已去边缘留白） |
| `scale` | 缩放系数，恒为 1.0，ZOOM 语义预留 |
| `ts` | 毫秒时间戳 |

| action | 触发条件 | 底层应做 |
|---|---|---|
| `MOVE` | 每帧，食指（或掌心中点）作为指针；握拳时冻结 | 指针移动到 (x, y) |
| `GRAB` | 拇指+食指捏合开始（食指需伸出） | 鼠标左键按下 |
| `RELEASE` | 捏合结束，或**手离开画面** | 左键抬起 |
| `CLICK` | 食指尖碰中指尖（碰一次触发一次） | 单击 |
| `SWIPE_UP/DOWN/LEFT/RIGHT` | 0.42s 窗口内位移 > 0.18（抓取中/握拳不触发） | 上/下/左/右翻页、滚动 |
| `ZOOM_IN/OUT` | 握拳后手整体上移 / 下移（0.5s 窗口内位移 > 0.15） | 缩放 |

> 手丢失时一定补发 `RELEASE`，否则底层会卡在按下状态——这是容易踩的坑。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--source` | 0 | 摄像头编号或视频文件路径 |
| `--out` | 空 | 事件输出文件（JSONL，追加写） |
| `--quiet` | off | 不打印事件，只落文件 |
| `--preview` | off | 骨架叠加预览（需图形环境） |
| `--control-margin` | 0.10 | 控制区内缩比例，手贴边时防出画丢事件 |
| `--no-mirror` | off | 前置摄像头成镜像，默认开启符合直觉 |
| `--smooth` | 0.35 | 平滑系数，越小越稳越滞后 |
| `--max-hands` | 1 | 最大跟踪手数 |
| `--palm-threshold` | 0.4 | 手掌检测阈值 |
| `--hand-conf` | 0.5 | 关键点置信度阈值 |
| `--max-seconds` | 0 | 定时退出，0=不限 |
| `--stats` | off | 打印帧率、检出率、性能统计 |
| `--backend` / `--target` | 0 | cv.dnn 后端/目标，NPU 调优用 |

## 摄像头安装要求（重要）

- 装在**会议屏上沿朝人**，拍操作者双手，不需要拍到屏幕或会议室全景。
- 手距镜头 **0.6~1 m**；720p@30fps 足够。
- **固定机位 + 固定对焦**，关掉会议摄像头常见的「自动构图/自动取景/自动曝光」——
  这些一动，像素坐标与屏幕的映射就会漂移。
- 光照均匀、避免逆光（窗户别在操作者身后），背景避免肤色杂物。
- 取景框即控制区，四周留约 10% 边距；装好后做一次**四角标定**。

## 摄像头枚举

板子上可能有多个 `/dev/video*`（含 CSI/ISP 节点，非 UVC 摄像头）。
开机后先确认哪个是真实 USB 摄像头：

```bash
ls -l /dev/video*
v4l2-ctl --list-devices        # 若无该命令：apt install v4l-utils
for i in 0 1 2 3 4; do
  echo -n "video$i: "
  timeout 5 python3 -c "
import cv2,sys
c=cv2.VideoCapture($i)
ok,f=c.read()
print('OK' if ok and f is not None else 'fail', '' if f is None else f.shape)
"
done
```

识别到编号后传给 `--source`。

## 性能

riscv64 + cv.dnn CPU 后端，**单手 224×224 关键点 + 192×192 手掌检测，
在 x86 上约 70+ FPS**；板端 CPU 推理预计 15~30 FPS，满足手势交互要求。
若需更高帧率，可转 int8 量化模型并接 SpacemiT NPU：

```bash
# 板子自带 spacemit 版 onnxruntime
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# 显示 SpacemiTExecutionProvider 即 NPU 可用
```

## 注入侧（已实现，OTG HID）

识别侧之外，注入侧也已在板端实现并装机自启，不再依赖外部开发：

```
kvm_gesture.py → events.jsonl → hid_inject.py → /dev/hidg0 → USB 线 → 会议屏
```

| 文件/服务 | 作用 |
|---|---|
| `setup_hid_gadget.sh` | 在 adbd 的 gadget g1 中追加 `hid.usb0`，组成 **ADB+键鼠复合设备**（ADB 调试不丢）。HID 为单接口双 Report ID：ID1=绝对坐标鼠标+滚轮，ID2=键盘 |
| `hid_inject.py` | tail 事件文件（或 `--stdin` 管道），翻译成 HID 报告写 `/dev/hidg0`。无屏连接时报错等待不退出 |
| `kvm-hid-gadget.service` | 开机配置 HID gadget（After=adbd.service） |
| `kvm-hid-inject.service` | 开机启动注入程序 |

事件 → 屏端操作映射：`MOVE`→指针移动；`GRAB`/`RELEASE`→左键按下/抬起（即拖拽）；
`CLICK`→单击；`SWIPE_UP/DOWN`→滚轮；`SWIPE_LEFT/RIGHT`→键盘 →/←（左滑=下一页）；
`ZOOM_IN/OUT`→Ctrl+滚轮。

运维命令：

```bash
sudo bash setup_hid_gadget.sh status   # 查看 gadget 状态
journalctl -u kvm-hid-inject -f        # 注入侧日志
# 联调演练（不接屏，打印报告字节）：
echo '{"action":"MOVE","x":0.5,"y":0.5}' | python3 hid_inject.py --stdin --dry-run
```

坐标约定：识别侧输出归一化 0~1，注入侧乘以 32767 转 HID 绝对坐标，屏端自行适配分辨率。

## 已知限制

- `ZOOM` 为单手实现：握拳后手整体上移=放大、下移=缩小（握拳期间指针冻结，
  主机 Ctrl+滚轮作用在指针停着的位置）。双手张合缩放未实现（需 `max_hands=2`，
  帧率会明显下降）。
- HID 为鼠标+触摸双模同发：同一事件同时产出绝对坐标鼠标报告和 digitizer
  触点报告，主机认哪个用哪个（Windows 认鼠标；只认 digitizer 的会议大屏用触摸）。

## 联调建议

1. 识别侧先用 `--out` 落文件，注入侧读文件回放，确认协议理解一致。
2. 接屏验证：板子 OTG 口（Type-C）USB 线连屏的 USB-A 口，屏应识别出键鼠。
3. 双方确认：坐标绝对/相对、事件粒度、缩放语义、抓取语义、延迟要求。