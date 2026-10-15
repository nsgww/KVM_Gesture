#!/usr/bin/env bash
# KVM 手势 —— USB HID gadget 配置（复合设备：在 adbd 的 g1 中追加 HID 接口）
#
# 原理：板子 OTG 口的 gadget g1 由 adbd.service 创建（ffs.adb）。本脚本在 g1 里
# 追加一个 hid.usb0 function 并重新绑定，使板子对屏同时呈现为 ADB 设备 + 键鼠。
# ADB 调试不中断（重新绑定瞬间会断一次，随后自动恢复）。
#
# 用法（需要 root）：
#   sudo bash setup_hid_gadget.sh start    # 加入 HID 接口并绑定
#   sudo bash setup_hid_gadget.sh stop     # 移除 HID 接口
#   sudo bash setup_hid_gadget.sh status
#
# HID 描述符为单接口双 Report ID：
#   ID=1 鼠标：绝对坐标 X/Y (0~32767) + 5 键 + 滚轮
#   ID=2 键盘：标准 boot 键盘（修饰键 + 6 键）
set -euo pipefail

G=/sys/kernel/config/usb_gadget/g1
UDC=$(ls /sys/class/udc | head -1)
HID_FUNC=hid.usb0

die() { echo "ERROR: $*" >&2; exit 1; }

[ -d "$G" ] || die "$G 不存在。g1 由 adbd.service 创建，请先确认 adbd.service 已启动。"

unbind() {
    if [ -s "$G/UDC" ]; then
        echo "" > "$G/UDC"
        sleep 0.3
    fi
}

rebind() {
    echo "$UDC" > "$G/UDC"
    sleep 0.5
}

start() {
    if [ -d "$G/functions/$HID_FUNC" ]; then
        echo "[skip] $HID_FUNC 已存在"
        [ -e /dev/hidg0 ] && echo "[ok] /dev/hidg0 就绪" || rebind
        return 0
    fi

    unbind

    mkdir "$G/functions/$HID_FUNC"
    # 非 boot 协议（描述符自带 Report ID，复合鼠标+键盘）
    echo 0 > "$G/functions/$HID_FUNC/protocol"
    echo 0 > "$G/functions/$HID_FUNC/subclass"
    # 最长报告 = 键盘报告 9 字节（ID + modifier + reserved + 6 keys）
    echo 9 > "$G/functions/$HID_FUNC/report_length"

    # 用 python3 写二进制报告描述符（sh 的 printf 转义易错）
    python3 - "$G/functions/$HID_FUNC/report_desc" <<'PY'
import sys
desc = bytes([
    # ---- Report ID 1: 绝对坐标鼠标 + 滚轮 ----
    0x05, 0x01,       # Usage Page (Generic Desktop)
    0x09, 0x02,       # Usage (Mouse)
    0xA1, 0x01,       # Collection (Application)
    0x85, 0x01,       #   Report ID (1)
    0x09, 0x01,       #   Usage (Pointer)
    0xA1, 0x00,       #   Collection (Physical)
    0x05, 0x09,       #     Usage Page (Button)
    0x19, 0x01,       #     Usage Minimum (1)
    0x29, 0x05,       #     Usage Maximum (5)
    0x15, 0x00,       #     Logical Minimum (0)
    0x25, 0x01,       #     Logical Maximum (1)
    0x95, 0x05,       #     Report Count (5)
    0x75, 0x01,       #     Report Size (1)
    0x81, 0x02,       #     Input (Data,Var,Abs)   5 个按键
    0x95, 0x01,       #     Report Count (1)
    0x75, 0x03,       #     Report Size (3)
    0x81, 0x01,       #     Input (Const)          对齐填充
    0x05, 0x01,       #     Usage Page (Generic Desktop)
    0x09, 0x30,       #     Usage (X)
    0x09, 0x31,       #     Usage (Y)
    0x16, 0x00, 0x00, #     Logical Minimum (0)
    0x26, 0xFF, 0x7F, #     Logical Maximum (32767)
    0x75, 0x10,       #     Report Size (16)
    0x95, 0x02,       #     Report Count (2)
    0x81, 0x02,       #     Input (Data,Var,Abs)   绝对坐标 X/Y
    0x09, 0x38,       #     Usage (Wheel)
    0x15, 0x81,       #     Logical Minimum (-127)
    0x25, 0x7F,       #     Logical Maximum (127)
    0x75, 0x08,       #     Report Size (8)
    0x95, 0x01,       #     Report Count (1)
    0x81, 0x06,       #     Input (Data,Var,Rel)   滚轮（相对）
    0xC0,             #   End Collection
    0xC0,             # End Collection
    # ---- Report ID 2: 标准键盘 ----
    0x05, 0x01,       # Usage Page (Generic Desktop)
    0x09, 0x06,       # Usage (Keyboard)
    0xA1, 0x01,       # Collection (Application)
    0x85, 0x02,       #   Report ID (2)
    0x05, 0x07,       #   Usage Page (Key Codes)
    0x19, 0xE0,       #   Usage Minimum (224)
    0x29, 0xE7,       #   Usage Maximum (231)
    0x15, 0x00,       #   Logical Minimum (0)
    0x25, 0x01,       #   Logical Maximum (1)
    0x75, 0x01,       #   Report Size (1)
    0x95, 0x08,       #   Report Count (8)
    0x81, 0x02,       #   Input (Data,Var,Abs)     修饰键
    0x95, 0x01,       #   Report Count (1)
    0x75, 0x08,       #   Report Size (8)
    0x81, 0x01,       #   Input (Const)            保留字节
    0x95, 0x06,       #   Report Count (6)
    0x75, 0x08,       #   Report Size (8)
    0x15, 0x00,       #   Logical Minimum (0)
    0x25, 0x65,       #   Logical Maximum (101)
    0x05, 0x07,       #   Usage Page (Key Codes)
    0x19, 0x00,       #   Usage Minimum (0)
    0x29, 0x65,       #   Usage Maximum (101)
    0x81, 0x00,       #   Input (Data,Array)       6 个按键
    0xC0,             # End Collection
])
with open(sys.argv[1], "wb") as f:
    f.write(desc)
print(f"[ok] report_desc {len(desc)} bytes")
PY

    ln -s "$G/functions/$HID_FUNC" "$G/configs/c.1/$HID_FUNC"
    rebind

    [ -e /dev/hidg0 ] || die "绑定后 /dev/hidg0 未出现，请检查 dmesg"
    chmod 666 /dev/hidg0   # 允许 bianbu 用户直接写，注入程序免 root
    echo "[ok] HID gadget 就绪: /dev/hidg0 (UDC=$UDC)"
    echo "[ok] 屏端将看到复合设备：ADB + 键鼠（VID:PID=$(cat $G/idVendor):$(cat $G/idProduct)）"
}

stop() {
    [ -d "$G/functions/$HID_FUNC" ] || { echo "[skip] $HID_FUNC 不存在"; return 0; }
    unbind
    rm -f "$G/configs/c.1/$HID_FUNC"
    rmdir "$G/functions/$HID_FUNC"
    rebind
    echo "[ok] HID 接口已移除，ADB 保持可用"
}

status() {
    echo "UDC: $UDC"
    echo "g1 绑定状态: $(cat "$G/UDC" 2>/dev/null || echo '(未绑定)')"
    echo "functions: $(ls "$G/functions" 2>/dev/null | tr '\n' ' ')"
    echo "config c.1: $(ls "$G/configs/c.1" 2>/dev/null | grep -v strings | tr '\n' ' ')"
    ls -l /dev/hidg* 2>/dev/null || echo "/dev/hidg*: 无"
}

case "${1:-}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    *) echo "用法: $0 {start|stop|status}"; exit 1 ;;
esac
