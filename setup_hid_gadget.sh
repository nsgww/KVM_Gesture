#!/usr/bin/env bash
# KVM 手势 —— USB HID gadget 配置（复合设备：在 adbd 的 g1 中追加 HID 接口）
#
# 原理：板子 OTG 口的 gadget g1 由 adbd.service 创建（ffs.adb）。本脚本在 g1 里
# 追加一个 hid.usb0 function 并重新绑定，使板子对主机同时呈现为
# ADB 设备 + 绝对坐标鼠标 + 双指触摸屏 + 键盘。
# ADB 调试不中断（重新绑定瞬间会断一次，随后自动恢复）。
#
# 为什么鼠标和触摸都带：会议大屏的触摸口实测只认 digitizer 不认鼠标，
# 而 Windows 实测认鼠标。双模同发，谁认谁响应。
#
# 用法（需要 root）：
#   sudo bash setup_hid_gadget.sh start    # 加入 HID 接口并绑定
#   sudo bash setup_hid_gadget.sh stop     # 移除 HID 接口
#   sudo bash setup_hid_gadget.sh status
#
# HID 描述符为单接口三 Report ID：
#   ID=1 绝对坐标鼠标：X/Y (0~32767) + 5 键 + 滚轮
#   ID=2 双指触摸屏：contact0/1 + 扫描时间 + 触点数（微软官方范例结构）
#   ID=3 键盘：标准 boot 键盘（修饰键 + 6 键）
set -euo pipefail

G=/sys/kernel/config/usb_gadget/g1
UDC=$(ls /sys/class/udc | head -1)
HID_FUNC=hid.usb0
REPORT_LEN=28    # ID(1) + contact0(12) + contact1(12) + scan_time(2) + contact_count(1)

die() { echo "ERROR: $*" >&2; exit 1; }

[ -d "$G" ] || die "$G 不存在。g1 由 adbd.service 创建，请先确认 adbd.service 已启动。"

unbind() {
    # 注意：sysfs 属性的 st_size 恒为页大小，不能用 -s 判断内容，必须读出来看
    if [ -n "$(cat "$G/UDC" 2>/dev/null)" ]; then
        echo "" > "$G/UDC" || true
        sleep 0.3
    fi
}

rebind() {
    echo "$UDC" > "$G/UDC"
    sleep 0.5
}

remove_func() {
    rm -f "$G/configs/c.1/$HID_FUNC"
    rmdir "$G/functions/$HID_FUNC"
}

start() {
    if [ -d "$G/functions/$HID_FUNC" ]; then
        # 描述符/长度可能有旧版（鼠标版），不一致则重建
        cur_len=$(cat "$G/functions/$HID_FUNC/report_length" 2>/dev/null || echo 0)
        if [ "$cur_len" != "$REPORT_LEN" ]; then
            echo "[info] 检测到旧描述符（report_length=$cur_len），重建触摸接口"
            unbind
            remove_func
        else
            echo "[skip] $HID_FUNC 已存在"
            [ -e /dev/hidg0 ] && echo "[ok] /dev/hidg0 就绪" || rebind
            return 0
        fi
    fi

    unbind

    mkdir "$G/functions/$HID_FUNC"
    # 非 boot 协议（描述符自带 Report ID，触摸+键盘复合）
    echo 0 > "$G/functions/$HID_FUNC/protocol"
    echo 0 > "$G/functions/$HID_FUNC/subclass"
    echo "$REPORT_LEN" > "$G/functions/$HID_FUNC/report_length"

    # 用 python3 写二进制报告描述符（sh 的 printf 转义易错）
    python3 - "$G/functions/$HID_FUNC/report_desc" <<'PY'
import sys
desc = bytes([
    # ---- Report ID 1: 绝对坐标鼠标 + 滚轮（Windows 已验证可用）----
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
    # ---- Report ID 2: 双指触摸屏（按微软官方范例，会议屏触摸口用）----
    0x05, 0x0D,       # Usage Page (Digitizer)
    0x09, 0x04,       # Usage (Touch Screen)
    0xA1, 0x01,       # Collection (Application)
    0x85, 0x02,       #   Report ID (2)
    # ---- 触点 0 ----
    0x09, 0x22,       #   Usage (Finger)
    0xA1, 0x02,       #   Collection (Logical)
    0x09, 0x42,       #     Usage (Tip Switch)
    0x15, 0x00,       #     Logical Minimum (0)
    0x25, 0x01,       #     Logical Maximum (1)
    0x75, 0x01,       #     Report Size (1)
    0x95, 0x01,       #     Report Count (1)
    0x81, 0x02,       #     Input (Data,Var,Abs)   按下标志
    0x95, 0x07,       #     Report Count (7)
    0x81, 0x03,       #     Input (Const)          对齐填充
    0x75, 0x08,       #     Report Size (8)
    0x09, 0x51,       #     Usage (Contact Id)
    0x95, 0x01,       #     Report Count (1)
    0x81, 0x02,       #     Input                  触点编号
    0x05, 0x01,       #     Usage Page (Generic Desktop)
    0x26, 0xFF, 0x0F, #     Logical Maximum (4095)
    0x75, 0x10,       #     Report Size (16)
    0x55, 0x0E,       #     Unit Exponent (-2)
    0x65, 0x13,       #     Unit (Inch, EngLinear)
    0x09, 0x30,       #     Usage (X)
    0x35, 0x00,       #     Physical Minimum (0)
    0x46, 0xB5, 0x04, #     Physical Maximum (1205 = 12.05in)
    0x95, 0x02,       #     Report Count (2)
    0x81, 0x02,       #     Input                  X
    0x46, 0x8A, 0x03, #     Physical Maximum (906 = 9.06in)
    0x09, 0x31,       #     Usage (Y)
    0x81, 0x02,       #     Input                  Y
    0x05, 0x0D,       #     Usage Page (Digitizer)
    0x09, 0x48,       #     Usage (Width)
    0x09, 0x49,       #     Usage (Height)
    0x81, 0x02,       #     Input (16bit x2)       触点宽高
    0x95, 0x01,       #     Report Count (1)
    0x55, 0x0C,       #     Unit Exponent (-4)
    0x65, 0x12,       #     Unit (Radians)
    0x35, 0x00,       #     Physical Minimum (0)
    0x47, 0x6F, 0xF5, 0x00, 0x00,  # Physical Maximum (62831)
    0x15, 0x00,       #     Logical Minimum (0)
    0x27, 0x6F, 0xF5, 0x00, 0x00,  # Logical Maximum (62831)
    0x09, 0x3F,       #     Usage (Azimuth)
    0x81, 0x02,       #     Input (16bit)          触点方向
    0xC0,             #   End Collection
    # ---- 触点 1（双指缩放用）----
    0x09, 0x22,       #   Usage (Finger)
    0xA1, 0x02,       #   Collection (Logical)
    0x09, 0x42,       #     Usage (Tip Switch)
    0x15, 0x00,       #     Logical Minimum (0)
    0x25, 0x01,       #     Logical Maximum (1)
    0x75, 0x01,       #     Report Size (1)
    0x95, 0x01,       #     Report Count (1)
    0x81, 0x02,       #     Input
    0x95, 0x07,       #     Report Count (7)
    0x81, 0x03,       #     Input (Const)
    0x75, 0x08,       #     Report Size (8)
    0x09, 0x51,       #     Usage (Contact Id)
    0x95, 0x01,       #     Report Count (1)
    0x81, 0x02,       #     Input
    0x05, 0x01,       #     Usage Page (Generic Desktop)
    0x26, 0xFF, 0x0F, #     Logical Maximum (4095)
    0x75, 0x10,       #     Report Size (16)
    0x55, 0x0E,       #     Unit Exponent (-2)
    0x65, 0x13,       #     Unit (Inch)
    0x09, 0x30,       #     Usage (X)
    0x35, 0x00,       #     Physical Minimum (0)
    0x46, 0xB5, 0x04, #     Physical Maximum (1205)
    0x95, 0x02,       #     Report Count (2)
    0x81, 0x02,       #     Input
    0x46, 0x8A, 0x03, #     Physical Maximum (906)
    0x09, 0x31,       #     Usage (Y)
    0x81, 0x02,       #     Input
    0x05, 0x0D,       #     Usage Page (Digitizer)
    0x09, 0x48,       #     Usage (Width)
    0x09, 0x49,       #     Usage (Height)
    0x81, 0x02,       #     Input
    0x95, 0x01,       #     Report Count (1)
    0x55, 0x0C,       #     Unit Exponent (-4)
    0x65, 0x12,       #     Unit (Radians)
    0x35, 0x00,       #     Physical Minimum (0)
    0x47, 0x6F, 0xF5, 0x00, 0x00,
    0x15, 0x00,
    0x27, 0x6F, 0xF5, 0x00, 0x00,
    0x09, 0x3F,       #     Usage (Azimuth)
    0x81, 0x02,       #     Input
    0xC0,             #   End Collection
    # ---- 扫描时间 + 触点统计 ----
    0x05, 0x0D,       #   Usage Page (Digitizer)
    0x55, 0x0C,       #   Unit Exponent (-4)
    0x66, 0x01, 0x10, #   Unit (Seconds)
    0x47, 0xFF, 0xFF, 0x00, 0x00,  # Physical Maximum (65535)
    0x27, 0xFF, 0xFF, 0x00, 0x00,  # Logical Maximum (65535)
    0x75, 0x10,       #   Report Size (16)
    0x95, 0x01,       #   Report Count (1)
    0x09, 0x56,       #   Usage (Scan Time)
    0x81, 0x02,       #   Input                    扫描时间
    0x09, 0x54,       #   Usage (Contact Count)
    0x25, 0x7F,       #   Logical Maximum (127)
    0x95, 0x01,       #   Report Count (1)
    0x75, 0x08,       #   Report Size (8)
    0x81, 0x02,       #   Input                    本报告实际触点数
    # 注意：不含 Contact Count Maximum Feature——hidg 对 GET_REPORT 的应答是全零，
    # 主机解析为"0 个触点"会丢光所有触点数据，故不放（安卓 hid-multitouch 可容忍缺失）
    0xC0,             # End Collection
    # ---- Report ID 3: 标准键盘 ----
    0x05, 0x01,       # Usage Page (Generic Desktop)
    0x09, 0x06,       # Usage (Keyboard)
    0xA1, 0x01,       # Collection (Application)
    0x85, 0x03,       #   Report ID (3)
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
    echo "[ok] 屏端将看到复合设备：ADB + 双指触摸屏（VID:PID=$(cat $G/idVendor):$(cat $G/idProduct)）"
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
