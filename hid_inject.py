#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KVM 手势事件 → USB HID 注入（板端，注入侧）。

读 kvm_gesture.py 输出的事件流（JSONL 文件 tail 或 stdin），翻译成 HID 报告
写入 /dev/hidg0。HID 描述符由 setup_hid_gadget.sh 配置，单接口双 Report ID：

  ID=1 鼠标（7 字节）: [id][buttons][x_lo][x_hi][y_lo][y_hi][wheel]
                      x/y 为绝对坐标 0~32767，wheel 相对 -127~127
  ID=2 键盘（9 字节）: [id][modifier][reserved][key1..key6]  boot 协议

事件映射：
  MOVE                指针移动到 (x,y)
  GRAB                左键按下（此后 MOVE 即拖拽）
  RELEASE             左键抬起
  CLICK               左键点按
  SWIPE_UP/DOWN       滚轮 ±1（上滑=向上滚动）
  SWIPE_LEFT/RIGHT    键盘 →/←（翻页语义：左滑=下一页）
  ZOOM_IN/OUT         Ctrl+滚轮 ±1（浏览器/看图通用缩放）

用法：
  # 跟随事件文件（识别服务落盘模式，推荐）
  python3 hid_inject.py --events /var/log/kvm_gesture/events.jsonl

  # 管道直连（不落盘）
  python3 kvm_gesture.py --source 0 --quiet | python3 hid_inject.py --stdin

  # 不写设备，打印报告字节（调试用）
  python3 hid_inject.py --events ev.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HIDG = "/dev/hidg0"
ABS_MAX = 32767

RID_MOUSE = 1
RID_KEYBOARD = 2

BTN_LEFT = 0x01
MOD_LCTRL = 0x01
KEY_RIGHT = 0x4F
KEY_LEFT = 0x50

CLICK_HOLD_S = 0.04


class HidWriter:
    """/dev/hidg0 写入封装：非阻塞打开，无主机/断连时不崩溃，自动重开。"""

    def __init__(self, path: str, dry_run: bool = False):
        self.path = path
        self.dry_run = dry_run
        self.fd = None
        self._err_ts = 0.0
        if not dry_run:
            self._open()

    def _open(self):
        try:
            self.fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
            print(f"[hid] opened {self.path}", flush=True)
        except OSError as e:
            self.fd = None
            self._log_err(f"open {self.path}: {e}")

    def _log_err(self, msg: str):
        now = time.time()
        if now - self._err_ts > 5:      # 错误日志限频，避免刷爆 journal
            self._err_ts = now
            print(f"[hid] {msg}", file=sys.stderr, flush=True)

    def write(self, report: bytes) -> bool:
        if self.dry_run:
            print("[dry] " + report.hex(" "), flush=True)
            return True
        if self.fd is None:
            self._open()
            if self.fd is None:
                return False
        try:
            os.write(self.fd, report)
            return True
        except BlockingIOError:
            return False               # 主机来不及读，丢一帧比卡住强
        except OSError as e:           # 无主机连接 / 线缆拔出
            self._log_err(f"write: {e}（屏未连接或未识别，继续等待）")
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            time.sleep(0.5)
            return False


class Injector:
    """事件流 → HID 报告。维护按键状态，GRAB 后的 MOVE 即为拖拽。"""

    def __init__(self, hid: HidWriter, move_rate: float = 60.0):
        self.hid = hid
        self.buttons = 0
        self.x = ABS_MAX // 2
        self.y = ABS_MAX // 2
        self._last_move = 0.0
        self._move_interval = 1.0 / move_rate

    # -- HID 报告 ----------------------------------------------------------
    def _mouse(self, wheel: int = 0) -> bytes:
        return bytes([
            RID_MOUSE, self.buttons,
            self.x & 0xFF, (self.x >> 8) & 0xFF,
            self.y & 0xFF, (self.y >> 8) & 0xFF,
            wheel & 0xFF,
        ])

    @staticmethod
    def _keyboard(modifier: int = 0, key: int = 0) -> bytes:
        return bytes([RID_KEYBOARD, modifier, 0, key, 0, 0, 0, 0, 0])

    def _set_xy(self, ev: dict):
        if ev.get("x") is not None:
            self.x = max(0, min(ABS_MAX, round(float(ev["x"]) * ABS_MAX)))
        if ev.get("y") is not None:
            self.y = max(0, min(ABS_MAX, round(float(ev["y"]) * ABS_MAX)))

    def _wheel_tick(self, delta: int):
        self.hid.write(self._mouse(wheel=delta))
        self.hid.write(self._mouse(wheel=0))

    def _key_tap(self, key: int):
        self.hid.write(self._keyboard(key=key))
        time.sleep(0.03)
        self.hid.write(self._keyboard())

    def _zoom(self, delta: int):
        self.hid.write(self._keyboard(modifier=MOD_LCTRL))
        self._wheel_tick(delta)
        self.hid.write(self._keyboard())

    # -- 事件分发 -----------------------------------------------------------
    def handle(self, ev: dict):
        act = ev.get("action")
        self._set_xy(ev)

        if act == "MOVE":
            now = time.monotonic()
            if now - self._last_move < self._move_interval:
                return                       # 限频，防主机中断端点拥塞
            self._last_move = now
            self.hid.write(self._mouse())

        elif act == "GRAB":
            self.buttons |= BTN_LEFT
            self.hid.write(self._mouse())

        elif act == "RELEASE":
            self.buttons &= ~BTN_LEFT
            self.hid.write(self._mouse())

        elif act == "CLICK":
            self.hid.write(self._mouse())    # 先到位
            self.buttons |= BTN_LEFT
            self.hid.write(self._mouse())
            time.sleep(CLICK_HOLD_S)
            self.buttons &= ~BTN_LEFT
            self.hid.write(self._mouse())

        elif act == "SWIPE_UP":
            self._wheel_tick(1)
        elif act == "SWIPE_DOWN":
            self._wheel_tick(-1)
        elif act == "SWIPE_LEFT":
            self._key_tap(KEY_RIGHT)         # 左滑 = 下一页
        elif act == "SWIPE_RIGHT":
            self._key_tap(KEY_LEFT)          # 右滑 = 上一页

        elif act == "ZOOM_IN":
            self._zoom(1)
        elif act == "ZOOM_OUT":
            self._zoom(-1)
        # 未知 action 忽略，保证协议向后兼容


def follow_file(path: str):
    """tail -f：从文件末尾开始读；文件被截断/重建时自动重开。"""
    fd = None
    inode = None
    while True:
        try:
            if fd is None:
                fd = open(path, "r", encoding="utf-8")
                fd.seek(0, os.SEEK_END)
                inode = os.fstat(fd.fileno()).st_ino
                print(f"[inject] following {path}", flush=True)
            line = fd.readline()
            if line:
                yield line
                continue
            # 检查文件是否被替换（日志轮转等）
            try:
                if os.stat(path).st_ino != inode:
                    fd.close()
                    fd = None
            except OSError:
                fd.close()
                fd = None
            time.sleep(0.02)
        except OSError:
            if fd is not None:
                fd.close()
                fd = None
            print(f"[inject] 等待事件文件 {path} ...", flush=True)
            time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="KVM 手势事件 → USB HID 注入")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--events", default="", help="事件 JSONL 文件（tail 跟随）")
    src.add_argument("--stdin", action="store_true", help="从 stdin 读事件（管道模式）")
    ap.add_argument("--hidg", default=HIDG, help="hidg 设备路径")
    ap.add_argument("--dry-run", action="store_true", help="不写设备，打印报告字节")
    ap.add_argument("--move-rate", type=float, default=60.0, help="MOVE 事件最大转发频率")
    args = ap.parse_args()

    hid = HidWriter(args.hidg, dry_run=args.dry_run)
    inj = Injector(hid, move_rate=args.move_rate)

    lines = sys.stdin if args.stdin else follow_file(args.events)
    n = 0
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        inj.handle(ev)
        n += 1
        if n % 500 == 0:
            print(f"[inject] {n} events processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
