#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KVM 手势事件 → USB HID 注入（板端，注入侧）。鼠标+触摸双模同发。

读 kvm_gesture.py 输出的事件流（JSONL 文件 tail 或 stdin），翻译成 HID 报告
写入 /dev/hidg0。HID 描述符由 setup_hid_gadget.sh 配置，单接口三 Report ID：

  ID=1 绝对坐标鼠标（7 字节）: [id][buttons][x:2][y:2][wheel]  x/y 0~32767
  ID=2 双指触摸屏（28 字节）:  [id][c0(12B)][c1(12B)][scan_time:2][count]
      contact: [flags][id][x:2][y:2][w:2][h:2][azim:2]  x/y 0~4095
  ID=3 键盘（9 字节）: [id][modifier][reserved][key1..key6]  boot 协议

为什么双模：会议大屏的触摸口实测只认 digitizer 不认鼠标；Windows 实测认鼠标。
每个事件同时发鼠标报告和触摸报告，主机认哪个用哪个。

事件映射：
  MOVE                鼠标指针移动；抓取中=鼠标拖拽+触摸拖动
  GRAB                鼠标左键按下 + 触摸触点 0 按下
  RELEASE             鼠标左键抬起 + 触摸抬起
  CLICK               鼠标点击 + 触摸点按
  SWIPE_UP/DOWN       鼠标滚轮 ±1 + 触摸真实滑动
  SWIPE_LEFT/RIGHT    键盘 →/← + 触摸水平滑动（左滑=下一页）
  ZOOM_IN/OUT         Ctrl+滚轮 ±1 + 触摸双指捏合

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
MOUSE_MAX = 32767            # 鼠标绝对坐标上限（与描述符 Logical Maximum 一致）
TOUCH_MAX = 4095             # 触摸坐标上限（对齐微软官方范例）

RID_MOUSE = 1
RID_TOUCH = 2
RID_KEYBOARD = 3

BTN_LEFT = 0x01
FLAG_TIP = 0x01
CONTACT_W = 100              # 触点宽高，给个固定小值即可
CONTACT_H = 100

MOD_LCTRL = 0x01
KEY_RIGHT = 0x4F
KEY_LEFT = 0x50

CLICK_HOLD_S = 0.04
SWIPE_STEPS = 10
SWIPE_INTERVAL_S = 0.012
SWIPE_DIST = 0.25            # 归一化位移（占屏比例）
ZOOM_STEPS = 8
ZOOM_INTERVAL_S = 0.015
ZOOM_MIN_R = 0.04            # 双指起始半径（归一化）
ZOOM_MAX_R = 0.18            # 双指结束半径


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
            # 主机中断端点短暂拥塞（几 ms 就会轮询一次）：短重试，
            # 保证"按键抬起"这类关键报告不丢，否则会卡在按下状态。
            for _ in range(20):
                time.sleep(0.005)
                try:
                    os.write(self.fd, report)
                    return True
                except BlockingIOError:
                    continue
                except OSError:
                    break
            return False
        except OSError as e:           # 无主机连接 / 线缆拔出
            self._log_err(f"write: {e}（主机未连接或未识别，继续等待）")
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            time.sleep(0.5)
            return False


class Injector:
    """事件流 → HID 报告（鼠标+触摸双模同发）。

    坐标内部用归一化 0~1 保存，发报告时按各设备的 Logical Maximum 缩放。
    """

    def __init__(self, hid: HidWriter, move_rate: float = 60.0):
        self.hid = hid
        self.nx = 0.5
        self.ny = 0.5
        self.buttons = 0                 # 鼠标按键状态
        self.grabbing = False
        self._last_move = 0.0
        self._move_interval = 1.0 / move_rate

    # -- HID 报告 ----------------------------------------------------------
    def _mouse(self, wheel: int = 0) -> bytes:
        x = max(0, min(MOUSE_MAX, round(self.nx * MOUSE_MAX)))
        y = max(0, min(MOUSE_MAX, round(self.ny * MOUSE_MAX)))
        return bytes([RID_MOUSE, self.buttons,
                      x & 0xFF, x >> 8, y & 0xFF, y >> 8, wheel & 0xFF])

    @staticmethod
    def _contact(cid: int, tip: bool, nx: float, ny: float) -> bytes:
        x = max(0, min(TOUCH_MAX, round(nx * TOUCH_MAX)))
        y = max(0, min(TOUCH_MAX, round(ny * TOUCH_MAX)))
        return bytes([FLAG_TIP if tip else 0, cid,
                      x & 0xFF, x >> 8, y & 0xFF, y >> 8,
                      CONTACT_W & 0xFF, CONTACT_W >> 8,
                      CONTACT_H & 0xFF, CONTACT_H >> 8,
                      0, 0])

    def _touch(self, c0=None, c1=None) -> bytes:
        """c0/c1: (tip, nx, ny) 或 None。报告定长 28 字节，不活跃槽位清零。"""
        data = bytearray([RID_TOUCH])
        count = 0
        for cid, c in ((0, c0), (1, c1)):
            if c is None:
                data += bytes(12)
                continue
            tip, nx, ny = c
            data += self._contact(cid, tip, nx, ny)
            count += int(tip)
        scan = int(time.monotonic() * 10000) & 0xFFFF
        data += bytes([scan & 0xFF, scan >> 8, count])
        return bytes(data)

    @staticmethod
    def _keyboard(modifier: int = 0, key: int = 0) -> bytes:
        return bytes([RID_KEYBOARD, modifier, 0, key, 0, 0, 0, 0, 0])

    # -- 复合动作 -----------------------------------------------------------
    def _set_xy(self, ev: dict):
        if ev.get("x") is not None:
            self.nx = max(0.0, min(1.0, float(ev["x"])))
        if ev.get("y") is not None:
            self.ny = max(0.0, min(1.0, float(ev["y"])))

    def _click(self):
        self.hid.write(self._mouse())                      # 鼠标：先到位
        self.buttons |= BTN_LEFT
        self.hid.write(self._mouse())
        self.hid.write(self._touch(c0=(True, self.nx, self.ny)))  # 触摸：按下
        time.sleep(CLICK_HOLD_S)
        self.buttons &= ~BTN_LEFT
        self.hid.write(self._mouse())
        self.hid.write(self._touch())                      # 触摸：抬起

    def _swipe(self, dx: float, dy: float, wheel: int = 0):
        sx, sy = self.nx, self.ny
        if wheel:
            self.hid.write(self._mouse(wheel=wheel))       # 鼠标：滚轮
            self.hid.write(self._mouse())
        for i in range(SWIPE_STEPS + 1):                   # 触摸：真实滑动
            t = i / SWIPE_STEPS
            self.hid.write(self._touch(c0=(True, sx + dx * t, sy + dy * t)))
            time.sleep(SWIPE_INTERVAL_S)
        self.hid.write(self._touch())

    def _key_tap(self, key: int):
        self.hid.write(self._keyboard(key=key))
        time.sleep(0.03)
        self.hid.write(self._keyboard())

    def _zoom(self, zoom_in: bool):
        # 鼠标：Ctrl+滚轮。Ctrl 抬起放 finally，保证任何情况下不卡键。
        try:
            self.hid.write(self._keyboard(modifier=MOD_LCTRL))
            self.hid.write(self._mouse(wheel=1 if zoom_in else -1))
            self.hid.write(self._mouse())
        finally:
            self.hid.write(self._keyboard())
        # 触摸：双指捏合
        r0, r1 = (ZOOM_MIN_R, ZOOM_MAX_R) if zoom_in else (ZOOM_MAX_R, ZOOM_MIN_R)
        for i in range(ZOOM_STEPS + 1):
            t = i / ZOOM_STEPS
            r = r0 + (r1 - r0) * t
            self.hid.write(self._touch(c0=(True, self.nx - r, self.ny),
                                       c1=(True, self.nx + r, self.ny)))
            time.sleep(ZOOM_INTERVAL_S)
        self.hid.write(self._touch())

    # -- 事件分发 -----------------------------------------------------------
    def handle(self, ev: dict):
        act = ev.get("action")
        self._set_xy(ev)

        if act == "MOVE":
            now = time.monotonic()
            if now - self._last_move < self._move_interval:
                return                       # 限频，防主机中断端点拥塞
            self._last_move = now
            self.hid.write(self._mouse())    # 鼠标指针始终跟随
            if self.grabbing:                # 抓取中同步拖触摸点
                self.hid.write(self._touch(c0=(True, self.nx, self.ny)))

        elif act == "GRAB":
            self.grabbing = True
            self.buttons |= BTN_LEFT
            self.hid.write(self._mouse())
            self.hid.write(self._touch(c0=(True, self.nx, self.ny)))

        elif act == "RELEASE":
            self.grabbing = False
            self.buttons &= ~BTN_LEFT
            self.hid.write(self._mouse())
            self.hid.write(self._touch())
            self.hid.write(self._keyboard())   # 兜底清空修饰键，防 Ctrl 卡死

        elif act == "CLICK":
            self._click()

        elif act == "SWIPE_UP":
            self._swipe(0, -SWIPE_DIST, wheel=1)
        elif act == "SWIPE_DOWN":
            self._swipe(0, SWIPE_DIST, wheel=-1)
        elif act == "SWIPE_LEFT":            # 左滑 = 下一页
            self._swipe(-SWIPE_DIST, 0)
            self._key_tap(KEY_RIGHT)
        elif act == "SWIPE_RIGHT":           # 右滑 = 上一页
            self._swipe(SWIPE_DIST, 0)
            self._key_tap(KEY_LEFT)

        elif act == "ZOOM_IN":
            self._zoom(True)
        elif act == "ZOOM_OUT":
            self._zoom(False)
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
    ap = argparse.ArgumentParser(description="KVM 手势事件 → USB HID 注入（鼠标+触摸双模）")
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
