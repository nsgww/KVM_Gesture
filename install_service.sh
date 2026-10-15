#!/usr/bin/env bash
# 安装 kvm-gesture 系统服务（开机自启、异常自动重启）
set -euo pipefail

DIR=/opt/KVM_Gesture
SERVICE=/etc/systemd/system/kvm-gesture.service
CAMERA="${1:-0}"

echo "==> 生成 systemd service（摄像头编号：$CAMERA）"

cat > "$SERVICE" <<EOF
[Unit]
Description=KVM Gesture Recognition (识别侧)
After=network.target
# 摄像头设备就绪后再启动
After=systemd-udev-settle.service

[Service]
Type=simple
User=bianbu
Group=bianbu
WorkingDirectory=$DIR
Environment=PYTHONUNBUFFERED=1
# 注意：camera 编号需按实际枚举结果调整，用 v4l2-ctl --list-devices 确认
ExecStart=/usr/bin/python3 $DIR/kvm_gesture.py \\
    --source $CAMERA \\
    --out /var/log/kvm_gesture/events.jsonl \\
    --quiet \\
    --stats \\
    --palm-every 3
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

mkdir -p /var/log/kvm_gesture
chown bianbu:bianbu /var/log/kvm_gesture

echo "==> 写入 $SERVICE"
sudo systemctl daemon-reload
sudo systemctl enable kvm-gesture.service
echo "==> 已设置开机自启"

cat <<'EOS'

启动：
    sudo systemctl start kvm-gesture
查看状态：
    systemctl status kvm-gesture
查看实时日志：
    journalctl -u kvm-gesture -f
停止：
    sudo systemctl stop kvm-gesture

提示：摄像头编号需与实际一致。先确认哪个是真实 UVC 摄像头：
    ls -l /dev/video*
    sudo apt install -y v4l-utils && v4l2-ctl --list-devices
EOS