#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# start-ibus.sh — Start/Restart IBus daemon for current session
#
# Run this if the IME doesn't appear after installation:
#   bash start-ibus.sh
#
# This creates a persistent systemd --user service for ibus-daemon.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "Warning: No display detected. Are you in a desktop session?"
fi

# Kill any existing ibus-daemon
pkill ibus-daemon 2>/dev/null || true
sleep 1

# Clean stale socket state
rm -rf ~/.cache/ibus

echo "Starting ibus-daemon..."

# Method 1: systemd --user (preferred, persists across tool sessions)
if systemctl --user list-units --all 2>/dev/null | grep -q 'dbus'; then
    mkdir -p ~/.config/systemd/user/
    cat > ~/.config/systemd/user/ibus-daemon.service << 'SERVICE'
[Unit]
Description=IBus Daemon
After=graphical-session.target

[Service]
ExecStart=/usr/bin/ibus-daemon --panel disable --xim
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
SERVICE

    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user start ibus-daemon.service 2>/dev/null || true
    sleep 2

    if systemctl --user is-active ibus-daemon.service 2>/dev/null | grep -q active; then
        echo "✅ ibus-daemon running via systemd (PID: $(systemctl --user show -p MainPID ibus-daemon.service | cut -d= -f2))"
        exit 0
    fi
fi

# Method 2: Direct launch with nohup
ibus-daemon --panel disable --xim &
sleep 2
echo "✅ ibus-daemon started in background"
echo ""
echo "Check with: ibus list-engine | grep ime-llm"
