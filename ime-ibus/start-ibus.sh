#!/bin/bash
# ime-ibus-start.sh — 启动 IME LLM 输入法引擎
# 
# 用法:
#   bash ime-ibus-start.sh          # 启动并切换到 ime-llm
#   bash ime-ibus-start.sh stop     # 停止 ime-llm

set -e

ENGINE_NAME="ime-llm"
ENGINE_SCRIPT="/usr/share/ibus-ime-llm/ibus_main.py"

start() {
    echo "==> 启动 IME LLM 引擎..."
    
    # 先杀掉残留进程
    for pid in $(pgrep -f "ibus_main.py" 2>/dev/null); do
        kill "$pid" 2>/dev/null || true
    done
    
    # 后台启动引擎
    nohup python3 "$ENGINE_SCRIPT" --ibus >/dev/null 2>&1 &
    
    # 等待注册
    for i in $(seq 1 8); do
        sleep 1
        if ibus engine 2>/dev/null | grep -q "$ENGINE_NAME"; then
            echo "  ✓ 引擎已在运行中"
            return 0
        fi
        # 尝试切换
        ibus engine "$ENGINE_NAME" 2>/dev/null && {
            echo "  ✓ 已切换到 $ENGINE_NAME"
            return 0
        }
    done
    
    echo "  ⚠ 启动完成，尝试切换..."
    ibus engine "$ENGINE_NAME" 2>/dev/null || true
}

stop() {
    echo "==> 停止 IME LLM 引擎..."
    for pid in $(pgrep -f "ibus_main.py" 2>/dev/null); do
        kill "$pid" 2>/dev/null || true
    done
    echo "  ✓ 已停止"
    ibus engine libpinyin 2>/dev/null || true
}

case "${1:-start}" in
    start|--start) start ;;
    stop|--stop) stop ;;
    restart) stop; sleep 1; start ;;
    *) echo "用法: $0 [start|stop|restart]"; exit 1 ;;
esac
