#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# install.sh — Install IME LLM as a system-level IBus engine
#
# Usage:
#   sudo ./install.sh           # install
#   sudo ./install.sh --uninstall  # remove
#
# This script:
#   1. Creates /usr/libexec/ibus-engine-ime-llm (launcher script)
#   2. Creates /usr/share/ibus-ime-llm/       (icon)
#   3. Installs component XML to /usr/share/ibus/component/
#   4. Runs ibus write-cache to register the engine
#   5. Restarts ibus-daemon so engine appears in switcher
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IME_CORE_DIR="$PROJECT_DIR/ime-core"
IME_IBUS_DIR="$PROJECT_DIR/ime-ibus"

LIBEXEC_TARGET="/usr/libexec/ibus-engine-ime-llm"
COMPONENT_TARGET="/usr/share/ibus/component/ime-llm.xml"
ICON_DIR="/usr/share/ibus-ime-llm/icons"
ICON_TARGET="$ICON_DIR/ime-llm.svg"
ENGINE_SCRIPT="$IME_CORE_DIR/ibus_main.py"

do_install() {
    echo "==> Installing IME LLM IBus engine..."

    # 1. Launcher script at /usr/libexec/
    echo "  → Creating $LIBEXEC_TARGET"
    cat > "$LIBEXEC_TARGET" << 'LAUNCHER'
#!/bin/sh
# IME LLM IBus engine launcher
exec /usr/bin/python3 /usr/share/ibus-ime-llm/ibus_main.py "$@"
LAUNCHER
    chmod 755 "$LIBEXEC_TARGET"

    # 2. Copy the Python engine to /usr/share/ibus-ime-llm/
    mkdir -p /usr/share/ibus-ime-llm/
    # We need the full ime-core package, not just bus_main.py
    echo "  → Copying ime-core modules to /usr/share/ibus-ime-llm/"
    cp "$IME_CORE_DIR"/engine.py    /usr/share/ibus-ime-llm/
    cp "$IME_CORE_DIR"/cache.py     /usr/share/ibus-ime-llm/
    cp "$IME_CORE_DIR"/config.py    /usr/share/ibus-ime-llm/
    cp "$IME_CORE_DIR"/llm_backend.py /usr/share/ibus-ime-llm/
    cp "$IME_CORE_DIR"/pinyin_map.py /usr/share/ibus-ime-llm/
    cp "$IME_CORE_DIR"/ibus_main.py  /usr/share/ibus-ime-llm/

    # 3. Icon
    echo "  → Installing icon"
    mkdir -p "$ICON_DIR"
    cp "$IME_IBUS_DIR/ime-llm.svg" "$ICON_TARGET"

    # 4. Component XML
    echo "  → Installing component XML"
    cp "$IME_IBUS_DIR/ime-llm.xml" "$COMPONENT_TARGET"

    # 5. Refresh IBus registry
    echo "  → Refreshing IBus component cache"
    ibus write-cache 2>/dev/null || true

    # 6. Restart IBus
    echo "  → Restarting ibus-daemon"
    ibus restart 2>/dev/null || true

    echo ""
    echo "✅ IME LLM installed successfully!"
    echo ""
    echo "To use it:"
    echo "  1. Open IBus Preferences (ibus-setup)"
    echo "  2. Go to Input Method → Add → Chinese → IME LLM (AI 输入法)"
    echo "  3. Switch with Super+Space or the IBus panel menu"
    echo ""
    echo "NOTE: You need to set your LLM API key:"
    echo "  export LLM_API_KEY='sk-...'"
    echo "  (or edit ~/.bashrc to make it permanent)"
    echo ""
    echo "For local model (Ollama):"
    echo "  export LLM_ENDPOINT='http://localhost:11434/v1'"
    echo "  export LLM_MODEL='qwen2.5:1.5b'"
}

do_uninstall() {
    echo "==> Uninstalling IME LLM..."

    rm -f "$LIBEXEC_TARGET"
    rm -f "$COMPONENT_TARGET"
    rm -rf /usr/share/ibus-ime-llm/
    rm -rf /usr/share/ibus-ime-llm/icons/

    ibus write-cache 2>/dev/null || true
    ibus restart 2>/dev/null || true

    echo "✅ IME LLM uninstalled"
}

# ── Main ──

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root (sudo)." >&2
    exit 1
fi

case "${1:-install}" in
    install|--install)
        do_install
        ;;
    uninstall|--uninstall)
        do_uninstall
        ;;
    *)
        echo "Usage: $0 [install|uninstall]"
        exit 1
        ;;
esac
