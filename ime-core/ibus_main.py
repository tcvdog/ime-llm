#!/usr/bin/env python3
"""
IBus Engine for LLM-powered IME.

Architecture:
  - Subclasses IBus.Engine
  - Wraps ime-core Engine (cache + LLM layers)
  - Runs as a D-Bus service managed by ibus-daemon
  - Non-blocking LLM refinement via GLib timeout

Install:
  See ime-ibus/install.sh
"""

import sys
import os
import logging

from gi import require_version

require_version("IBus", "1.0")
require_version("GLib", "2.0")
from gi.repository import IBus, GLib

# ── Ensure ime-core modules are importable ──
_core_dir = os.path.dirname(os.path.abspath(__file__))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from engine import Engine
from config import DEFAULT_CONFIG

log = logging.getLogger("ime-ibus")
_DEFAULT_OBJECT_PATH = "/org/freedesktop/IBus/engine/IMEEngine/0"
_BUS_NAME = "org.freedesktop.IBus.IMEEngine"
_CACHE_SAVE_INTERVAL = 60  # seconds


class IMEBusEngine(IBus.Engine):
    """IBus engine wrapper for the LLM-driven IME."""

    __gtype_name__ = "IMEBusEngine"

    def __init__(self, bus: IBus.Bus, object_path: str = _DEFAULT_OBJECT_PATH):
        super().__init__(connection=bus.get_connection(), object_path=object_path)
        self._bus = bus

        # ── Core engine (Layer 1 + Layer 2 + Layer 3) ──
        self._engine = Engine()
        self._engine.load_cache()

        # ── State ──
        self._pinyin = ""          # current composing pinyin
        self._candidates = []      # [(text, source)] current candidates
        self._enabled = True

        # ── IBus UI components ──
        self._lookup_table = IBus.LookupTable(
            page_size=9,
            cursor_pos=0,
            cursor_visible=True,
            round=True,
        )
        self._auxiliary_text = ""

        # ── Timers ──
        # Poll for async LLM refinement
        GLib.timeout_add(400, self._on_poll_llm)
        # Periodic cache save
        GLib.timeout_add_seconds(_CACHE_SAVE_INTERVAL, self._on_save_cache)

        log.info("IMEBusEngine initialized")

    # ── IBus Engine overrides ──

    def do_process_key_event(self, keyval: int, keycode: int, state: int) -> bool:
        """Handle key events. Return True if consumed."""
        if not self._enabled:
            return False

        # Ignore key release events
        if state & IBus.ModifierType.RELEASE_MASK:
            return False

        # Ignore Ctrl/Alt modified keys (but allow Shift)
        if state & (IBus.ModifierType.CONTROL_MASK | IBus.ModifierType.MOD1_MASK):
            return False

        # ── Letter keys (a-z) — accumulate pinyin ──
        if 0x61 <= keyval <= 0x7a:
            self._pinyin += chr(keyval)
            self._refresh_candidates()
            self._update_ui()
            return True

        # ── Number keys (1-9) — select candidate ──
        if 0x31 <= keyval <= 0x39:
            idx = keyval - 0x31  # 1→0, 2→1, …, 9→8
            if idx < len(self._candidates):
                self._commit(idx)
            else:
                # Number not tied to candidate — forward key
                self._flush_pending()
                return False
            return True

        # ── Space — select top candidate ──
        if keyval == IBus.KEY_space:
            if self._pinyin and self._candidates:
                self._commit(0)
                return True
            # No pending pinyin — forward space to app
            return False

        # ── Return — commit top candidate ──
        if keyval in (IBus.KEY_Return, IBus.KEY_KP_Enter):
            if self._pinyin:
                self._commit(0)
                return True
            return False

        # ── BackSpace — delete last pinyin char ──
        if keyval == IBus.KEY_BackSpace:
            if self._pinyin:
                self._pinyin = self._pinyin[:-1]
                if self._pinyin:
                    self._refresh_candidates()
                else:
                    self._candidates = []
                self._update_ui()
                return True
            return False

        # ── Escape — cancel composing ──
        if keyval == IBus.KEY_Escape:
            if self._pinyin:
                self._pinyin = ""
                self._candidates = []
                self._update_ui()
                return True
            return False

        # ── Any other printable key — flush and forward ──
        if self._pinyin:
            self._flush_pending()
        return False

    def do_enable(self):
        self._enabled = True
        log.info("Engine enabled")

    def do_disable(self):
        self._enabled = False
        self._pinyin = ""
        self._candidates = []
        self._update_ui()
        log.info("Engine disabled")

    def do_reset(self):
        self._pinyin = ""
        self._candidates = []
        self._update_ui()

    def do_focus_in(self):
        pass

    def do_focus_out(self):
        pass

    def do_destroy(self):
        self._engine.save_cache()
        log.info("Cache saved, engine destroyed")
        super().destroy()

    # ── Internal methods ──

    def _refresh_candidates(self):
        """Re-process pinyin through the engine pipeline."""
        if not self._pinyin:
            self._candidates = []
            return
        raw = self._engine.process(self._pinyin)
        self._candidates = [(t, s) for t, _, s in raw]

    def _commit(self, index: int):
        """Commit the candidate at `index` to the application."""
        if index < 0 or index >= len(self._candidates):
            return
        text, source = self._candidates[index]
        self._engine.select(text, self._pinyin)

        log.info("Commit: %s (source=%s, pinyin=%s)", text, source, self._pinyin)

        self.commit_text(IBus.Text.new_from_string(text))

        # Reset state
        self._pinyin = ""
        self._candidates = []
        self._update_ui()

    def _flush_pending(self):
        """Commit current pinyin buffer as raw text."""
        if self._pinyin:
            log.info("Flush pending pinyin: %s", self._pinyin)
            self._flush_pending_raw = True
            self._pinyin = ""
            self._candidates = []
            self._update_ui()

    def _update_ui(self):
        """Update preedit text and lookup table."""
        # ── Preedit (composing text) ──
        if self._pinyin:
            preedit = IBus.Text.new_from_string(self._pinyin)
            self.update_preedit_text(preedit, len(self._pinyin), True)
        else:
            self.hide_preedit_text()

        # ── Lookup table (candidates) ──
        if self._pinyin and self._candidates:
            self._lookup_table.clear()
            for i, (text, source) in enumerate(self._candidates[:9]):
                self._lookup_table.append_candidate(
                    IBus.Text.new_from_string(text)
                )
            self.update_lookup_table(self._lookup_table, True)
        else:
            self.hide_lookup_table()

    def _on_poll_llm(self) -> bool:
        """Periodically check for async LLM re-rank results."""
        result = self._engine.llm_refine()
        if result:
            pinyin, reordered, latency = result
            # Only use result if user is still on same input
            if pinyin == self._pinyin:
                current_map = {t: s for t, s in self._candidates}
                new_list: list[tuple[str, str]] = []
                seen: set[str] = set()
                for text in reordered:
                    if text not in seen and text in current_map:
                        new_list.append((text, current_map[text]))
                        seen.add(text)
                # Append any candidates LLM didn't include (keeps original order)
                for t, s in self._candidates:
                    if t not in seen:
                        new_list.append((t, s))
                        seen.add(t)
                self._candidates = new_list
                self._update_ui()
                log.info("LLM re-rank: %d candidates (%.0fms)", len(reordered), latency * 1000)
        return True  # keep timer alive

    def _on_save_cache(self) -> bool:
        """Periodically persist the user cache."""
        try:
            self._engine.save_cache()
        except Exception as exc:
            log.warning("Cache save failed: %s", exc)
        return True  # keep timer alive


class EngineFactory(IBus.Factory):
    """Factory that creates IMEBusEngine instances."""

    __gtype_name__ = "EngineFactory"

    def __init__(self, bus: IBus.Bus):
        super().__init__(connection=bus.get_connection(), object_path=IBus.PATH_FACTORY)
        self._bus = bus
        self._engine_count = 0

    def do_create_engine(self, engine_name: str) -> IMEBusEngine:
        path = f"/org/freedesktop/IBus/engine/IMEEngine/{self._engine_count}"
        self._engine_count += 1
        log.info("Creating engine: %s at %s", engine_name, path)
        return IMEBusEngine(self._bus, object_path=path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    IBus.init()

    bus = IBus.Bus()
    if not bus.is_connected():
        log.error("Cannot connect to IBus bus")
        sys.exit(1)

    # Claim our bus name (use raw values: 1=PRIMARY_OWNER, 2=ALREADY_EXISTS)
    rc = bus.request_name(_BUS_NAME, 0)
    if rc == 2:
        log.warning("Bus name %s already claimed, continuing", _BUS_NAME)
    elif rc == 1:
        log.info("Bus name %s acquired", _BUS_NAME)

    # Register factory
    _ = EngineFactory(bus)

    log.info("IME LLM Engine ready")

    # Run main loop
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("Shutting down")
        loop.quit()


if __name__ == "__main__":
    main()
