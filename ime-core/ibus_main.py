#!/usr/bin/env python3
"""
IBus Engine for LLM-powered IME.

Architecture:
  - Subclasses IBus.Engine
  - Wraps ime-core Engine (cache + LLM layers)
  - Runs as a D-Bus service managed by ibus-daemon
  - Non-blocking LLM refinement via GLib timeout

Keybindings (IBus):
  a-z        → compose pinyin
  1-9        → select candidate (page)
  Space      → accept top candidate
  Enter      → commit raw pinyin letters
  +/-        → next/previous candidate page
  BackSpace  → delete last pinyin char
  Escape     → cancel composing
  , . ! ?…  → commit + output Chinese punctuation

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
from config import load_config

log = logging.getLogger("ime-ibus")
_DEFAULT_OBJECT_PATH = "/org/freedesktop/IBus/engine/IMEEngine/0"
_BUS_NAME = "org.freedesktop.IBus.IMEEngine"


# ASCII → fullwidth Chinese punctuation
PUNCTUATION_MAP = {
    ".": "。",
    ",": "，",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
    '"': "“",
    "'": "‘",
    "(": "（",
    ")": "）",
    "[": "【",
    "]": "】",
    "~": "～",
    "@": "＠",
    "#": "＃",
    "$": "￥",
    "%": "％",
    "^": "……",
    "&": "＆",
    "*": "＊",
    "<": "《",
    ">": "》",
    "_": "——",
    "\\": "、",
    "`": "·",
}

# Keys that produce Chinese punctuation instead of ASCII
PUNCT_KEYVAL_TO_CJK = {
    IBus.KEY_comma: "，",
    IBus.KEY_period: "。",
    IBus.KEY_question: "？",
    IBus.KEY_exclam: "！",
    IBus.KEY_colon: "：",
    IBus.KEY_semicolon: "；",
    IBus.KEY_quotedbl: "“",
    IBus.KEY_apostrophe: "‘",
    IBus.KEY_parenleft: "（",
    IBus.KEY_parenright: "）",
    IBus.KEY_bracketleft: "【",
    IBus.KEY_bracketright: "】",
    IBus.KEY_numbersign: "＃",
    IBus.KEY_dollar: "￥",
    IBus.KEY_percent: "％",
    IBus.KEY_at: "＠",
    IBus.KEY_asciitilde: "～",
    IBus.KEY_underscore: "——",
    IBus.KEY_backslash: "、",
    IBus.KEY_grave: "·",
    IBus.KEY_less: "《",
    IBus.KEY_greater: "》",
}


class IMEBusEngine(IBus.Engine):
    """IBus engine wrapper for the LLM-driven IME."""

    __gtype_name__ = "IMEBusEngine"

    def __init__(self, bus: IBus.Bus, object_path: str = _DEFAULT_OBJECT_PATH):
        super().__init__(connection=bus.get_connection(), object_path=object_path)
        self._bus = bus

        # ── Core engine ──
        self._engine = Engine(config=load_config())
        max_candidates = self._engine.config.get("engine", {}).get("max_candidates", 36)

        # ── State ──
        self._pinyin = ""          # current composing pinyin
        self._candidates = []      # [(text, source)] all candidates
        self._enabled = True

        # ── IBus UI components ──
        # Horizontal layout: 2 rows × 5 candidates per page
        self._lookup_table = IBus.LookupTable(
            page_size=10,
            cursor_pos=0,
            cursor_visible=True,
            round=True,
            orientation=IBus.Orientation.HORIZONTAL,
        )

        # ── Timer ──
        GLib.timeout_add(400, self._on_poll_llm)

        log.info("IMEBusEngine initialized")

    # ── IBus Engine overrides ──

    def do_process_key_event(self, keyval: int, keycode: int, state: int) -> bool:
        """Handle key events. Return True if consumed."""
        try:
            return self._do_process_key_event(keyval, keycode, state)
        except Exception as exc:
            log.error("do_process_key_event crashed: %s", exc, exc_info=True)
            # Reset state to prevent cascading crashes
            self._pinyin = ""
            self._candidates = []
            try:
                self._update_ui()
            except Exception:
                pass
            return True  # Consume to prevent bad data reaching app

    def _do_process_key_event(self, keyval: int, keycode: int, state: int) -> bool:
        """Internal key handler with caller exception protection."""
        if not self._enabled:
            return False

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

        # ── Number keys (1-9) — select candidate on current page ──
        if 0x31 <= keyval <= 0x39:
            idx = keyval - 0x31
            # Index on current page (IBus LookupTable handles cursor position)
            if idx < len(self._candidates):
                # Commit at global index = page_start + idx
                ps = self._lookup_table.page_size
                page_start = self._lookup_table.get_cursor_pos() // ps * ps
                global_idx = page_start + idx
                if global_idx < len(self._candidates):
                    self._commit(global_idx)
                else:
                    self._flush_pending()
                    return False
            else:
                self._flush_pending()
                return False
            return True

        # ── Space — select top candidate ──
        if keyval == IBus.KEY_space:
            if self._pinyin and self._candidates:
                self._commit(0)
                return True
            return False

        # ── Return — commit raw pinyin letters ──
        if keyval in (IBus.KEY_Return, IBus.KEY_KP_Enter):
            if self._pinyin:
                self._commit_raw_pinyin()
                return True
            return False

        # ── +/- (or =) — page navigation ──
        if keyval in (IBus.KEY_equal, IBus.KEY_plus, IBus.KEY_KP_Add):
            if self._pinyin and self._candidates:
                self._page_next()
                return True
            return False
        if keyval in (IBus.KEY_minus, IBus.KEY_KP_Subtract):
            if self._pinyin and self._candidates:
                self._page_prev()
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

        # ── Punctuation keys — commit + output Chinese punctuation ──
        cjk_punct = PUNCT_KEYVAL_TO_CJK.get(keyval)
        if cjk_punct is not None:
            self._commit_with_punctuation(cjk_punct)
            return True

        # ── Any other printable key — flush and forward ──
        if self._pinyin:
            self._flush_pending()
        return False

    def do_enable(self):
        try:
            self._enabled = True
            log.info("Engine enabled")
        except Exception as exc:
            log.error("do_enable error: %s", exc)

    def do_disable(self):
        try:
            self._enabled = False
            self._pinyin = ""
            self._candidates = []
            self._update_ui()
            log.info("Engine disabled")
        except Exception as exc:
            log.error("do_disable error: %s", exc)

    def do_reset(self):
        try:
            self._pinyin = ""
            self._candidates = []
            self._update_ui()
        except Exception as exc:
            log.error("do_reset error: %s", exc)

    def do_focus_in(self):
        pass

    def do_focus_out(self):
        pass

    def do_page_up(self):
        """Handle PageUp for candidate page navigation."""
        try:
            if self._pinyin and self._candidates:
                self._page_prev()
                return True
        except Exception as exc:
            log.error("do_page_up error: %s", exc)
        return False

    def do_page_down(self):
        """Handle PageDown for candidate page navigation."""
        try:
            if self._pinyin and self._candidates:
                self._page_next()
                return True
        except Exception as exc:
            log.error("do_page_down error: %s", exc)
        return False

    def do_cursor_down(self):
        """Handle cursor down → next page (common IME convention)."""
        try:
            if self._pinyin and self._candidates:
                self._page_next()
                return True
        except Exception as exc:
            log.error("do_cursor_down error: %s", exc)
        return False

    def do_destroy(self):
        try:
            log.info("Engine destroyed")
            super().destroy()
        except Exception as exc:
            log.error("do_destroy error: %s", exc)

    # ── Internal methods ──

    def _refresh_candidates(self):
        """Re-process pinyin through the engine pipeline."""
        try:
            if not self._pinyin:
                self._candidates = []
                return
            raw = self._engine.process(self._pinyin)
            self._candidates = [(t, s) for t, _, s in raw]
        except Exception as exc:
            log.error("_refresh_candidates failed: %s", exc, exc_info=True)
            self._candidates = []

    def _commit(self, index: int):
        """Commit the candidate at `index` (global index) to the application."""
        try:
            if index < 0 or index >= len(self._candidates):
                return
            text, source = self._candidates[index]
            self._engine.select(text, self._pinyin)

            log.info("Commit: %s (source=%s, pinyin=%s)", text, source, self._pinyin)

            self.commit_text(IBus.Text.new_from_string(text))

            self._pinyin = ""
            self._candidates = []
            self._update_ui()
        except Exception as exc:
            log.error("_commit crashed: %s", exc, exc_info=True)
            self._pinyin = ""
            self._candidates = []

    def _commit_raw_pinyin(self):
        """Enter key: commit raw pinyin letters as text."""
        try:
            if not self._pinyin:
                return
            raw = self._pinyin
            log.info("Commit raw pinyin: %s", raw)
            self.commit_text(IBus.Text.new_from_string(raw))
            self._pinyin = ""
            self._candidates = []
            self._update_ui()
        except Exception as exc:
            log.error("_commit_raw_pinyin crashed: %s", exc, exc_info=True)
            self._pinyin = ""
            self._candidates = []

    def _commit_with_punctuation(self, cjk_punct: str):
        """Commit composed pinyin (top candidate) then output punctuation."""
        try:
            if self._pinyin and self._candidates:
                top_text = self._candidates[0][0]
                self._engine.select(top_text, self._pinyin)
                self.commit_text(IBus.Text.new_from_string(top_text))
                log.info("Commit: %s + punct %s", top_text, cjk_punct)
            elif self._pinyin:
                # No candidates — commit raw pinyin
                self.commit_text(IBus.Text.new_from_string(self._pinyin))
            # Output punctuation
            self.commit_text(IBus.Text.new_from_string(cjk_punct))
            self._pinyin = ""
            self._candidates = []
            self._update_ui()
        except Exception as exc:
            log.error("_commit_with_punctuation crashed: %s", exc, exc_info=True)
            self._pinyin = ""
            self._candidates = []

    def _flush_pending(self):
        """Commit current pinyin buffer as raw text."""
        try:
            if self._pinyin:
                log.info("Flush pending pinyin: %s", self._pinyin)
                self._pinyin = ""
                self._candidates = []
                self._update_ui()
        except Exception as exc:
            log.error("_flush_pending crashed: %s", exc, exc_info=True)
            self._pinyin = ""
            self._candidates = []

    def _page_next(self):
        """Advance to next candidate page."""
        try:
            if not self._candidates:
                return
            self._lookup_table.page_down()
            self._update_ui()
            log.debug("Page next: cursor=%d", self._lookup_table.get_cursor_pos())
        except Exception as exc:
            log.error("_page_next failed: %s", exc)

    def _page_prev(self):
        """Go to previous candidate page."""
        try:
            if not self._candidates:
                return
            self._lookup_table.page_up()
            self._update_ui()
            log.debug("Page prev: cursor=%d", self._lookup_table.get_cursor_pos())
        except Exception as exc:
            log.error("_page_prev failed: %s", exc)

    def _update_ui(self):
        """Update preedit text, auxiliary source indicator, and lookup table."""
        try:
            # ── Preedit (composing text) ──
            if self._pinyin:
                preedit = IBus.Text.new_from_string(self._pinyin)
                self.update_preedit_text(preedit, len(self._pinyin), True)
            else:
                self.hide_preedit_text()

            # ── Source indicator ──
            has_llm = any(s == "llm" for _, s in self._candidates)
            if self._pinyin and self._candidates:
                if has_llm:
                    aux = IBus.Text.new_from_string("● LLM ✓")
                else:
                    aux = IBus.Text.new_from_string("○ 拼音映射")
                self.update_auxiliary_text(aux, True)
            elif self._pinyin:
                aux = IBus.Text.new_from_string("⌨ 输入中…")
                self.update_auxiliary_text(aux, True)
            else:
                self.hide_auxiliary_text()

            # ── Lookup table with source markers ──
            if self._pinyin and self._candidates:
                saved_pos = self._lookup_table.get_cursor_pos()
                self._lookup_table.clear()
                for text, source in self._candidates:
                    label = "✦" + text if source == "llm" else text
                    self._lookup_table.append_candidate(
                        IBus.Text.new_from_string(label)
                    )
                self._lookup_table.set_cursor_pos(
                    min(saved_pos, max(0, len(self._candidates) - 1))
                )
                self.update_lookup_table(self._lookup_table, True)
            else:
                self.hide_lookup_table()
        except Exception as exc:
            log.error("_update_ui failed: %s", exc, exc_info=True)

    def _on_poll_llm(self) -> bool:
        """Periodically check for async LLM re-rank results."""
        try:
            result = self._engine.llm_refine()
            if result:
                pinyin, reordered, latency = result
                if pinyin == self._pinyin and reordered:
                    current_map = {t: s for t, s in self._candidates}
                    new_list: list[tuple[str, str]] = []
                    seen: set[str] = set()
                    # LLM order first — mark source as "llm" for visual feedback
                    for text in reordered:
                        if text not in seen and text in current_map:
                            new_list.append((text, "llm"))
                            seen.add(text)
                    # Append candidates LLM didn't re-rank (original source)
                    for t, s in self._candidates:
                        if t not in seen:
                            new_list.append((t, s))
                            seen.add(t)
                    self._candidates = new_list
                    self._update_ui()
                    log.info("LLM re-rank: %d candidates (%.0fms)", len(reordered), latency * 1000)
        except Exception as exc:
            log.error("_on_poll_llm crashed: %s", exc, exc_info=True)
        return True

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

    rc = bus.request_name(_BUS_NAME, 0)
    if rc == 2:
        log.warning("Bus name %s already claimed, continuing", _BUS_NAME)
    elif rc == 1:
        log.info("Bus name %s acquired", _BUS_NAME)

    _ = EngineFactory(bus)

    log.info("IME LLM Engine ready")

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("Shutting down")
        loop.quit()


if __name__ == "__main__":
    main()
