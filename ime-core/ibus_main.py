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
        self._nav_edit = False     # user navigated (arrows/backspace) since last commit

        # ── IBus UI components ──
        # Horizontal layout: 2 rows × 5 candidates per page
        self._lookup_table = IBus.LookupTable(
            page_size=10,
            cursor_pos=0,
            cursor_visible=True,
            round=True,
            orientation=IBus.Orientation.HORIZONTAL,
        )

        # ── LLM state ──
        self._llm_loaded = False
        self._llm_spinner_idx = 0
        self._llm_can_update = True
        self._llm_results_applied = 0
        self._debounce_id = None

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

        # ── Ctrl+Shift+L — toggle LLM on/off ──
        if (state & IBus.ModifierType.CONTROL_MASK
                and state & IBus.ModifierType.SHIFT_MASK
                and keyval == ord('L')):
            self._toggle_llm()
            return True

        # ── F2 — re-rank with DeepSeek, exclude current top candidate ──
        if keyval == IBus.KEY_F2:
            if self._pinyin and self._candidates:
                self._request_deepseek_refine()
            return True

        # Ignore Ctrl/Alt modified keys (but allow Shift)
        if state & (IBus.ModifierType.CONTROL_MASK | IBus.ModifierType.MOD1_MASK):
            return False

        # ── Letter keys (a-z) — accumulate pinyin ──
        if 0x61 <= keyval <= 0x7a:
            self._nav_edit = False  # new pinyin input → nav chain broken
            self._pinyin += chr(keyval)
            self._refresh_candidates()
            self._update_ui()
            return True

        # ── Number keys (1-9) — select candidate on current page ──
        if 0x31 <= keyval <= 0x39:
            idx = keyval - 0x31
            if idx < len(self._candidates):
                self._llm_can_update = False
                ps = self._lookup_table.page_size
                page_start = self._lookup_table.get_cursor_pos() // ps * ps
                global_idx = page_start + idx
                if global_idx < len(self._candidates):
                    self._commit(global_idx)
                else:
                    self._flush_pending()
                    return False
                return True
            # Prediction mode: number selects prediction
            if not self._pinyin and self._engine._predictions:
                preds = self._engine._predictions
                if idx < len(preds):
                    self._commit_prediction(idx)
                    return True
            self._flush_pending()
            return False

        # ── Space — select top candidate ──
        if keyval == IBus.KEY_space:
            if self._pinyin and self._candidates:
                self._llm_can_update = False
                self._commit(0)
                return True
            # Prediction mode: space selects first prediction
            if not self._pinyin and self._engine._predictions:
                self._commit_prediction(0)
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
            self._nav_edit = True  # backspace with no composing = document nav
            return False

        # ── Escape — cancel composing ──
        if keyval == IBus.KEY_Escape:
            if self._pinyin:
                self._pinyin = ""
                self._candidates = []
                self._update_ui()
                return True
            return False

        # ── Navigation keys (arrows, Delete) when not composing — mark nav edit ──
        if keyval in (IBus.KEY_Left, IBus.KEY_Right, IBus.KEY_Up, IBus.KEY_Down,
                      IBus.KEY_Delete, IBus.KEY_KP_Delete):
            if not self._pinyin:
                self._nav_edit = True
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
        self._nav_edit = True
        return False

    def do_page_down(self):
        """Handle PageDown for candidate page navigation."""
        try:
            if self._pinyin and self._candidates:
                self._page_next()
                return True
        except Exception as exc:
            log.error("do_page_down error: %s", exc)
        self._nav_edit = True
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
            self._engine.save_learner()
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
            # Map layer only — instant feedback
            raw = self._engine.process_map(self._pinyin)
            self._candidates = [(t, s) for t, _, s in raw]
            self._llm_loaded = False
            self._llm_can_update = True
            self._llm_results_applied = 0
            # Schedule debounced LLM submission
            if hasattr(self, '_debounce_id') and self._debounce_id:
                GLib.source_remove(self._debounce_id)
            self._debounce_id = GLib.timeout_add(300, self._debounced_process)
        except Exception as exc:
            log.error("_refresh_candidates failed: %s", exc, exc_info=True)
            self._candidates = []

    def _debounced_process(self) -> bool:
        """Full processing with LLM, called after user stops typing."""
        self._debounce_id = None
        if self._pinyin:
            self._engine.process(self._pinyin)
        return False  # don't repeat

    def _commit(self, index: int):
        """Commit the candidate at `index` (global index) to the application."""
        try:
            if index < 0 or index >= len(self._candidates):
                return
            text, source = self._candidates[index]
            self._engine.select(text, self._pinyin, nav_edit=self._nav_edit)
            self._nav_edit = False

            log.info("Commit: %s (source=%s, pinyin=%s)", text, source, self._pinyin)

            self.commit_text(IBus.Text.new_from_string(text))

            self._pinyin = ""
            self._candidates = []
            self._update_ui()
        except Exception as exc:
            log.error("_commit crashed: %s", exc, exc_info=True)
            self._pinyin = ""
            self._candidates = []

    def _commit_prediction(self, index: int):
        """Commit a prediction (next-word suggestion) as text."""
        try:
            preds = self._engine._predictions
            if index < 0 or index >= len(preds):
                return
            text = preds[index]
            log.info("Prediction commit: %s", text)
            self.commit_text(IBus.Text.new_from_string(text))
            self._engine._context += text
            self._engine._predictions = self._engine._generate_predictions()
            self._update_ui()
        except Exception as exc:
            log.error("_commit_prediction crashed: %s", exc, exc_info=True)

    def _request_deepseek_refine(self):
        """Ctrl+; — user not satisfied, ask DeepSeek to re-rank excluding current top."""
        try:
            if not self._pinyin or not self._candidates:
                return
            top = self._candidates[0][0]
            exclude = list(dict.fromkeys(t for t, _ in self._candidates[:3]))
            self._engine.request_deepseek_refine(
                pinyin=self._pinyin,
                candidates=exclude,
                exclude=[top],
                context=self._engine.context,
            )
            log.info("DeepSeek refine requested: exclude '%s' from %s", top, self._pinyin)
            # Show spinner in aux text
            aux = IBus.Text.new_from_string("DeepSeek 重排中...")
            self.update_auxiliary_text(aux, True)
        except Exception as exc:
            log.error("_request_deepseek_refine error: %s", exc)

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
            self._engine._predictions.clear()
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

    def _toggle_llm(self):
        """Toggle LLM on/off via Ctrl+Shift+L shortcut."""
        llm_on = self._engine.config.get("llm_enabled", True)
        new_val = not llm_on
        self._engine.config["llm_enabled"] = new_val
        # Persist to config file
        from config import save_config
        save_config(self._engine.config)
        # Reload engine flags
        self._engine.reload_config(self._engine.config)
        # Show toast in auxiliary text
        status = "ON" if new_val else "OFF"
        toast = IBus.Text.new_from_string(f"LLM: {status}")
        self.update_auxiliary_text(toast, True)
        # Hide toast after 1.5s
        GLib.timeout_add(1500, self._hide_llm_toast)
        log.info("LLM toggled %s by user", status)
        # Reset any composing state
        self._pinyin = ""
        self._candidates = []
        self.hide_lookup_table()

    def _hide_llm_toast(self) -> bool:
        """Hide the LLM status toast."""
        if not self._pinyin:
            self.hide_auxiliary_text()
        return False

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
            has_ollama = any(s == "ollama" for _, s in self._candidates)
            has_deepseek = any(s in ("llm", "deepseek") for _, s in self._candidates)
            llm_loading = self._engine.llm_busy
            ollama_spinner = "\u25d0\u25d1\u25d2\u25d3"[self._llm_spinner_idx % 4]
            self._llm_spinner_idx += 1

            parts = []
            # LLM master switch indicator
            if not (self._engine._use_ollama or self._engine._use_deepseek):
                parts.append("LLM:OFF")
            parts.append("M\u2713")  # Map always ready
            # Confidence skip indicator
            if getattr(self._engine, '_llm_skipped', False):
                parts.append("\u2717L")  # LLM skipped (confident enough)
            if has_ollama:
                parts.append(f"O\u2713")
            elif self._engine._use_ollama and llm_loading:
                parts.append(f"O{ollama_spinner}")
            elif self._engine._use_ollama:
                parts.append(f"O_")

            if has_deepseek:
                parts.append(f"D\u2713")
            elif self._engine._use_deepseek and llm_loading:
                parts.append(f"D{ollama_spinner}")
            elif self._engine._use_deepseek:
                parts.append(f"D_")

            # Token usage stats — input/output separated, 15-char gap from LLM indicators
            if self._engine._use_deepseek:
                stats = self._engine.get_token_stats()
                pt = stats.get("prompt_tokens", 0)
                ct = stats.get("completion_tokens", 0)
                if pt or ct:
                    parts.append(" " * 14 + f"I:{pt} O:{ct}")

            aux_text = " ".join(parts)
            if self._pinyin:
                aux = IBus.Text.new_from_string(aux_text)
                self.update_auxiliary_text(aux, True)
            else:
                self.hide_auxiliary_text()

            # ── Lookup table with source markers ──
            if self._pinyin and self._candidates:
                saved_pos = self._lookup_table.get_cursor_pos()
                self._lookup_table.clear()
                for text, source in self._candidates:
                    if source in ("llm", "deepseek"):
                        label = "\u2726" + text
                    elif source == "ollama":
                        label = "\u25c9" + text
                    else:
                        label = text
                    self._lookup_table.append_candidate(
                        IBus.Text.new_from_string(label)
                    )
                self._lookup_table.set_cursor_pos(
                    min(saved_pos, max(0, len(self._candidates) - 1))
                )
                self.update_lookup_table(self._lookup_table, True)
            elif not self._pinyin and self._engine._predictions:
                # ── Prediction mode: show next-word predictions ──
                self._lookup_table.clear()
                for text in self._engine._predictions:
                    label = "\u25b8" + text
                    self._lookup_table.append_candidate(
                        IBus.Text.new_from_string(label)
                    )
                self._lookup_table.set_cursor_pos(0)
                self.update_lookup_table(self._lookup_table, True)
                aux = IBus.Text.new_from_string("\u25b8 联想")
                self.update_auxiliary_text(aux, True)
            else:
                self.hide_lookup_table()
        except Exception as exc:
            log.error("_update_ui failed: %s", exc, exc_info=True)

    def _on_poll_llm(self) -> bool:
        """Periodically check for async LLM refinement results."""
        try:
            results = self._engine.poll_results()
            for source, pinyin, refined, latency in results:
                if not refined:
                    continue
                # ── DeepSeek refine (Ctrl+;): replace candidates entirely ──
                if source == "deepseek_refine":
                    current_map = {t: s for t, s in self._candidates}
                    new_candidates = [(t, "llm") for t in refined if t in current_map]
                    # Also include new words DeepSeek suggested (not in original list)
                    for t in refined:
                        if t not in current_map:
                            new_candidates.append((t, "llm"))
                    if new_candidates:
                        self._candidates = new_candidates
                        self._llm_loaded = True
                        self._update_ui()
                        self._update_status(f"DeepSeek 重排完成")
                        log.info("DeepSeek refine replaced candidates: %s", refined[:5])
                    continue

                if self._llm_can_update:
                    # Feed LLM ranking back into WORD_MAP
                    timely = bool(pinyin) and pinyin == self._pinyin
                    if pinyin:
                        self._engine.feed_llm_ranking(pinyin, refined, source, timely)
                    current_map = {t: s for t, s in self._candidates}
                    # First result → #1, second → inserted after first
                    seen: set[str] = set()
                    new_list: list[tuple[str, str]] = []
                    slot = self._llm_results_applied
                    self._llm_results_applied += 1
                    # Build new list: keep top slot entries, then insert this source
                    kept = 0
                    for t, s in self._candidates:
                        if t not in seen:
                            if kept < slot:
                                new_list.append((t, s))
                                seen.add(t)
                                kept += 1
                            else:
                                break
                    for text in refined:
                        if text not in seen and text in current_map:
                            new_list.append((text, source))
                            seen.add(text)
                    for t, s in self._candidates:
                        if t not in seen:
                            new_list.append((t, s))
                            seen.add(t)
                    self._candidates = new_list
                    self._llm_loaded = True
                    self._update_ui()
                    log.info("%s refine: %d candidates (%.0fms)",
                             source, len(refined), latency * 1000)
                else:
                    # Display frozen (user selecting) — learn silently
                    p = self._engine._pending_map_data.get("pinyin", "")
                    if p:
                        self._engine.learn_late_llm_result(p, refined)
                        log.info("%s learned (frozen): %s", source, refined[0] if refined else "?")
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
