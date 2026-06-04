"""
tkinter GUI for IME prototype (LLM-only mode).

Keyboard shortcuts:
  Number 1-9 → select candidate (on current page)
  Space      → accept top candidate
  Enter      → commit raw pinyin letters
  Backspace  → delete last pinyin char
  +/-        → next/previous page
  Escape     → cancel composing
  , . ! ? ; : → output Chinese punctuation
"""

import itertools
import tkinter as tk
from tkinter import ttk
from typing import Optional

from engine import Engine
from config import DEFAULT_CONFIG, load_config
from settings import SettingsDialog


FONT_LARGE = ("Microsoft YaHei", 14)
FONT_SMALL = ("Microsoft YaHei", 10)
FONT_MONO = ("Consolas", 12)

COLOR_LLM = "#4CAF50"
COLOR_OLLAMA = "#2196F3"
COLOR_MAP = "#9E9E9E"
COLOR_BG = "#F5F5F5"
COLOR_CANDIDATE = "#E3F2FD"
COLOR_PAGE_ACTIVE = "#FF9800"
COLOR_LLM_LOADING = "#FF9800"

PAGE_SIZE = 9  # 3x3 grid

# ASCII → fullwidth Chinese punctuation mapping
PUNCTUATION_MAP = {
    ".": "。",
    ",": "，",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
    '"': "\u201c",
    "'": "\u2018",
    "(": "\uff08",
    ")": "\uff09",
    "[": "\u3010",
    "]": "\u3011",
    "~": "\uff5e",
    "@": "\uff20",
    "#": "\uff03",
    "$": "\uffe5",
    "%": "\uff05",
    "^": "\u2026\u2026",
    "&": "\uff06",
    "*": "\uff0a",
    "<": "\u300a",
    ">": "\u300b",
    "_": "\u2014\u2014",
    "\\": "、",
    "`": "\u00b7",
}

PUNCT_CHARS = set(PUNCTUATION_MAP.keys())

LLM_SPINNER = itertools.cycle(["\u25d0", "\u25d1", "\u25d2", "\u25d3"])  # ◐◑◒◓


class IMEGUI:
    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or Engine()
        cfg = getattr(engine, 'config', DEFAULT_CONFIG) if engine else DEFAULT_CONFIG
        self._page_size = cfg.get("engine", {}).get("page_size", PAGE_SIZE)

        self.root = tk.Tk()
        self.root.title("IME 输入法原型 - LLM模式")
        self.root.geometry("580x360")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # State
        self._current_pinyin = ""
        self._current_candidates: list[tuple[str, float, str]] = []
        self._current_page = 0
        self._candidate_buttons: list[tk.Button] = []
        self._llm_loaded = False
        self._llm_can_update = True
        self._llm_results_applied = 0
        self._debounce_id = None

        self._build_ui()
        self._bind_keys()
        self._poll_llm()

    # ── UI Build ──

    def _build_ui(self):
        # Pinyin input
        input_frame = tk.Frame(self.root, bg=COLOR_BG)
        input_frame.pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(input_frame, text="\u62fc\u97f3:", font=FONT_LARGE, bg=COLOR_BG).pack(side="left")
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            input_frame, textvariable=self.input_var,
            font=FONT_MONO, width=30, bd=2, relief="solid",
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.input_entry.focus()

        # Context display
        context_frame = tk.Frame(self.root, bg=COLOR_BG)
        context_frame.pack(fill="x", padx=12, pady=2)
        tk.Label(context_frame, text="\u8f93\u5165:", font=FONT_SMALL, bg=COLOR_BG).pack(side="left")
        self.context_var = tk.StringVar(value="(\u7a7a)")
        self.context_label = tk.Label(
            context_frame, textvariable=self.context_var,
            font=FONT_LARGE, bg="#FFF", anchor="w",
            height=1, padx=6,
        )
        self.context_label.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Separator
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # Candidate area (3 rows x 3 columns)
        cand_frame = tk.Frame(self.root, bg=COLOR_BG)
        cand_frame.pack(fill="x", padx=12, pady=4)
        tk.Label(cand_frame, text="\u5019\u9009:", font=FONT_SMALL, bg=COLOR_BG).pack(anchor="w")

        self.cand_grid = tk.Frame(cand_frame, bg=COLOR_BG)
        self.cand_grid.pack(fill="x", pady=2)

        self._candidate_buttons = []
        for idx in range(self._page_size):
            btn = tk.Button(
                self.cand_grid,
                text="  ", font=FONT_LARGE,
                bg=COLOR_CANDIDATE, fg="#333",
                relief="flat", padx=8, pady=4,
                anchor="w", cursor="hand2",
                state="disabled",
            )
            row, col = divmod(idx, 3)
            btn.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
            self.cand_grid.columnconfigure(col, weight=1)
            self._candidate_buttons.append(btn)

        # Page navigation
        page_frame = tk.Frame(self.root, bg=COLOR_BG)
        page_frame.pack(fill="x", padx=12, pady=(0, 4))
        self.page_prev_btn = tk.Button(
            page_frame, text=" \u25c0 ", font=FONT_SMALL,
            bg=COLOR_CANDIDATE, relief="flat",
            state="disabled", command=self._prev_page,
        )
        self.page_prev_btn.pack(side="left", padx=(0, 4))
        self.page_label = tk.Label(
            page_frame, text="", font=FONT_SMALL, bg=COLOR_BG, fg="#666",
        )
        self.page_label.pack(side="left", padx=4)
        self.page_next_btn = tk.Button(
            page_frame, text=" \u25b6 ", font=FONT_SMALL,
            bg=COLOR_CANDIDATE, relief="flat",
            state="disabled", command=self._next_page,
        )
        self.page_next_btn.pack(side="left", padx=4)

        # Source indicators — three status badges side by side
        status_badge_frame = tk.Frame(page_frame, bg=COLOR_BG)
        status_badge_frame.pack(side="right", padx=(4, 0))

        self._map_badge = tk.Label(
            status_badge_frame, text="\u6620\u5c04", font=FONT_SMALL,
            bg="#9E9E9E", fg="white", padx=4,
        )
        self._map_badge.pack(side="left", padx=1)

        self._ollama_badge = tk.Label(
            status_badge_frame, text="Ollama", font=FONT_SMALL,
            bg="#9E9E9E", fg="white", padx=4,
        )
        self._ollama_badge.pack(side="left", padx=1)

        self._ds_badge = tk.Label(
            status_badge_frame, text="DeepSeek", font=FONT_SMALL,
            bg="#9E9E9E", fg="white", padx=4,
        )
        self._ds_badge.pack(side="left", padx=1)

        # Status bar
        status_frame = tk.Frame(self.root, bg="#E0E0E0", bd=1, relief="sunken")
        status_frame.pack(fill="x", side="bottom")

        # Settings button
        self._settings_btn = tk.Button(
            status_frame, text="\u2699", font=FONT_SMALL,
            bg="#E0E0E0", relief="flat", cursor="hand2",
            command=self._open_settings, width=3,
        )
        self._settings_btn.pack(side="right", padx=(0, 4), pady=1)

        # Mode label
        self._mode_label = tk.Label(
            status_frame, text="", font=FONT_SMALL,
            bg="#E0E0E0", fg="#666",
        )
        self._mode_label.pack(side="right", padx=(4, 2))

        # Token stats label
        self._token_label = tk.Label(
            status_frame, text="", font=("Consolas", 9),
            bg="#E0E0E0", fg="#888",
        )
        self._token_label.pack(side="right", padx=(4, 2))
        self._update_token_stats()
        self._update_mode_label()

        self.status_var = tk.StringVar(value="\u5c31\u7eea")
        self.status_label = tk.Label(
            status_frame, textvariable=self.status_var,
            font=FONT_SMALL, bg="#E0E0E0", anchor="w",
        )
        self.status_label.pack(fill="x", padx=8, pady=2)

    # ── Key Bindings ──

    def _bind_keys(self):
        self.input_var.trace_add("write", self._on_input_change)

        for i in range(1, self._page_size + 1):
            self.input_entry.bind(
                str(i),
                lambda e, idx=i: self._select(idx - 1) or "break",
            )

        self.input_entry.bind("<space>", lambda e: self._select(0) or "break")
        self.input_entry.bind("<Return>", lambda e: self._commit_raw_pinyin() or "break")
        self.input_entry.bind("+", lambda e: self._next_page() or "break")
        self.input_entry.bind("-", lambda e: self._prev_page() or "break")
        self.input_entry.bind("<KP_Add>", lambda e: self._next_page() or "break")
        self.input_entry.bind("<KP_Subtract>", lambda e: self._prev_page() or "break")

        for ch in PUNCT_CHARS:
            seq = "<less>" if ch == "<" else ch
            self.input_entry.bind(seq, lambda e, c=ch: self._handle_punctuation(c) or "break")

    # ── Pinyin Input ──

    def _on_input_change(self, *_):
        try:
            pinyin = self.input_var.get()
            if pinyin == self._current_pinyin:
                return

            cleaned = pinyin.rstrip("".join(PUNCT_CHARS) + "\uff0c\u3002\uff1f\uff01\uff1b\uff1a")
            if cleaned != pinyin:
                punct_char = pinyin[len(cleaned):]
                self._current_pinyin = cleaned
                self._process_pinyin(cleaned)
                if punct_char:
                    self._auto_commit_pinyin()
                    self._append_to_context(self._to_cjk_punct(punct_char))
                    self.input_var.set("")
                    self._current_pinyin = ""
                return

            self._current_pinyin = pinyin
            self._process_pinyin(pinyin)
        except Exception as exc:
            print(f"[IME GUI] _on_input_change error: {exc}")
            self._current_pinyin = ""
            self._current_candidates = []
            self._clear_candidates()
            self._update_page_nav()

    def _process_pinyin(self, pinyin: str):
        if not pinyin.strip():
            self._clear_candidates()
            self._update_page_nav()
            return

        # Map layer only — instant, called on every keystroke
        candidates = self.engine.process_map(pinyin)
        self._current_candidates = candidates
        self._current_page = 0
        self._llm_loaded = False
        self._llm_can_update = True
        self._llm_results_applied = 0
        # Reset badges to initial state
        self._map_badge.config(text="\u6620\u5c04 \u2713", bg=COLOR_MAP, fg="white")
        if self.engine._use_ollama:
            self._ollama_badge.config(text="Ollama", bg="#9E9E9E", fg="white")
        if self.engine._use_deepseek:
            self._ds_badge.config(text="DeepSeek", bg="#9E9E9E", fg="white")
        self._update_candidates()
        self._update_page_nav()

        # Schedule debounced full process (cancel previous)
        if hasattr(self, '_debounce_id') and self._debounce_id:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(300, self._debounced_process, pinyin)

    def _debounced_process(self, pinyin: str):
        """Full processing with LLM, called after user stops typing for 300ms."""
        self._debounce_id = None
        if pinyin != self._current_pinyin:
            return  # pinyin changed since debounce fired
        # Re-process with full pipeline (map + LLM submission)
        self.engine.process(pinyin)

    def _commit_raw_pinyin(self):
        pinyin = self._current_pinyin
        if not pinyin:
            return
        self._append_to_context(pinyin)
        self.input_var.set("")
        self._current_pinyin = ""
        self._current_candidates = []
        self._clear_candidates()
        self._update_page_nav()
        self._update_status(f"\u8f93\u5165\u62fc\u97f3: {pinyin}")

    # ── Candidate Selection ──

    def _select(self, index: int):
        global_index = self._current_page * self._page_size + index
        if global_index < 0 or global_index >= len(self._current_candidates):
            return
        text, score, source = self._current_candidates[global_index]
        pinyin = self._current_pinyin

        self._llm_can_update = False
        self.engine.select(text, pinyin)

        self.input_var.set("")
        self._current_pinyin = ""
        self._current_candidates = []
        self._clear_candidates()
        self._update_page_nav()

        ctx = self.engine.context
        self.context_var.set(ctx if ctx else "(\u7a7a)")

        if source == "llm":
            self._update_status(f"DeepSeek \u63a8\u8350  |  \u9009\u62e9\u4e86: {text}")
        elif source == "ollama":
            self._update_status(f"Ollama \u63a8\u8350  |  \u9009\u62e9\u4e86: {text}")
        else:
            self._update_status(f"\u9009\u62e9\u4e86: {text}")

    # ── Punctuation Handling ──

    def _to_cjk_punct(self, ch: str) -> str:
        return PUNCTUATION_MAP.get(ch, ch)

    def _auto_commit_pinyin(self):
        candidates = self._current_candidates
        pinyin = self._current_pinyin
        if not pinyin:
            return
        if candidates:
            top_text = candidates[0][0]
            self.engine.select(top_text, pinyin)
        else:
            self._append_to_context(pinyin)

    def _handle_punctuation(self, ascii_punct: str):
        cjk = self._to_cjk_punct(ascii_punct)
        if self._current_pinyin:
            self._auto_commit_pinyin()
            self.input_var.set("")
            self._current_pinyin = ""
            self._current_candidates = []
            self._clear_candidates()
            self._update_page_nav()
        self._append_to_context(cjk)
        ctx = self.engine.context
        self.context_var.set(ctx if ctx else "(\u7a7a)")

    def _append_to_context(self, text: str):
        self.engine._context += text
        ctx = self.engine.context
        self.context_var.set(ctx if ctx else "(\u7a7a)")

    # ── Paging ──

    @property
    def _total_pages(self) -> int:
        if not self._current_candidates:
            return 0
        return (len(self._current_candidates) + self._page_size - 1) // self._page_size

    def _next_page(self):
        if self._total_pages <= 1:
            return
        self._current_page = (self._current_page + 1) % self._total_pages
        self._update_candidates()
        self._update_page_nav()

    def _prev_page(self):
        if self._total_pages <= 1:
            return
        self._current_page = (self._current_page - 1) % self._total_pages
        self._update_candidates()
        self._update_page_nav()

    # ── Candidate Display ──

    def _update_candidates(self):
        page_offset = self._current_page * self._page_size
        for idx, btn in enumerate(self._candidate_buttons):
            global_idx = page_offset + idx
            if global_idx < len(self._current_candidates):
                text, score, source = self._current_candidates[global_idx]
                label = f"[{idx+1}] {text}"
                btn.config(text=label, state="normal")
                if source == "llm":
                    btn.config(bg=COLOR_LLM, fg="white")
                elif source == "ollama":
                    btn.config(bg=COLOR_OLLAMA, fg="white")
                else:
                    btn.config(bg=COLOR_CANDIDATE, fg="#333")
                btn.config(command=lambda i=idx: self._select(i))
            else:
                btn.config(text="  ", state="disabled", bg=COLOR_CANDIDATE, fg="#333")

    def _clear_candidates(self):
        for btn in self._candidate_buttons:
            btn.config(text="  ", state="disabled", bg=COLOR_CANDIDATE, fg="#333")
        self._map_badge.config(text="\u6620\u5c04", bg="#9E9E9E", fg="white")
        self._ollama_badge.config(text="Ollama", bg="#9E9E9E", fg="white")
        self._ds_badge.config(text="DeepSeek", bg="#9E9E9E", fg="white")

    def _update_page_nav(self):
        total = self._total_pages
        page = self._current_page + 1 if total > 0 else 0
        if total > 1:
            self.page_label.config(text=f"\u7b2c {page}/{total} \u9875")
            self.page_prev_btn.config(state="normal")
            self.page_next_btn.config(state="normal")
        else:
            self.page_label.config(text="")
            self.page_prev_btn.config(state="disabled")
            self.page_next_btn.config(state="disabled")

    # ── Status ──

    def _update_status(self, text: str):
        self.status_var.set(text)

    def _update_source_indicator(self):
        """Update three badges: Map ✓, Ollama ✓/◐, DeepSeek ✓/◐, plus token stats."""
        # Map is always available
        self._map_badge.config(text="\u6620\u5c04 \u2713", bg=COLOR_MAP, fg="white")

        # Check which sources are in current candidates
        has_ollama = any(src == "ollama" for _, _, src in self._current_candidates)
        has_deepseek = any(src in ("llm", "deepseek") for _, _, src in self._current_candidates)

        # Ollama badge
        if has_ollama:
            self._ollama_badge.config(text="Ollama \u2713", bg=COLOR_OLLAMA, fg="white")
        elif self.engine.llm_busy:
            sp = next(LLM_SPINNER)
            self._ollama_badge.config(text=f"Ollama {sp}", bg=COLOR_LLM_LOADING, fg="white")
        else:
            self._ollama_badge.config(text="Ollama", bg="#9E9E9E", fg="white")

        # DeepSeek badge
        if has_deepseek:
            self._ds_badge.config(text="DeepSeek \u2713", bg=COLOR_LLM, fg="white")
        else:
            self._ds_badge.config(text="DeepSeek", bg="#9E9E9E", fg="white")

    def _update_mode_label(self):
        mode = self.engine.mode
        labels = {
            "map_only": "\u2726 映射",
            "map_ollama": "\u2726 Ollama",
            "map_deepseek": "\u2726 DeepSeek",
            "map_ollama_deepseek": "\u2726 O+D",
        }
        llm_off = not (self.engine._use_ollama or self.engine._use_deepseek)
        if llm_off:
            self._mode_label.config(text="\u2726 仅映射（LLM 已关）", fg="#E65100")
        else:
            self._mode_label.config(text=labels.get(mode, mode), fg="#666")

    def _update_token_stats(self):
        """Update token usage display in status bar."""
        stats = Engine.get_token_stats()
        total = stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0)
        if total:
            self._token_label.config(text=f"T:{total}")
        else:
            self._token_label.config(text="")

    def _open_settings(self):
        dialog = SettingsDialog(self.root)
        if dialog.result:
            self.engine.reload_config(dialog.result)
            self._update_mode_label()
            self._update_status("\u8bbe\u7f6e\u5df2\u4fdd\u5b58")
            # Re-process current pinyin with new settings
            if self._current_pinyin:
                self._process_pinyin(self._current_pinyin)

    # ── Async LLM Polling ──

    def _poll_llm(self):
        try:
            # Check for newly completed LLM results (Ollama or DeepSeek)
            results = self.engine.poll_results()
            for source, pinyin, refined, latency in results:
                # Feed LLM ranking back into WORD_MAP with scenario-aware weight
                timely = bool(pinyin) and pinyin == self._current_pinyin
                if pinyin:
                    self.engine.feed_llm_ranking(pinyin, refined, source, timely)
                # First to arrive → #1, second → #2 slot
                slot = self._llm_results_applied
                self._llm_results_applied += 1
                self._current_candidates = Engine.merge_ranking(
                    self._current_candidates, refined, source, slot,
                )
                self._current_page = 0
                self._llm_loaded = True
                self._update_candidates()
                self._update_page_nav()
                self._update_source_indicator()
                self._update_token_stats()
                src_label = "Ollama" if source == "ollama" else "DeepSeek"
                top = refined[0] if refined else "?"
                self._update_status(
                    f"{src_label} \u5230\u8fbe: [{top}] ({latency*1000:.0f}ms)"
                )

            # Update badges: show spinner on whichever source is still loading
            busy = self.engine.llm_busy
            if busy:
                sp = next(LLM_SPINNER)
                # Check if Ollama is still busy
                has_ollama_done = any(src == "ollama" for _, _, src in self._current_candidates)
                has_ds_done = any(src in ("llm", "deepseek") for _, _, src in self._current_candidates)
                if not has_ollama_done and self.engine._use_ollama:
                    self._ollama_badge.config(text=f"Ollama {sp}", bg=COLOR_LLM_LOADING, fg="white")
                if not has_ds_done and self.engine._use_deepseek:
                    self._ds_badge.config(text=f"DeepSeek {sp}", bg=COLOR_LLM_LOADING, fg="white")
        except Exception as exc:
            print(f"[IME GUI] _poll_llm error: {exc}")
        finally:
            self.root.after(500, self._poll_llm)

    def _add_llm_candidate(self, refined: list[str]):
        """Merge LLM refinement results into candidate list.

        Handles both re-ranked pinyin_map candidates and free-form
        LLM conversions (mixed 简拼/全拼 not in original candidates).
        """
        if not refined:
            return
        current_map = {t: (sc, s) for t, sc, s in self._current_candidates}
        seen: set[str] = set()
        new_list: list[tuple[str, float, str]] = []
        # LLM order first — mark as "llm" source
        for t in refined:
            if t not in seen:
                sc = current_map.get(t, (1.0,))[0]
                new_list.append((t, sc, "llm"))
                seen.add(t)
        # Append candidates LLM missed (preserve original source)
        for t, sc, s in self._current_candidates:
            if t not in seen:
                new_list.append((t, sc, s))
                seen.add(t)
        self._current_candidates = new_list
        self._current_page = 0
        self._llm_loaded = True
        self._update_candidates()
        self._update_page_nav()
        self._update_source_indicator()

    # ── Run ──

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.engine.save_learner()
        self.root.destroy()
