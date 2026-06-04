"""
tkinter GUI for IME prototype.

Layout:
  ┌──────────────────────────────────────────┐
  │ 拼音: [_____________________________]    │
  │ 输入: 你好世界                            │
  │                                          │
  │ 候选: [1]虾面 [2]下面 [3]夏眠            │
  │       [4]虾免 [5]瞎面 [6]虾棉            │
  │       [7]虾民 [8]侠面 [9]峡棉            │
  │       ← 第 1/3 页 →                     │
  │ 来源: 拼音映射                           │
  └──────────────────────────────────────────┘

Keyboard shortcuts:
  Number 1-9 → select candidate (on current page)
  Space      → accept top candidate
  Enter      → commit raw pinyin letters
  Backspace  → delete last pinyin char
  +/-        → next/previous page
  Escape     → cancel composing
  , . ! ? ; : → output Chinese punctuation
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

from engine import Engine
from config import DEFAULT_CONFIG


FONT_LARGE = ("Microsoft YaHei", 14)
FONT_SMALL = ("Microsoft YaHei", 10)
FONT_MONO = ("Consolas", 12)

COLOR_CACHE = "#2196F3"
COLOR_LLM = "#4CAF50"
COLOR_MAP = "#9E9E9E"
COLOR_BG = "#F5F5F5"
COLOR_CANDIDATE = "#E3F2FD"
COLOR_CANDIDATE_HOVER = "#BBDEFB"
COLOR_PAGE_ACTIVE = "#FF9800"

PAGE_SIZE = 9  # 3x3 grid

# ASCII → fullwidth Chinese punctuation mapping
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

PUNCT_CHARS = set(PUNCTUATION_MAP.keys())


class IMEGUI:
    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or Engine()
        cfg = getattr(engine, 'config', DEFAULT_CONFIG) if engine else DEFAULT_CONFIG
        self._page_size = cfg.get("engine", {}).get("page_size", PAGE_SIZE)

        self.root = tk.Tk()
        self.root.title("IME 输入法原型")
        self.root.geometry("580x360")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # State
        self._current_pinyin = ""
        self._current_candidates: list[tuple[str, float, str]] = []
        self._current_page = 0
        self._candidate_buttons: list[tk.Button] = []

        self._build_ui()
        self._bind_keys()
        self._load_cache()
        self._init_llm_providers()
        self._poll_llm()

    # ── UI Build ──

    def _build_ui(self):
        # Pinyin input
        input_frame = tk.Frame(self.root, bg=COLOR_BG)
        input_frame.pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(input_frame, text="拼音:", font=FONT_LARGE, bg=COLOR_BG).pack(side="left")
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
        tk.Label(context_frame, text="输入:", font=FONT_SMALL, bg=COLOR_BG).pack(side="left")
        self.context_var = tk.StringVar(value="(空)")
        self.context_label = tk.Label(
            context_frame, textvariable=self.context_var,
            font=FONT_LARGE, bg="#FFF", anchor="w",
            height=1, padx=6,
        )
        self.context_label.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Separator
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # Candidate area (3 rows x 3 columns of buttons)
        cand_frame = tk.Frame(self.root, bg=COLOR_BG)
        cand_frame.pack(fill="x", padx=12, pady=4)
        tk.Label(cand_frame, text="候选:", font=FONT_SMALL, bg=COLOR_BG).pack(anchor="w")

        self.cand_grid = tk.Frame(cand_frame, bg=COLOR_BG)
        self.cand_grid.pack(fill="x", pady=2)

        # Create 9 candidate buttons (3x3 grid)
        self._candidate_buttons = []
        for idx in range(self._page_size):
            btn = tk.Button(
                self.cand_grid,
                text=f"  ", font=FONT_LARGE,
                bg=COLOR_CANDIDATE, fg="#333",
                relief="flat", padx=8, pady=4,
                anchor="w", cursor="hand2",
                state="disabled",
            )
            row, col = divmod(idx, 3)
            btn.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
            self.cand_grid.columnconfigure(col, weight=1)
            self._candidate_buttons.append(btn)

        # Page navigation bar
        page_frame = tk.Frame(self.root, bg=COLOR_BG)
        page_frame.pack(fill="x", padx=12, pady=(0, 4))
        self.page_prev_btn = tk.Button(
            page_frame, text=" ◀ ", font=FONT_SMALL,
            bg=COLOR_CANDIDATE, relief="flat",
            state="disabled", command=self._prev_page,
        )
        self.page_prev_btn.pack(side="left", padx=(0, 4))
        self.page_label = tk.Label(
            page_frame, text="", font=FONT_SMALL, bg=COLOR_BG, fg="#666",
        )
        self.page_label.pack(side="left", padx=4)
        self.page_next_btn = tk.Button(
            page_frame, text=" ▶ ", font=FONT_SMALL,
            bg=COLOR_CANDIDATE, relief="flat",
            state="disabled", command=self._next_page,
        )
        self.page_next_btn.pack(side="left", padx=4)
        # Source indicator — lights up green when LLM results are adopted
        self._source_indicator = tk.Label(
            page_frame, text="映射", font=FONT_SMALL,
            bg="#9E9E9E", fg="white", padx=6,
        )
        self._source_indicator.pack(side="right", padx=(4, 0))

        # Status bar
        status_frame = tk.Frame(self.root, bg="#E0E0E0", bd=1, relief="sunken")
        status_frame.pack(fill="x", side="bottom")
        # LLM indicator dot (green = connected, gray = offline)
        self._llm_dot = tk.Canvas(status_frame, width=14, height=14,
                                  bg="#E0E0E0", bd=0, highlightthickness=0)
        self._llm_dot.pack(side="left", padx=(8, 2), pady=2)
        self._llm_dot_id = self._llm_dot.create_oval(1, 1, 13, 13,
                                                       fill="#9E9E9E",
                                                       outline="#666")
        tk.Label(status_frame, text="LLM", font=FONT_SMALL,
                 bg="#E0E0E0", fg="#555").pack(side="left", padx=(0, 2))

        # LLM provider selector dropdown
        self._llm_provider_var = tk.StringVar()
        self._llm_provider_combo = ttk.Combobox(
            status_frame, textvariable=self._llm_provider_var,
            font=FONT_SMALL, state="readonly", width=10,
        )
        self._llm_provider_combo.pack(side="left", padx=(0, 6), pady=2)
        self._llm_provider_combo.bind("<<ComboboxSelected>>", self._on_llm_switch)

        self.status_var = tk.StringVar(value="就绪")
        self.status_label = tk.Label(
            status_frame, textvariable=self.status_var,
            font=FONT_SMALL, bg="#E0E0E0", anchor="w",
        )
        self.status_label.pack(fill="x", padx=8, pady=2)
        # Set initial indicator color
        self._update_llm_indicator()

    # ── Key Bindings ──

    def _bind_keys(self):
        # Pinyin input changes
        self.input_var.trace_add("write", self._on_input_change)

        # Number keys for candidate selection
        for i in range(1, self._page_size + 1):
            self.input_entry.bind(
                str(i),
                lambda e, idx=i: self._select(idx - 1) or "break",
            )

        # Space → top candidate on current page
        self.input_entry.bind("<space>", lambda e: self._select(0) or "break")

        # Enter → commit raw pinyin letters
        self.input_entry.bind("<Return>", lambda e: self._commit_raw_pinyin() or "break")

        # +/- → page navigation
        self.input_entry.bind("+", lambda e: self._next_page() or "break")
        self.input_entry.bind("-", lambda e: self._prev_page() or "break")
        self.input_entry.bind("<KP_Add>", lambda e: self._next_page() or "break")
        self.input_entry.bind("<KP_Subtract>", lambda e: self._prev_page() or "break")

        # Punctuation keys → commit pinyin + output Chinese punct
        for ch in PUNCT_CHARS:
            seq = "<less>" if ch == "<" else ch
            self.input_entry.bind(seq, lambda e, c=ch: self._handle_punctuation(c) or "break")

    # ── Pinyin Input ──

    def _on_input_change(self, *_):
        try:
            pinyin = self.input_var.get()
            if pinyin == self._current_pinyin:
                return

            # Detect and strip trailing punctuation typed after pinyin
            # (handles case where punctuation binding didn't fire)
            cleaned = pinyin.rstrip("".join(PUNCT_CHARS) + "，。？！；：")
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

        candidates = self.engine.process(pinyin)
        self._current_candidates = candidates
        self._current_page = 0
        self._update_candidates()
        self._update_page_nav()
        self._update_status(f"来源: 拼音映射")
        self._update_source_indicator()

    def _commit_raw_pinyin(self):
        """Enter key: commit the raw pinyin letters as text."""
        pinyin = self._current_pinyin
        if not pinyin:
            return
        # Append raw pinyin to context
        self._append_to_context(pinyin)
        # Clear state
        self.input_var.set("")
        self._current_pinyin = ""
        self._current_candidates = []
        self._clear_candidates()
        self._update_page_nav()
        self._update_status(f"输入拼音: {pinyin}")

    # ── Candidate Selection ──

    def _select(self, index: int):
        """Select a candidate by index on the current page."""
        global_index = self._current_page * self._page_size + index
        if global_index < 0 or global_index >= len(self._current_candidates):
            return
        text, score, source = self._current_candidates[global_index]
        pinyin = self._current_pinyin

        # Engine learns
        self.engine.select(text, pinyin)

        # Clear input
        self.input_var.set("")
        self._current_pinyin = ""
        self._current_candidates = []
        self._clear_candidates()
        self._update_page_nav()

        # Update context display
        ctx = self.engine.context
        self.context_var.set(ctx if ctx else "(空)")

        # Status
        if source == "cache":
            self._update_status(f"缓存命中 ✓  |  选择了: {text}")
        elif source == "llm":
            self._update_status(f"LLM 推荐  |  选择了: {text}")
        else:
            self._update_status(f"选择了: {text}")

        # Save cache periodically
        self.engine.save_cache()

    # ── Punctuation Handling ──

    def _to_cjk_punct(self, ch: str) -> str:
        """Convert ASCII punctuation to Chinese fullwidth equivalent."""
        return PUNCTUATION_MAP.get(ch, ch)

    def _auto_commit_pinyin(self):
        """Commit current pinyin (as converted text) to context without clearing UI."""
        candidates = self._current_candidates
        pinyin = self._current_pinyin
        if not pinyin:
            return
        if candidates:
            # Commit top candidate
            top_text = candidates[0][0]
            self.engine.select(top_text, pinyin)
        else:
            # Fallback: commit raw pinyin
            self._append_to_context(pinyin)

    def _handle_punctuation(self, ascii_punct: str):
        """Handle punctuation key while composing."""
        cjk = self._to_cjk_punct(ascii_punct)
        if self._current_pinyin:
            # Commit current composed text first
            self._auto_commit_pinyin()
            # Clear input state
            self.input_var.set("")
            self._current_pinyin = ""
            self._current_candidates = []
            self._clear_candidates()
            self._update_page_nav()
        # Output the Chinese punctuation mark
        self._append_to_context(cjk)
        ctx = self.engine.context
        self.context_var.set(ctx if ctx else "(空)")

    def _append_to_context(self, text: str):
        """Append text to engine context and update display."""
        self.engine._context += text
        ctx = self.engine.context
        self.context_var.set(ctx if ctx else "(空)")

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
                if source == "cache":
                    btn.config(bg=COLOR_CACHE, fg="white")
                elif source == "llm":
                    btn.config(bg=COLOR_LLM, fg="white")
                else:
                    btn.config(bg=COLOR_CANDIDATE, fg="#333")
                btn.config(command=lambda i=idx: self._select(i))
            else:
                btn.config(text="  ", state="disabled", bg=COLOR_CANDIDATE, fg="#333")

    def _clear_candidates(self):
        for btn in self._candidate_buttons:
            btn.config(text="  ", state="disabled", bg=COLOR_CANDIDATE, fg="#333")
        self._source_indicator.config(text="—", bg="#9E9E9E", fg="white")

    def _update_page_nav(self):
        total = self._total_pages
        page = self._current_page + 1 if total > 0 else 0
        if total > 1:
            self.page_label.config(text=f"第 {page}/{total} 页")
            self.page_prev_btn.config(state="normal")
            self.page_next_btn.config(state="normal")
        else:
            self.page_label.config(text="")
            self.page_prev_btn.config(state="disabled")
            self.page_next_btn.config(state="disabled")

    # ── Status ──

    def _update_status(self, text: str):
        self.status_var.set(text)

    def _init_llm_providers(self):
        providers = self.engine.list_llm_providers()
        self._llm_provider_combo["values"] = providers
        active = self.engine.active_llm_name
        if active in providers:
            self._llm_provider_var.set(active)
        elif providers:
            self._llm_provider_var.set(providers[0])

    def _on_llm_switch(self, event=None):
        name = self._llm_provider_var.get()
        if name and self.engine.switch_llm(name):
            info = self.engine.llm_provider_info(name)
            self._update_llm_indicator()
            self._update_status(
                f"已切换: {name} ({info.get('model', '?')})"
            )

    def _update_source_indicator(self):
        """Update the source badge — green when LLM results are active, 
        gray for map, blue for cache."""
        has_llm = any(src == "llm" for _, _, src in self._current_candidates)
        has_cache = any(src == "cache" for _, _, src in self._current_candidates)
        if has_llm:
            self._source_indicator.config(text="LLM ✓", bg="#4CAF50", fg="white")
        elif has_cache:
            self._source_indicator.config(text="缓存", bg="#2196F3", fg="white")
        else:
            self._source_indicator.config(text="映射", bg="#9E9E9E", fg="white")

    def _update_llm_indicator(self):
        available = self.engine.llm.available
        color = "#4CAF50" if available else "#9E9E9E"
        self._llm_dot.itemconfig(self._llm_dot_id, fill=color)

    # ── Async LLM Polling ──

    def _poll_llm(self):
        try:
            result = self.engine.llm_refine()
            if result:
                pinyin, refined, latency = result
                if pinyin and pinyin == self._current_pinyin:
                    self._add_llm_candidate(refined)
                top = refined[0] if refined else "?"
                self._update_status(
                    f"LLM 重排序: [{top}] ({latency*1000:.0f}ms)"
                )
                # Briefly flash green when LLM returns results
                self._llm_dot.itemconfig(self._llm_dot_id, fill="#00E676")
                self.root.after(600, self._update_llm_indicator)
        except Exception as exc:
            print(f"[IME GUI] _poll_llm error: {exc}")
        finally:
            # Always reschedule the poll, even on crash
            self.root.after(500, self._poll_llm)

    def _add_llm_candidate(self, reordered: list[str]):
        """Re-rank candidates using LLM's reordered list. Mark LLM items green."""
        if not reordered:
            return
        current_map = {t: (sc, s) for t, sc, s in self._current_candidates}
        llm_set = set(reordered)
        seen: set[str] = set()
        new_list: list[tuple[str, float, str]] = []
        # LLM order first — mark source as "llm" so buttons show green
        for t in reordered:
            if t not in seen and t in current_map:
                sc, _ = current_map[t]
                new_list.append((t, sc, "llm"))
                seen.add(t)
        # Append any candidates LLM didn't mention (keeps original source)
        for t, sc, s in self._current_candidates:
            if t not in seen:
                new_list.append((t, sc, s))
                seen.add(t)
        self._current_candidates = new_list
        self._current_page = 0
        self._update_candidates()
        self._update_page_nav()
        self._update_source_indicator()

    # ── Cache Persistence ──

    def _load_cache(self):
        self.engine.load_cache()
        cache_size = self.engine.cache.size()
        stats = self.engine.cache_stats()
        hit_rate = 0
        if stats["hits"] + stats["misses"] > 0:
            hit_rate = stats["hits"] / (stats["hits"] + stats["misses"]) * 100
        self._update_status(f"已加载 {cache_size} 条缓存记录")

    # ── Run ──

    def run(self):
        self.root.mainloop()
