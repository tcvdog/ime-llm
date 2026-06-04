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

import tkinter as tk
from tkinter import ttk
from typing import Optional

from engine import Engine
from config import DEFAULT_CONFIG


FONT_LARGE = ("Microsoft YaHei", 14)
FONT_SMALL = ("Microsoft YaHei", 10)
FONT_MONO = ("Consolas", 12)

COLOR_LLM = "#4CAF50"
COLOR_MAP = "#9E9E9E"
COLOR_BG = "#F5F5F5"
COLOR_CANDIDATE = "#E3F2FD"
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
        self.root.title("IME 输入法原型 - LLM模式")
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

        # Candidate area (3 rows x 3 columns)
        cand_frame = tk.Frame(self.root, bg=COLOR_BG)
        cand_frame.pack(fill="x", padx=12, pady=4)
        tk.Label(cand_frame, text="候选:", font=FONT_SMALL, bg=COLOR_BG).pack(anchor="w")

        self.cand_grid = tk.Frame(cand_frame, bg=COLOR_BG)
        self.cand_grid.pack(fill="x", pady=2)

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

        # Page navigation
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

        # Source indicator
        self._source_indicator = tk.Label(
            page_frame, text="映射", font=FONT_SMALL,
            bg="#9E9E9E", fg="white", padx=6,
        )
        self._source_indicator.pack(side="right", padx=(4, 0))

        # Status bar
        status_frame = tk.Frame(self.root, bg="#E0E0E0", bd=1, relief="sunken")
        status_frame.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="就绪")
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
        self._update_status("来源: 拼音映射")
        self._update_source_indicator()

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
        self._update_status(f"输入拼音: {pinyin}")

    # ── Candidate Selection ──

    def _select(self, index: int):
        global_index = self._current_page * self._page_size + index
        if global_index < 0 or global_index >= len(self._current_candidates):
            return
        text, score, source = self._current_candidates[global_index]
        pinyin = self._current_pinyin

        self.engine.select(text, pinyin)

        self.input_var.set("")
        self._current_pinyin = ""
        self._current_candidates = []
        self._clear_candidates()
        self._update_page_nav()

        ctx = self.engine.context
        self.context_var.set(ctx if ctx else "(空)")

        if source == "llm":
            self._update_status(f"LLM 推荐  |  选择了: {text}")
        else:
            self._update_status(f"选择了: {text}")

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
        self.context_var.set(ctx if ctx else "(空)")

    def _append_to_context(self, text: str):
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
                if source == "llm":
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

    def _update_source_indicator(self):
        has_llm = any(src == "llm" for _, _, src in self._current_candidates)
        if has_llm:
            self._source_indicator.config(text="LLM ✓", bg="#4CAF50", fg="white")
        else:
            self._source_indicator.config(text="映射", bg="#9E9E9E", fg="white")

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
        except Exception as exc:
            print(f"[IME GUI] _poll_llm error: {exc}")
        finally:
            self.root.after(500, self._poll_llm)

    def _add_llm_candidate(self, reordered: list[str]):
        """Re-rank candidates using LLM's reordered list."""
        if not reordered:
            return
        current_map = {t: (sc, s) for t, sc, s in self._current_candidates}
        seen: set[str] = set()
        new_list: list[tuple[str, float, str]] = []
        for t in reordered:
            if t not in seen and t in current_map:
                sc, _ = current_map[t]
                new_list.append((t, sc, "llm"))
                seen.add(t)
        for t, sc, s in self._current_candidates:
            if t not in seen:
                new_list.append((t, sc, s))
                seen.add(t)
        self._current_candidates = new_list
        self._current_page = 0
        self._update_candidates()
        self._update_page_nav()
        self._update_source_indicator()

    # ── Run ──

    def run(self):
        self.root.mainloop()
