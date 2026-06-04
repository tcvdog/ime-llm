"""
tkinter GUI for IME prototype.

Layout:
  ┌──────────────────────────────────────────┐
  │ 拼音: [_____________________________]    │
  │ 已经输入: 你好世界                        │
  │                                          │
  │ 候选: [1]虾面 [2]下面 [3]夏眠            │
  │       [4]虾免 [5]瞎面 [6]虾棉            │
  │                                          │
  │ 来源: 拼音映射  缓存:12条  命中:85%      │
  └──────────────────────────────────────────┘

Keyboard shortcuts:
  Number 1-9 → select candidate
  Space      → accept top candidate
  Enter      → accept top candidate
  Backspace  → delete last pinyin char
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


class IMEGUI:
    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or Engine()

        self.root = tk.Tk()
        self.root.title("IME 输入法原型")
        self.root.geometry("520x320")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # State
        self._current_pinyin = ""
        self._current_candidates: list[tuple[str, float, str]] = []
        self._candidate_buttons: list[tk.Button] = []

        self._build_ui()
        self._bind_keys()
        self._load_cache()
        self._poll_llm()  # start async polling loop

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
        for idx in range(9):
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
        # Pinyin input changes
        self.input_var.trace_add("write", self._on_input_change)

        # Number keys for candidate selection
        # Bind on the Entry widget with "break" to prevent character insertion
        for i in range(1, 10):
            self.input_entry.bind(
                str(i),
                lambda e, idx=i: self._select(idx - 1) or "break",
            )

        # Space / Enter → top candidate
        self.input_entry.bind("<space>", lambda e: self._select(0) or "break")
        self.input_entry.bind("<Return>", lambda e: self._select(0) or "break")

    # ── Event Handlers ──

    def _on_input_change(self, *_):
        pinyin = self.input_var.get()
        if pinyin == self._current_pinyin:
            return
        self._current_pinyin = pinyin

        if not pinyin.strip():
            self._clear_candidates()
            return

        # Process through engine
        candidates = self.engine.process(pinyin)
        self._current_candidates = candidates
        self._update_candidates()

        # Update status
        self._update_status(f"来源: 拼音映射")

    def _select(self, index: int):
        if index < 0 or index >= len(self._current_candidates):
            return
        text, score, source = self._current_candidates[index]
        pinyin = self._current_pinyin

        # Engine learns
        self.engine.select(text, pinyin)

        # Clear input
        self.input_var.set("")
        self._current_pinyin = ""
        self._current_candidates = []
        self._clear_candidates()

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

    # ── Candidate Display ──

    def _update_candidates(self):
        for idx, btn in enumerate(self._candidate_buttons):
            if idx < len(self._current_candidates):
                text, score, source = self._current_candidates[idx]
                label = f"[{idx+1}] {text}"
                btn.config(text=label, state="normal")

                # Color by source
                if source == "cache":
                    btn.config(bg=COLOR_CACHE, fg="white")
                elif source == "llm":
                    btn.config(bg=COLOR_LLM, fg="white")
                else:
                    btn.config(bg=COLOR_CANDIDATE, fg="#333")

                # Click handler
                btn.config(command=lambda i=idx: self._select(i))
            else:
                btn.config(text="  ", state="disabled", bg=COLOR_CANDIDATE, fg="#333")

    def _clear_candidates(self):
        for btn in self._candidate_buttons:
            btn.config(text="  ", state="disabled", bg=COLOR_CANDIDATE, fg="#333")

    # ── Status ──

    def _update_status(self, text: str):
        self.status_var.set(text)

    # ── Async LLM Polling ──

    def _poll_llm(self):
        """Poll for async LLM refinement result."""
        result = self.engine.llm_refine()
        if result:
            pinyin, refined, latency = result
            # If user is still on the same input, add LLM result to candidates
            if pinyin and pinyin == self._current_pinyin:
                self._add_llm_candidate(refined)
            self._update_status(
                f"LLM: {refined} ({latency*1000:.0f}ms)"
            )

        self.root.after(500, self._poll_llm)

    def _add_llm_candidate(self, llm_text: str):
        """Insert LLM result into current candidate list if not already visible."""
        existing_texts = [t for t, _, _ in self._current_candidates]
        if llm_text in existing_texts:
            return
        self._current_candidates.insert(0, (llm_text, 200.0, "llm"))
        if len(self._current_candidates) > 9:
            self._current_candidates.pop()
        self._update_candidates()

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
