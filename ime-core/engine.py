"""
IME Core Engine — lightweight orchestrator.

Architecture:
  Baseline candidates from built-in pinyin_map (zero latency)
  Async LLM re-ranking for smart disambiguation (non-blocking)
  No local vocabulary / cache / Viterbi — everything delegated to LLM
"""

from __future__ import annotations

from typing import Optional

from config import DEFAULT_CONFIG
from pinyin_map import segment_pinyin, generate_candidates
from llm_backend import LLMBackend


class Engine:
    def __init__(self, config: dict = None):
        cfg = config or DEFAULT_CONFIG
        self.config = cfg

        self.llm = LLMBackend(
            endpoint=cfg["llm"]["endpoint"],
            model=cfg["llm"]["model"],
            api_key=cfg["llm"]["api_key"],
            timeout=cfg["llm"].get("timeout", 15),
        )

        # Session state
        self._context = ""
        self._context_max = 200
        self._pending_llm = None

    # ── Public API ──

    def process(self, pinyin: str) -> list[tuple[str, float, str]]:
        """
        Generate candidates from pinyin_map baseline + fire async LLM.

        Returns list of (text, score, source) where source is "map" or "llm".
        """
        if not pinyin or not pinyin.strip():
            return []

        max_candidates = self.config.get("engine", {}).get("max_candidates", 36)
        candidates = generate_candidates(pinyin, max_combinations=max_candidates)
        if not candidates:
            return []

        # Score by position (pinyin_map already orders by frequency)
        scored = [
            (text, 1.0 / (i + 1), "map")
            for i, text in enumerate(candidates)
        ]

        # Fire async LLM re-rank
        if self.llm.available:
            texts = [t for t, _, _ in scored]
            self._pending_llm = (pinyin, texts, self._context)

        return scored

    def select(self, text: str, pinyin: str):
        """Update session context when user selects a candidate."""
        self._context = (self._context + text)[-self._context_max:]

    def llm_refine(self) -> Optional[tuple[str, list[str], float]]:
        """
        Called asynchronously to re-rank candidates via LLM.

        Returns (pinyin, reordered_texts, latency_ms) or None.
        """
        if not self._pending_llm:
            return None

        pinyin, candidates, context = self._pending_llm
        self._pending_llm = None

        reordered, _, elapsed = self.llm.rank(pinyin, candidates, context)
        if reordered is None:
            return None

        return pinyin, reordered, elapsed

    @property
    def context(self) -> str:
        return self._context

    @context.setter
    def context(self, value: str):
        self._context = value

    def reset_context(self):
        self._context = ""
