"""
IME Core Engine — orchestrates three layers:

  Layer 1: Pinyin→characters map (guaranteed baseline)
  Layer 2: Local vocabulary cache (learned user preference, 0ms)
  Layer 3: LLM backend (async refinement ~700ms, non-blocking)
"""

from typing import Optional

from config import DEFAULT_CONFIG
from pinyin_map import segment_pinyin, get_characters, generate_candidates
from cache import Cache
from llm_backend import LLMBackend


class Engine:
    def __init__(self, config: dict = None):
        cfg = config or DEFAULT_CONFIG
        self.config = cfg

        # Layer 1: built-in pinyin map (always available)
        #   handled directly in process()

        # Layer 2: local cache
        self.cache = Cache(
            context_window=cfg["cache"]["context_window"],
            save_path=cfg["cache"]["save_path"],
        )

        # Layer 3: LLM backend
        self.llm = LLMBackend(
            endpoint=cfg["llm"]["endpoint"],
            model=cfg["llm"]["model"],
            api_key=cfg["llm"]["api_key"],
        )

        # Session state
        self._context = ""  # accumulated text before current input
        self._pending_llm = None  # (pinyin, context) waiting for LLM

    # ── Public API ──

    def process(self, pinyin: str) -> list[tuple[str, float, str]]:
        """
        Main pipeline: pinyin → candidates.

        Returns list of (text, score, source) where source is "map", "cache", or "llm".

        Flow:
        1. Generate all candidates from pinyin map (Layer 1)
        2. Re-rank with cache preference (Layer 2)
        3. Fire async LLM request (Layer 3) to re-rank candidates
        """
        if not pinyin or not pinyin.strip():
            return []

        # ── Step 1: Generate from pinyin map ──
        candidates = generate_candidates(pinyin)
        if not candidates:
            return []

        # Score and tag
        scored = [(text, 1.0 / (i + 1), "map") for i, text in enumerate(candidates)]

        # ── Step 2: Re-rank with cache ──
        preferred = self.cache.suggest(pinyin, self._context)
        if preferred:
            found = False
            new_list = []
            for text, score, source in scored:
                if text == preferred:
                    new_list.insert(0, (text, score + 10.0, "cache"))
                    found = True
                else:
                    new_list.append((text, score, source))
            if not found:
                new_list.insert(0, (preferred, 100.0, "cache"))
            scored = new_list
            scored.sort(key=lambda x: -x[1])

        # ── Step 3: Fire async LLM to re-rank all candidates ──
        if self.llm.available:
            texts = [t for t, _, _ in scored]
            self._pending_llm = (pinyin, list(texts), self._context)

        return scored

    def select(self, text: str, pinyin: str):
        """
        Called when user selects a candidate.
        - Learns the selection into cache
        - Appends to session context
        """
        # Learn
        self.cache.learn(pinyin, self._context, text)

        # Update context
        self._context += text

    def llm_refine(self) -> Optional[tuple[str, list[str], float]]:
        """
        Called asynchronously (non-blocking) to re-rank candidates via LLM.
        Returns (pinyin, reordered_texts, latency_ms) or None if not ready.
        Called from GUI idle loop.
        """
        if not self._pending_llm:
            return None

        pinyin, candidates, context = self._pending_llm
        self._pending_llm = None

        reordered, elapsed = self.llm.rank(pinyin, candidates, context)
        if reordered is None:
            return None

        return pinyin, reordered, elapsed

    @property
    def context(self) -> str:
        return self._context

    @context.setter
    def context(self, value: str):
        self._context = value

    # ── Persistence ──

    def save_cache(self, path: str = None):
        self.cache.save(path)

    def load_cache(self, path: str = None):
        self.cache.load(path)

    def cache_stats(self) -> dict:
        return dict(self.cache.stats)

    def reset_context(self):
        self._context = ""
