"""
IME Core Engine — orchestrates three layers:

  Layer 1: Pinyin→characters map (guaranteed baseline, frequency-weighted)
  Layer 2: Local vocabulary cache (learned user preference, 0ms)
  Layer 3: LLM backend (async refinement + weight adjustment)
"""

from __future__ import annotations

from typing import Optional

from config import DEFAULT_CONFIG
from pinyin_map import segment_pinyin, generate_candidates, local_rank
from cache import Cache
from llm_backend import LLMBackend
from freq_db import CharFrequencyDB


def _text_score(text: str, freq_db: CharFrequencyDB) -> float:
    """Compute a frequency-based score for any candidate text.

    Single character: direct FreqDB weight.
    Multi-character: average of character weights.
    """
    try:
        if not text:
            return 0.0
        if len(text) == 1:
            return freq_db.get_weight(text[0])
        total = sum(freq_db.get_weight(ch) for ch in text)
        return total / len(text)
    except (AttributeError, TypeError, KeyError):
        return 1.0


class Engine:
    def __init__(self, config: dict = None):
        cfg = config or DEFAULT_CONFIG
        self.config = cfg

        # Layer 0: Character frequency database (always available)
        self.freq_db = CharFrequencyDB(
            save_path=cfg.get("freq_db", {}).get("save_path", ""),
        )
        self.freq_db.load()

        # Layer 2: local cache
        self.cache = Cache(
            context_window=cfg["cache"]["context_window"],
            save_path=cfg["cache"]["save_path"],
        )

        # Layer 3: LLM backend(s) — multi-provider support
        self.llm_providers: dict[str, LLMBackend] = {}
        providers_cfg = cfg.get("llm_providers", {})
        for name, llm_cfg in providers_cfg.items():
            self.llm_providers[name] = LLMBackend(
                endpoint=llm_cfg.get("endpoint", cfg["llm"]["endpoint"]),
                model=llm_cfg.get("model", cfg["llm"]["model"]),
                api_key=llm_cfg.get("api_key", cfg["llm"]["api_key"]),
                timeout=llm_cfg.get("timeout", cfg["llm"].get("timeout", 15)),
            )

        active_name = cfg.get("llm_active", "remote")
        self._active_llm_name = active_name if active_name in self.llm_providers else ""
        self.llm = self.llm_providers.get(self._active_llm_name) or LLMBackend(
            endpoint=cfg["llm"]["endpoint"],
            model=cfg["llm"]["model"],
            api_key=cfg["llm"]["api_key"],
        )

        # Session state
        self._context = ""
        self._context_max = 200  # keep last ~200 chars for LLM context
        self._pending_llm = None

    # ── Public API ──

    def process(self, pinyin: str) -> list[tuple[str, float, str]]:
        """
        Main pipeline: pinyin → candidates.

        Returns list of (text, score, source) where source is "map", "cache", or "llm".

        Flow:
        1. Generate all candidates from pinyin map (Layer 1) using freq weights
        2. Re-rank with cache preference (Layer 2)
        3. Re-rank with local bigram or async LLM (Layer 3)
        """
        if not pinyin or not pinyin.strip():
            return []

        # ── Step 1: Generate from pinyin map (freq-weighted) ──
        freq_db = getattr(self, "freq_db", None)
        max_candidates = self.config.get("engine", {}).get("max_candidates", 36)
        candidates = generate_candidates(
            pinyin,
            max_combinations=max_candidates,
            freq_db=freq_db,
        )
        if not candidates:
            return []

        # Score by frequency weight + position boost
        scored = [
            (text, _text_score(text, freq_db) + (1.0 / (i + 1)), "map")
            for i, text in enumerate(candidates)
        ]
        scored.sort(key=lambda x: -x[1])

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

        # ── Step 3: Re-rank candidates ──
        # When LLM is available → fire async request (non-blocking)
        # When LLM is offline → use local bigram-frequency re-ranker
        if self.llm.available:
            texts = [t for t, _, _ in scored]
            # Store Viterbi/n-gram ordering as baseline for LLM context
            self._pending_llm = (pinyin, list(texts), self._context, list(texts))
        else:
            texts = [t for t, _, _ in scored]
            reordered = local_rank(texts)
            if reordered:
                source_map = {(t, sc): s for t, sc, s in scored}
                new_scored: list[tuple[str, float, str]] = []
                seen: set[str] = set()
                for t in reordered:
                    if t not in seen:
                        for ot, osc, osrc in scored:
                            if ot == t:
                                new_scored.append((t, osc, osrc))
                                seen.add(t)
                                break
                for t, sc, s in scored:
                    if t not in seen:
                        new_scored.append((t, sc, s))
                        seen.add(t)
                scored = new_scored

        return scored

    def select(self, text: str, pinyin: str):
        """
        Called when user selects a candidate.
        - Learns the selection into cache
        - Appends to session context
        - Saves frequency adjustments
        """
        self.cache.learn(pinyin, self._context, text)

        # Boost the selected characters in frequency DB
        for ch in text:
            self.freq_db.adjust_weight(ch, 0.05)

        self._context = (self._context + text)[-self._context_max:]

    def llm_refine(self) -> Optional[tuple[str, list[str], float]]:
        """
        Called asynchronously to re-rank candidates via LLM.
        Also applies any weight adjustments returned by the LLM.
        Returns (pinyin, reordered_texts, latency_ms) or None.
        """
        if not self._pending_llm:
            return None

        pinyin, candidates, context, baseline = self._pending_llm
        self._pending_llm = None

        reordered, weight_adj, elapsed = self.llm.rank(
            pinyin, candidates, context, baseline=baseline,
        )
        if reordered is None:
            return None

        # Apply any weight adjustments from LLM
        if weight_adj:
            self.freq_db.apply_llm_adjustments(weight_adj)

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

    def save_freq(self, path: str = None):
        """Save frequency DB adjustments."""
        self.freq_db.save(path)

    def load_freq(self, path: str = None):
        """Load frequency DB adjustments."""
        self.freq_db.load(path)

    def freq_stats(self) -> dict:
        return self.freq_db.stats()

    def cache_stats(self) -> dict:
        return dict(self.cache.stats)

    def reset_context(self):
        self._context = ""

    # ── Multi-provider LLM switching ──

    def switch_llm(self, name: str) -> bool:
        """Switch the active LLM provider by name.

        Returns True if switch succeeded, False if name not found.
        """
        if name in self.llm_providers:
            self.llm = self.llm_providers[name]
            self._active_llm_name = name
            # Clear pending LLM task when switching
            self._pending_llm = None
            return True
        return False

    @property
    def active_llm_name(self) -> str:
        return self._active_llm_name

    def list_llm_providers(self) -> list[str]:
        """Return list of available LLM provider names."""
        return list(self.llm_providers.keys())

    def llm_provider_info(self, name: str = "") -> dict:
        """Return info about the active (or named) LLM provider."""
        target = self.llm_providers.get(name or self._active_llm_name)
        if target is None:
            return {}
        return {
            "name": name or self._active_llm_name,
            "model": target.model,
            "endpoint": target.endpoint,
            "available": target.available,
        }
