"""
Character Frequency Database — pre-computed weights + runtime LLM adjustments.

Architecture:
  - PINYIN_ORDER loaded from data/pinyin_map_ext.json (3512 chars, 407 syllables)
  - Frequency scores derived from rank position (logarithmic decay)
  - CharFrequencyDB wraps with runtime adjustable weights and persistence

Each character has a base frequency score (from corpus statistics)
plus an optional runtime adjustment (from LLM learning).

Adjustments persist across sessions via save/load.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional


def _compute_score(pos: int, total: int) -> float:
    """Compute frequency score from position in ordered list.

    Score decays exponentially: top char ≈ 100, bottom char ≈ 0.1.
    """
    if total <= 1:
        return 50.0
    raw = 100.0 * math.exp(-2.5 * pos / total)
    return max(0.1, round(raw, 1))


def _load_pinyin_order() -> dict[str, list[str]]:
    """Load extended pinyin order from data file, falling back to SYLLABLE_MAP."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    data_path = os.path.join(data_dir, "pinyin_map_ext.json")
    if os.path.exists(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: build from SYLLABLE_MAP
    from pinyin_map import SYLLABLE_MAP
    return {k: list(v) for k, v in SYLLABLE_MAP.items()}


# Load once at module level
_PINYIN_ORDER: dict[str, list[str]] = _load_pinyin_order()


def build_char_base() -> dict[str, float]:
    """Build base frequency table from PINYIN_ORDER."""
    base: dict[str, float] = {}
    for syllable, chars in _PINYIN_ORDER.items():
        total = len(chars)
        for i, ch in enumerate(chars):
            score = _compute_score(i, total)
            if ch not in base or score > base[ch]:
                base[ch] = score
    return base


def build_syllable_index(
    base: dict[str, float]
) -> dict[str, list[tuple[str, float]]]:
    """Build syllable -> [(char, score)] index sorted by score desc."""
    idx: dict[str, list[tuple[str, float]]] = {}
    for syllable, chars in _PINYIN_ORDER.items():
        entries = [(ch, base.get(ch, 0.1)) for ch in chars]
        entries.sort(key=lambda x: -x[1])
        idx[syllable] = entries
    return idx


# Module-level precomputed data
_CHAR_BASE: dict[str, float] = build_char_base()
_SYLLABLE_IDX: dict[str, list[tuple[str, float]]] = build_syllable_index(_CHAR_BASE)


class CharFrequencyDB:
    """Character frequency database with runtime-adjustable weights.

    Each character has a base frequency score (from corpus statistics)
    plus an optional runtime adjustment (from LLM learning).

    Adjustments persist across sessions via save/load.
    """

    def __init__(self, save_path: str = ""):
        self._base: dict[str, float] = dict(_CHAR_BASE)
        self._syllable_idx: dict[str, list[tuple[str, float]]] = {}
        self._adjustments: dict[str, float] = {}

        for k, v in _SYLLABLE_IDX.items():
            self._syllable_idx[k] = list(v)

        self._save_path = save_path or os.path.expanduser(
            "~/.cache/ime-llm/char_freq_adjustments.json"
        )

    # -- Public API --

    def get_weight(self, char: str) -> float:
        """Get current weight for a character (base + adjustment).

        The effective weight is clamped to [0.01, 200.0] to prevent
        extreme values, but the raw adjustment delta is NOT clamped
        so that successive small increments (e.g. select() +0.05) apply.
        """
        base = self._base.get(char, 0.1)
        adj = self._adjustments.get(char, 0.0)
        return max(0.01, min(200.0, base + adj))

    def get_syllable_chars(self, syllable: str
                           ) -> list[tuple[str, float]]:
        """Get characters for a pinyin syllable, sorted by current weight."""
        syllable = syllable.lower()
        base_list = self._syllable_idx.get(syllable)
        if base_list is None:
            return []

        if not self._adjustments:
            return base_list

        adjusted = []
        for ch, base_score in base_list:
            adj = self._adjustments.get(ch, 0.0)
            adjusted.append((ch, max(0.01, min(200.0, base_score + adj))))
        adjusted.sort(key=lambda x: -x[1])
        return adjusted

    def get_top_characters(self, syllable: str, max_count: int = 10
                           ) -> list[str]:
        chars = self.get_syllable_chars(syllable)
        return [ch for ch, _ in chars[:max_count]]

    def adjust_weight(self, char: str, delta: float):
        current = self._adjustments.get(char, 0.0)
        self._adjustments[char] = current + delta

    def apply_llm_adjustments(self, adjustments: dict[str, float]):
        for char, delta in adjustments.items():
            self.adjust_weight(char, delta)

    def get_all_adjustments(self) -> dict[str, float]:
        return dict(self._adjustments)

    def get_syllable_for_char(self, char: str) -> Optional[str]:
        for syllable, chars in self._syllable_idx.items():
            for c, _ in chars:
                if c == char:
                    return syllable
        return None

    def total_base_chars(self) -> int:
        return len(self._base)

    # -- Persistence --

    def save(self, path: str = ""):
        save_path = path or self._save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {"adjustments": self._adjustments, "version": 2}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str = ""):
        load_path = path or self._save_path
        if not os.path.exists(load_path):
            return
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.get("adjustments", {}).items():
                if isinstance(k, str) and isinstance(v, (int, float)):
                    self._adjustments[k] = float(v)
        except (json.JSONDecodeError, OSError):
            pass

    def stats(self) -> dict:
        return {
            "total_chars": len(self._base),
            "adjustments_count": len(self._adjustments),
            "syllables_count": len(self._syllable_idx),
        }
