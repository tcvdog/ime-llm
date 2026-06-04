"""
Bigram language model for Chinese IME.

Probability model: P(char_i | char_{i-1}) with add-k smoothing
and linear interpolation with unigram probabilities (libpinyin-style λ).

Data sourced from gen_bigram_data.py → bigram_counts.json + unigram_counts.json.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Default add-k smoothing parameter (libpinyin uses 0.1) ────
_DEFAULT_K = 0.1

# ── Interpolation weight: λ * bigram + (1-λ) * unigram ──────
# Higher λ = more bigram influence, lower λ = more unigram fallback
_DEFAULT_LAMBDA = 0.7

# ── Unigram weight for initial emission: P(char) = unigram_weight
# Applied as: P(char|syllable) ∝ freq_score * (1 + unigram_weight * P(char))
# This helps common characters get a small boost regardless of syllable context.
_UNIGRAM_EMISSION_WEIGHT = 0.01


def _load_counts(path: str) -> dict[str, int]:
    """Load JSON counts, returning empty dict on failure."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_singleton_char_freq() -> dict[str, float]:
    """Load PYINYIN_ORDER from freq_db to build per-syllable char frequency."""
    # We use freq_db directly at runtime; this is a fallback fallback
    return {}


class BigramModel:
    """N-gram language model for Chinese character sequences.

    Provides:
      - P(char_i | char_{i-1})  — transition probability (bigram)
      - P(char)                 — unigram probability
      - P(char | syllable)      — emission probability (delegated to freq_db)

    Smoothing: add-k (Laplace-style) with linear interpolation (λ).
    """

    def __init__(
        self,
        k: float = _DEFAULT_K,
        interpolation_lambda: float = _DEFAULT_LAMBDA,
    ):
        self.k = k
        self.lambd = interpolation_lambda  # λ weight for bigram vs unigram

        # Load data
        bg_path = os.path.join(_DATA_DIR, "bigram_counts.json")
        ug_path = os.path.join(_DATA_DIR, "unigram_counts.json")

        self._bg_counts: dict[str, int] = _load_counts(bg_path)
        self._ug_counts: dict[str, int] = _load_counts(ug_path)

        # Precompute unigram probabilities
        self._total_chars = sum(self._ug_counts.values()) or 1
        self._vocab_size = len(self._ug_counts) or 1  # V for smoothing

        # Precompute unigram probabilities: P(char)
        self._unigram_prob: dict[str, float] = {
            ch: cnt / self._total_chars
            for ch, cnt in self._ug_counts.items()
        }

        # Precompute per-char totals: sum_{c2} count(c1, c2)
        self._char_prefix_sum: dict[str, int] = {}
        for bigram, cnt in self._bg_counts.items():
            c1 = bigram[0]
            self._char_prefix_sum[c1] = self._char_prefix_sum.get(c1, 0) + cnt

    # ── Public API ──

    def bigram_prob(self, c1: str, c2: str) -> float:
        """P(c2 | c1) — bigram transition probability with add-k smoothing.

        Uses interpolation: λ * P_bigram(c2|c1) + (1-λ) * P_unigram(c2)

        This ensures:
          - Unknown bigrams still get some probability (unigram fallback)
          - Rare bigrams don't get zero probability (add-k)
          - Common characters still have reasonable transition probability
        """
        # Bigram count with add-k
        bg_key = c1 + c2
        bg_count = self._bg_counts.get(bg_key, 0)
        c1_total = self._char_prefix_sum.get(c1, 0)

        # Add-k smoothed bigram probability
        # P_bigram = (count(c1,c2) + k) / (sum_{c'} count(c1,c') + k * V)
        p_bigram = (bg_count + self.k) / (c1_total + self.k * self._vocab_size)

        # Unigram probability
        p_unigram = self._unigram_prob.get(c2, 1.0 / self._total_chars)

        # Linear interpolation
        return self.lambd * p_bigram + (1.0 - self.lambd) * p_unigram

    def unigram_prob(self, char: str) -> float:
        """P(char) — standalone probability of a character."""
        return self._unigram_prob.get(char, 1.0 / self._total_chars)

    def emission_score(self, char: str, freq_weight: float) -> float:
        """Combined emission score for a character.

        This is NOT a strict probability — it combines:
          1. freq_weight: the character's frequency in its syllable
             (from CharFrequencyDB / web corpus)
          2. unigram boost: slight bump for globally common characters

        Used in Viterbi as the observation likelihood.
        """
        unigram_boost = self.unigram_prob(char) * _UNIGRAM_EMISSION_WEIGHT
        return max(freq_weight + unigram_boost, 0.001)

    def sequence_score(self, chars: list[str], freq_weights: list[float]) -> float:
        """Score a full character sequence (log-probability).

        score = Σ log(P_emission(char_i | syllable_i))
                + Σ log(P_transition(char_i | char_{i-1}))

        Higher = more likely.
        """
        if not chars:
            return float("-inf")

        score = 0.0
        for i, ch in enumerate(chars):
            # Emission
            fw = freq_weights[i] if i < len(freq_weights) else 0.1
            score += math.log(self.emission_score(ch, fw))

        # Transitions
        for i in range(1, len(chars)):
            score += math.log(self.bigram_prob(chars[i - 1], chars[i]))

        return score

    def chars_in_vocab(self) -> set[str]:
        """Return all characters known to the model."""
        return set(self._ug_counts.keys())

    def stats(self) -> dict:
        return {
            "bigrams": len(self._bg_counts),
            "vocab_chars": self._vocab_size,
            "total_unigram_count": self._total_chars,
            "k": self.k,
            "lambda": self.lambd,
        }
