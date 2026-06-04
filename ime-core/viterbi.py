"""
Viterbi decoder with beam search for Chinese pinyin input.

Given a sequence of pinyin syllables and their candidate characters,
finds the most likely character sequences using:
  - Emission probabilities (char frequency within each syllable)
  - Transition probabilities (bigram model between characters)

This implements a standard HMM Viterbi with beam search (width=k),
keeping the top-k partial hypotheses at each step.

Reference: libpinyin's PhoneticLookup<2,3> trellis search.
"""

from __future__ import annotations

import math
from typing import Optional

from bigram_model import BigramModel


def _softmax(log_scores: dict[str, float]) -> dict[str, float]:
    """Convert log scores to probabilities (softmax)."""
    scores = list(log_scores.values())
    max_log = max(scores) if scores else 0.0
    exp_scores = {k: math.exp(v - max_log) for k, v in log_scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {k: v / total for k, v in exp_scores.items()}


def viterbi_decode(
    syllables: list[str],
    syllable_chars: list[list[tuple[str, float]]],
    bigram_model: Optional[BigramModel] = None,
    beam_size: int = 5,
    max_results: int = 9,
) -> list[list[tuple[str, float]]]:
    """Viterbi beam search over pinyin syllables.

    Args:
        syllables: list of pinyin syllables (e.g. ["wo", "shi"])
        syllable_chars: for each syllable, [(char, freq_score), ...]
                        sorted by frequency descending
        bigram_model: BigramModel instance (or None for unigram-only)
        beam_size: top-K partial hypotheses kept at each step
        max_results: max full sequences to return

    Returns:
        list of (sequence, cumulative_score) sorted by score descending.
        Each sequence is a list of characters with same length as syllables.
    """
    if not syllables or not syllable_chars:
        return []

    n_syllables = len(syllables)
    n_syllable_chars = len(syllable_chars)

    if n_syllables == 0 or n_syllable_chars == 0:
        return []

    # Pad syllable_chars to match syllables length if needed
    while len(syllable_chars) < n_syllables:
        syllable_chars.append([])

    model = bigram_model or BigramModel()

    # ── Step 1: Initialize (first syllable) ──
    # Each hypothesis: (score, [char_sequence], used_chars_set_for_dedup)
    if not syllable_chars[0]:
        return []

    beam: list[tuple[float, list[str], set[str]]] = []
    for ch, fw in syllable_chars[0]:
        em_score = model.emission_score(ch, fw)
        # Initial score = emission + unigram prob (no transition yet)
        log_score = math.log(em_score + 1e-10)
        beam.append((log_score, [ch], {ch}))

    # Sort beam by score desc, keep top beam_size
    beam.sort(key=lambda x: -x[0])
    beam = beam[:beam_size]

    # ── Step 2: Recurse for remaining syllables ──
    for step in range(1, n_syllables):
        if not syllable_chars[step]:
            # Empty syllable: extend beam with repeats (shouldn't happen)
            beam = [(s + math.log(0.5), seq + [""], used) for s, seq, used in beam]
            continue

        candidates: list[tuple[float, list[str], set[str]]] = []

        for prev_score, prev_seq, prev_used in beam:
            if not prev_seq:
                prev_char = ""
            else:
                prev_char = prev_seq[-1]

            for ch, fw in syllable_chars[step]:
                # Emission probability
                em_score = model.emission_score(ch, fw)
                log_em = math.log(em_score + 1e-10)

                # Transition probability (bigram)
                if prev_char and model:
                    trans_prob = model.bigram_prob(prev_char, ch)
                    log_trans = math.log(trans_prob + 1e-10)
                else:
                    log_trans = 0.0

                new_score = prev_score + log_em + log_trans
                new_seq = prev_seq + [ch]
                new_used = prev_used | {ch}
                candidates.append((new_score, new_seq, new_used))

        # Keep top beam_size candidates
        candidates.sort(key=lambda x: -x[0])
        beam = candidates[:beam_size]

    # ── Step 3: Return top sequences ──
    beam.sort(key=lambda x: -x[0])

    # Deduplicate by full text
    seen_texts: set[str] = set()
    results: list[list[tuple[str, float]]] = []

    for score, seq, _ in beam:
        text = "".join(seq)
        if text not in seen_texts:
            seen_texts.add(text)
            # Normalise: pair each char with its score share
            results.append([(ch, score) for ch in seq])

        if len(results) >= max_results:
            break

    return results


def viterbi_rerank(
    candidates: list[str],
    syllables: list[str],
    bigram_model: Optional[BigramModel] = None,
) -> list[str]:
    """Re-rank candidate strings using bigram sequence score.

    Args:
        candidates: list of full-text candidates (e.g. ["下面", "虾面", "夏眠"])
        syllables: the syllable segmentation
        bigram_model: BigramModel instance

    Returns:
        candidates re-ranked by bigram sequence score (descending).
        Ties broken by original position.
    """
    model = bigram_model or BigramModel()

    def score(text: str) -> float:
        if len(text) <= 1:
            # Single char: use unigram
            return math.log(model.unigram_prob(text[0]) + 1e-10)

        total = 0.0
        for i in range(len(text)):
            # Use placeholder freq weight = 1.0 (we only need relative ranking)
            total += math.log(model.emission_score(text[i], 1.0) + 1e-10)
        for i in range(1, len(text)):
            total += math.log(model.bigram_prob(text[i - 1], text[i]) + 1e-10)
        return total

    scored = [(c, score(c), i) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: (-x[1], x[2]))
    return [c for c, _, _ in scored]
