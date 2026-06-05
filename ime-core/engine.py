"""
IME Core Engine — parallel async LLM pipeline.

Architecture:
  Layer 0: Pinyin Map (zero latency, always available)
  Layer 1&2: Ollama + DeepSeek submitted simultaneously (async, parallel)

Whichever LLM returns first gets #1 position.
The later LLM gets #2. The user's past selections stay at #1 via phrase_boost.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config import load_config, DEFAULT_CONFIG
from pinyin_map import segment_pinyin, segment_pinyin_all, generate_candidates
from llm_backend import LLMBackend
from learner import UserPreferenceDB

log = logging.getLogger("ime-engine")

VALID_MODES = ("map_only", "map_ollama", "map_deepseek", "map_ollama_deepseek")
LLM_SOURCES = ("ollama", "deepseek")


def _build_user_hints(learner: UserPreferenceDB, pinyin_key: str) -> str:
    if not pinyin_key:
        return ""
    phrase_data = learner._phrase.get(pinyin_key, {})
    if not phrase_data:
        return ""
    items = sorted(phrase_data.items(), key=lambda x: -x[1])
    hints = "，".join(f"{w}({c}次)" for w, c in items)
    return f"用户历史：{pinyin_key} → {hints}"


class Engine:
    def __init__(self, config: dict = None):
        cfg = config or load_config()
        self.config = cfg

        # DeepSeek
        self.llm = LLMBackend(
            endpoint=cfg["llm"]["endpoint"],
            model=cfg["llm"]["model"],
            api_key=cfg["llm"]["api_key"],
            timeout=cfg["llm"].get("timeout", 15),
        )

        # Ollama
        ollama_cfg = cfg.get("ollama", {})
        self.ollama = LLMBackend(
            endpoint=ollama_cfg.get("endpoint", "http://localhost:11434/v1"),
            model=ollama_cfg.get("model", "qwen2.5:1.5b"),
            api_key="",
            timeout=ollama_cfg.get("timeout", 5),
        )

        # Mode
        mode = cfg.get("mode", "map_ollama_deepseek")
        self._mode = mode if mode in VALID_MODES else "map_ollama_deepseek"
        self._use_ollama = self._mode in ("map_ollama", "map_ollama_deepseek")
        self._use_deepseek = self._mode in ("map_deepseek", "map_ollama_deepseek")
        # Master LLM switch — overrides mode selection
        if not cfg.get("llm_enabled", True):
            self._use_ollama = False
            self._use_deepseek = False

        self._load_word_dict()

        # User preference learning
        learn_cfg = cfg.get("learning", {})
        self.learner = UserPreferenceDB(
            save_path=learn_cfg.get("save_path"),
        )

        # Session state
        self._context = ""
        self._context_max = 200
        self._llm_skipped = False
        self._predictions: list[str] = []  # next-word predictions when idle
        self._bigrams: dict[str, dict[str, int]] = {}  # {prev_text: {next_text: count}}
        self._last_committed_text: str = ""             # last selected word for bigram learning

        self._load_bigrams()

        # Compound detection
        self._selection_chain: list[tuple[str, str, float]] = []
        self._compound_window = 3.0
        # Recent single-char selections for compound mining (keep last 50)
        self._single_char_history: list[tuple[str, str]] = []
        self._single_char_max = 50

        # Parallel LLM pool (2 workers: one for Ollama, one for DeepSeek)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._futures: dict[str, object] = {}          # source -> Future
        self._applied: set[str] = set()                 # sources already merged into UI
        self._pending_map_data: dict = {}               # stored for late submissions
        self._deepseek_deferred = False                 # gatekeeper mode flag

        # LLM score feedback: {pinyin_key: {word: accumulated_weighted_score}}
        # Persisted alongside learner data to make map smarter over time
        self._llm_scores: dict[str, dict[str, float]] = {}
        self._load_llm_scores()

    def reload_config(self, config: dict = None):
        cfg = config or load_config()
        self.config = cfg
        mode = cfg.get("mode", "map_ollama_deepseek")
        self._mode = mode if mode in VALID_MODES else "map_ollama_deepseek"
        self._use_ollama = self._mode in ("map_ollama", "map_ollama_deepseek")
        self._use_deepseek = self._mode in ("map_deepseek", "map_ollama_deepseek")
        # Master LLM switch — overrides mode selection
        llm_enabled = cfg.get("llm_enabled", True)
        if not llm_enabled:
            self._use_ollama = False
            self._use_deepseek = False
        self.llm = LLMBackend(
            endpoint=cfg["llm"]["endpoint"],
            model=cfg["llm"]["model"],
            api_key=cfg["llm"]["api_key"],
            timeout=cfg["llm"].get("timeout", 15),
        )
        ollama_cfg = cfg.get("ollama", {})
        self.ollama = LLMBackend(
            endpoint=ollama_cfg.get("endpoint", "http://localhost:11434/v1"),
            model=ollama_cfg.get("model", "qwen2.5:1.5b"),
            api_key="",
            timeout=ollama_cfg.get("timeout", 5),
        )

    # ── Word dictionary loading ──

    def _load_word_dict(self):
        search_paths = [
            os.path.join(os.path.dirname(__file__), "data", "word_dict.json"),
            os.path.expanduser("~/.cache/ime-llm/word_dict.json"),
            "/usr/share/ibus-ime-llm/data/word_dict.json",
        ]
        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    word_map = data.get("word_map", {})
                    if not word_map:
                        return
                    import pinyin_map as pm
                    old_count = len(pm.WORD_MAP)
                    for pk, words in word_map.items():
                        if pk not in pm.WORD_MAP:
                            pm.WORD_MAP[pk] = words
                        else:
                            existing = set(pm.WORD_MAP[pk])
                            new_words = [w for w in words if w not in existing]
                            if new_words:
                                pm.WORD_MAP[pk] = pm.WORD_MAP[pk] + new_words
                    new_count = len(pm.WORD_MAP)
                    total_words = sum(len(v) for v in pm.WORD_MAP.values())
                    if new_count > old_count:
                        print(f"[词库] 加载 {path}: {old_count} → {new_count} 组合, {total_words} 词")
                    return
                except Exception as e:
                    print(f"[词库] 加载失败 {path}: {e}")

    # ── LLM score persistence ──

    def _llm_scores_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "llm_scores.json",
        )

    def _bigrams_path(self) -> str:
        return os.path.join(
            os.path.expanduser("~/.cache/ime-llm"), "bigrams.json",
        )

    def _load_llm_scores(self):
        path = self._llm_scores_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._llm_scores = json.load(f)
                total = sum(len(v) for v in self._llm_scores.values())
                if total > 0:
                    print(f"[LLM分数] 加载 {len(self._llm_scores)} 组合, {total} 条")
            except Exception:
                self._llm_scores = {}

    def _load_bigrams(self):
        """Load persisted bigram data."""
        path = self._bigrams_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._bigrams = json.load(f)
            except Exception:
                self._bigrams = {}

    def save_llm_scores(self):
        """Persist accumulated LLM feedback scores."""
        path = self._llm_scores_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._llm_scores, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def save_bigrams(self):
        """Persist bigram prediction data."""
        path = self._bigrams_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._bigrams, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ── Public API ──

    def process_map(self, pinyin: str) -> list[tuple[str, float, str]]:
        """Map layer only (instant). For real-time typing feedback.
        
        Called on every keystroke. Returns immediately with map candidates.
        Considers all valid pinyin segmentation paths, weighted by path score.
        """
        self._predictions.clear()
        if not pinyin or not pinyin.strip():
            return []

        max_candidates = self.config.get("engine", {}).get("max_candidates", 36)
        paths = segment_pinyin_all(pinyin.strip())
        if not paths:
            return []

        # Use best path for phrase_boost and LLM submission
        best_syllables, _ = paths[0]
        pinyin_key = " ".join(best_syllables) if len(best_syllables) > 1 else ""

        phrase_boost: dict[str, str] = {}
        if pinyin_key:
            top = self.learner.get_top_phrase(pinyin_key)
            if top:
                phrase_boost[pinyin_key] = top

        # Generate candidates from all paths, weighted by path score
        all_scored: dict[str, float] = {}
        for syllables, path_score in paths:
            pk = " ".join(syllables)
            pinyin_input = " ".join(syllables)  # Use space-separated for accurate segmentation
            candidates = generate_candidates(
                pinyin_input, max_combinations=max_candidates,
                user_weights=self.learner.get_syllable_weights(),
                phrase_boost=phrase_boost if pk == pinyin_key else {},
            )
            for i, text in enumerate(candidates):
                score = (1.0 / (i + 1)) * path_score
                if text not in all_scored or score > all_scored[text]:
                    all_scored[text] = score

        if not all_scored:
            return []

        scored = [(text, sc, "map") for text, sc in
                  sorted(all_scored.items(), key=lambda x: -x[1])]

        # Apply accumulated LLM score feedback
        if pinyin_key and pinyin_key in self._llm_scores:
            lm = self._llm_scores[pinyin_key]
            if lm:
                max_lm = max(lm.values()) or 1.0
                scored = [
                    (t, sc + lm.get(t, 0) / max_lm * 0.5, s)
                    for t, sc, s in scored
                ]
                scored.sort(key=lambda x: -x[1])

        return scored

    def process(self, pinyin: str) -> list[tuple[str, float, str]]:
        """
        Map layer + submit LLM requests (called after debounce).
        Returns map candidates immediately, LLM results come via poll_results().
        """
        if not pinyin or not pinyin.strip():
            return []

        max_candidates = self.config.get("engine", {}).get("max_candidates", 36)
        paths = segment_pinyin_all(pinyin.strip())
        if not paths:
            return []

        best_syllables, _ = paths[0]
        pinyin_key = " ".join(best_syllables) if len(best_syllables) > 1 else ""

        phrase_boost: dict[str, str] = {}
        if pinyin_key:
            top = self.learner.get_top_phrase(pinyin_key)
            if top:
                phrase_boost[pinyin_key] = top

        user_hints = _build_user_hints(self.learner, pinyin_key)

        # Generate candidates from all paths, weighted by path score
        all_scored: dict[str, float] = {}
        for syllables, path_score in paths:
            pk = " ".join(syllables)
            pinyin_input = " ".join(syllables)
            candidates = generate_candidates(
                pinyin_input, max_combinations=max_candidates,
                user_weights=self.learner.get_syllable_weights(),
                phrase_boost=phrase_boost if pk == pinyin_key else {},
            )
            for i, text in enumerate(candidates):
                score = (1.0 / (i + 1)) * path_score
                if text not in all_scored or score > all_scored[text]:
                    all_scored[text] = score

        if not all_scored:
            return []

        scored = [(text, sc, "map") for text, sc in
                  sorted(all_scored.items(), key=lambda x: -x[1])]

        # Apply accumulated LLM score feedback
        if pinyin_key and pinyin_key in self._llm_scores:
            lm = self._llm_scores[pinyin_key]
            if lm:
                max_lm = max(lm.values()) or 1.0
                scored = [
                    (t, sc + lm.get(t, 0) / max_lm * 0.5, s)
                    for t, sc, s in scored
                ]
                scored.sort(key=lambda x: -x[1])

        texts = [t for t, _, _ in scored]

        # ── Confidence skip: skip LLM if map already confident ──
        skip = self._should_skip_llm(scored, pinyin_key)

        # Cancel pending by clearing old futures
        self._futures.clear()
        self._applied.clear()

        if skip:
            self._llm_skipped = True
            return scored
        else:
            self._llm_skipped = False

        # Store map data for merging late results
        self._pending_map_data = {
            "pinyin": pinyin,
            "texts": texts,
            "context": self._context,
            "user_hints": user_hints,
            "pinyin_key": pinyin_key,
        }

        # Gatekeeper mode: submit Ollama first; DeepSeek only if Ollama disagrees
        self._deepseek_deferred = False
        if self._use_ollama and self.ollama.available:
            self._deepseek_deferred = True
            self._futures["ollama"] = self._executor.submit(
                self._do_rank, "ollama", pinyin, texts, self._context, user_hints,
            )
        elif self._use_deepseek and self.llm.available:
            # No Ollama available → submit DeepSeek directly
            self._futures["deepseek"] = self._executor.submit(
                self._do_rank, "deepseek", pinyin, texts, self._context, user_hints,
            )

        return scored

    def _should_skip_llm(self, scored: list[tuple[str, float, str]],
                         pinyin_key: str) -> bool:
        """Decide whether to skip LLM submission based on map confidence.

        Skips when:
          1. Confidence skip is enabled in config
          2. There are at least 2 candidates
          3. Top1 score / Top2 score >= ratio threshold
          4. User has selected the top word for this pinyin >= min_selections times
        """
        cfg = self.config.get("confidence", {})
        if not cfg.get("enabled", True):
            return False
        if len(scored) < 2:
            return False
        ratio_threshold = cfg.get("ratio", 1.5)
        min_sel = cfg.get("min_selections", 3)

        t1_score = scored[0][1]
        t2_score = scored[1][1]
        if t2_score <= 0:
            return False
        if t1_score / t2_score < ratio_threshold:
            return False

        if not pinyin_key:
            return False
        top_word = scored[0][0]
        phrase_counts = self.learner._phrase.get(pinyin_key, {})
        user_count = phrase_counts.get(top_word, 0)
        return user_count >= min_sel

    def _maybe_submit_deepseek(self, ollama_ranking: list[str]):
        """Gatekeeper: submit DeepSeek only if Ollama disagrees with Map.

        Compares Ollama's top 3 with Map's top 3.
        If ≥ 2 match → they agree, skip DeepSeek.
        Otherwise → submit DeepSeek for a second opinion.
        """
        data = self._pending_map_data
        if not data:
            self._deepseek_deferred = False
            return
        # If DeepSeek already submitted somehow, don't resubmit
        if "deepseek" in self._futures and "deepseek" not in self._applied:
            return

        map_top3 = data.get("texts", [])[:3]
        ollama_top3 = ollama_ranking[:3]
        overlap = len(set(map_top3) & set(ollama_top3))

        if overlap >= 2:
            # Agree → skip DeepSeek
            self._deepseek_deferred = False
            log.info("Ollama agrees with Map (%d/3 top3) → DeepSeek skipped", overlap)
            return

        # Disagree → submit DeepSeek
        self._deepseek_deferred = False
        pinyin = data.get("pinyin", "")
        texts = data.get("texts", [])
        context = data.get("context", "")
        user_hints = data.get("user_hints", "")
        if self._use_deepseek and self.llm.available and texts:
            log.info("Ollama disagrees with Map (%d/3) → submitting DeepSeek", overlap)
            self._futures["deepseek"] = self._executor.submit(
                self._do_rank, "deepseek", pinyin, texts, context, user_hints,
            )

    def _do_rank(
        self, source: str, pinyin: str, candidates: list[str],
        context: str, user_hints: str,
    ) -> tuple[str, str, list[str], float, int, int] | None:
        """Run ranking for a single LLM source. Runs in thread pool.
        Returns (source, pinyin, reordered_list, latency, prompt_tokens, completion_tokens) or None."""
        backend = self.ollama if source == "ollama" else self.llm
        if not backend.available:
            return None

        # Trim to top 15 candidates for speed (LLM results refine the top picks)
        trimmed = candidates[:15]

        # For Ollama: skip convert() (it's slow and redundant — map already has candidates)
        if source == "ollama":
            reordered, _, rank_elapsed = backend.rank(
                pinyin, trimmed, context, user_hints=user_hints,
            )
            total_elapsed = rank_elapsed
            if not reordered:
                return None
            result = list(dict.fromkeys(reordered))
            for t in trimmed:
                if t not in result:
                    result.append(t)
            return source, pinyin, result, total_elapsed, backend._last_prompt_tokens, backend._last_completion_tokens
        else:
            # DeepSeek: convert + rank
            converted, conv_elapsed = backend.convert(
                pinyin, context, user_hints=user_hints,
            )
            pt1, ct1 = backend._last_prompt_tokens, backend._last_completion_tokens
            reordered, _, rank_elapsed = backend.rank(
                pinyin, trimmed, context, user_hints=user_hints,
            )
            pt2, ct2 = backend._last_prompt_tokens, backend._last_completion_tokens
            total_elapsed = conv_elapsed + rank_elapsed

            result: list[str] = []
            seen: set[str] = set()
            if converted:
                for t in converted.split():
                    if t not in seen:
                        result.append(t)
                        seen.add(t)
            if reordered:
                for t in reordered:
                    if t not in seen:
                        result.append(t)
                        seen.add(t)
            elif not converted:
                return None
            for t in trimmed:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
            return source, pinyin, result, total_elapsed, pt1 + pt2, ct1 + ct2

    def poll_results(self) -> list[tuple[str, str, list[str], float]]:
        """Check for newly completed LLM results.

        Returns list of (source, pinyin, reordered_list, latency) for each newly done task.
        Also records token usage from DeepSeek responses to cumulative stats.
        In gatekeeper mode, submits DeepSeek if Ollama disagrees with Map.
        """
        results = []
        for source in ("ollama", "deepseek", "deepseek_refine"):
            if source not in self._futures or source in self._applied:
                continue
            future = self._futures[source]
            if not future.done():
                continue
            try:
                outcome = future.result()
                if outcome:
                    src, pinyin, reordered, latency, pt, ct = outcome
                    results.append((src, pinyin, reordered, latency))
                    # Record token usage for DeepSeek (Ollama local returns 0)
                    if src == "deepseek" and (pt or ct):
                        import token_stats
                        token_stats.record(prompt=pt, completion=ct)
                    # Gatekeeper: Ollama returned → check if DeepSeek is needed
                    if src == "ollama" and self._deepseek_deferred:
                        self._maybe_submit_deepseek(reordered)
                else:
                    # Ollama failed/returned None → submit DeepSeek if deferred
                    if source == "ollama" and self._deepseek_deferred:
                        self._maybe_submit_deepseek([])
            except Exception:
                # Ollama failed/returned None → submit DeepSeek if deferred
                if source == "ollama" and self._deepseek_deferred:
                    self._maybe_submit_deepseek([])
            self._applied.add(source)
        return results

    @property
    def llm_busy(self) -> bool:
        return any(
            s in self._futures and s not in self._applied
            and not self._futures[s].done()
            for s in self._futures
        )

    # Source confidence weights (higher = more influence on final ranking)
    # Balanced to avoid single-model dominance while still rewarding accuracy
    SOURCE_WEIGHT = {"map": 1.0, "ollama": 1.5, "deepseek": 2.0}

    @staticmethod
    def merge_ranking(
        current: list[tuple[str, float, str]],
        new_order: list[str],
        source: str,
        slot: int = 0,
    ) -> list[tuple[str, float, str]]:
        """
        Merge a new LLM ranking using weighted scoring.

        Each source has a confidence weight:
          map(1.0) < ollama(2.0) < deepseek(3.0)

        For each candidate, combined_score = base_score + sum of
        source_weight / (position + 1) from each source that ranked it.
        Final ranking is sorted by combined_score descending.

        The source label shows which source gave this candidate the
        highest individual rank (not which contributed the most score).
        """
        weight = Engine.SOURCE_WEIGHT.get(source, 2.0)
        scores: dict[str, float] = {}
        best_rank: dict[str, tuple[str, int]] = {}  # text -> (source, position)

        # Initialize from current
        for t, sc, s in current:
            scores[t] = sc
            # Preserve the best rank from previous merges
            # (a high position number = low priority placeholder)
            best_rank[t] = (s, 999)

        # Add weighted position bonus from new ranking
        for pos, text in enumerate(new_order):
            if text in scores:
                bonus = weight / (pos + 1)
                scores[text] += bonus
                # Track best single-source rank (lower position = better)
                cur_src, cur_pos = best_rank.get(text, (source, 999))
                if pos < cur_pos:
                    best_rank[text] = (source, pos)

        # Sort by combined score descending
        sorted_texts = sorted(scores.keys(), key=lambda t: -scores[t])
        seen: set[str] = set()
        result: list[tuple[str, float, str]] = []
        for t in sorted_texts:
            if t not in seen:
                label = best_rank.get(t, ("map", 999))[0]
                result.append((t, scores[t], label))
                seen.add(t)
        return result

    # ── Selection & Learning ──
    # Weight strategy:
    #   User direct selection: 1.0 (最高)
    #   LLM timely, user agrees: 0.8 (LLM推荐 + 用户确认)
    #   LLM timely, user didn't select: 0.4 (LLM推荐但用户没选)
    #   LLM late, user selected same: 0.5 (LLM后来同意用户)
    #   LLM late, user selected different: 0.15 (LLM后来不同意)
    #   LLM ranking position: weight / (pos + 1) × scenario_factor

    def _update_llm_scores(self, key: str, word: str, bonus: float):
        """Store a weighted score for a word into accumulated LLM feedback."""
        scores = self._llm_scores.setdefault(key, {})
        scores[word] = max(scores.get(word, 0), bonus)

    def select(self, text: str, pinyin: str, nav_edit: bool = False):
        syllables = segment_pinyin(pinyin.strip())
        key = " ".join(syllables) if len(syllables) > 1 else pinyin.strip()
        self.learner.record(key, text)
        self._context = (self._context + text)[-self._context_max:]

        # Word recombination: when user edits via nav keys between multi-char selections
        if nav_edit and len(text) >= 2 and hasattr(self, '_last_multi_word'):
            prev_text, prev_key = self._last_multi_word
            self._recombine_compound(prev_text, text, prev_key, key)
        # Track last multi-character word
        self._last_multi_word = (text, key) if len(text) >= 2 else None

        now = time.time()
        self._selection_chain = [
            (pk, txt, ts) for pk, txt, ts in self._selection_chain
            if now - ts < self._compound_window
        ]
        self._selection_chain.append((key, text, now))
        self._detect_compound()

        # Track single-character selections for compound mining (FIFO, last 50)
        if len(text) == 1 and " " not in key:
            self._mine_cross_compound(key, text)
            self._single_char_history.append((key, text))
            if len(self._single_char_history) > self._single_char_max:
                self._single_char_history.pop(0)

        # User direct selection → full weight in LLM scores
        self._update_llm_scores(key, text, 1.0)

        # Generate next-word predictions from context
        self._predictions = self._generate_predictions()

        # Learn bigram: if there was a previous word, record (prev → current)
        if self._last_committed_text and len(text) >= 1:
            bigram = self._bigrams.setdefault(self._last_committed_text, {})
            bigram[text] = bigram.get(text, 0) + 1
        self._last_committed_text = text

        # Re-generate predictions with updated last_committed_text for accurate bigram lookup
        self._predictions = self._generate_predictions()

        # Record what user selected for late LLM comparison
        self._last_selection: dict[str, str] = {}
        self._last_selection[key] = text

        self.learner.save()

    def feed_llm_ranking(self, pinyin: str, llm_order: list[str], source: str,
                          timely: bool = True):
        """Feed LLM ranking back into WORD_MAP with scenario-aware weights.

        Args:
            pinyin: raw pinyin input
            llm_order: ranked list from LLM
            source: "ollama" or "deepseek"
            timely: True if user was still viewing when LLM returned
        """
        if not llm_order or len(llm_order) < 2:
            return

        weight = self.SOURCE_WEIGHT.get(source, 2.0)
        syllables = segment_pinyin(pinyin.strip())
        key = " ".join(syllables) if len(syllables) > 1 else ""
        if not key:
            return

        # Determine scenario factor
        user_picked = getattr(self, '_last_selection', {}).get(key, None)
        llm_top = llm_order[0]

        if timely:
            if user_picked and user_picked == llm_top:
                scenario = 0.8  # LLM推荐 + 用户确认
            elif user_picked:
                scenario = 0.4  # LLM推荐但用户选了别的
            else:
                scenario = 0.4  # LLM推荐，用户没选
        else:
            if user_picked and user_picked == llm_top:
                scenario = 0.5  # LLM后来同意用户
            elif user_picked:
                scenario = 0.15  # LLM后来不同意
            else:
                scenario = 0.2  # LLM后来返回，用户已离开

        # Store weighted scores for all ranked words
        for pos, word in enumerate(llm_order):
            if word:
                bonus = weight / (pos + 1) * scenario
                self._update_llm_scores(key, word, bonus)

        # Update WORD_MAP ordering
        import pinyin_map as pm
        if key in pm.WORD_MAP:
            existing = pm.WORD_MAP[key]
            overlap = [w for w in llm_order if w in existing]
            if len(overlap) >= 2:
                reordered = list(dict.fromkeys(overlap + existing))
                if reordered != existing:
                    pm.WORD_MAP[key] = reordered

        # Auto-save after each LLM feedback
        self.save_llm_scores()

    def _detect_compound(self):
        chain = self._selection_chain
        if len(chain) < 2:
            return
        last_text = chain[-1][1]
        if len(last_text) != 1:
            return
        max_lookback = min(6, len(chain))
        for length in range(2, max_lookback + 1):
            segment = chain[-length:]
            times = [ts for _, _, ts in segment]
            if max(times) - min(times) > self._compound_window:
                continue
            pinyins = [pk for pk, _, _ in segment]
            texts = [t for _, t, _ in segment]
            combined_key = " ".join(pinyins)
            combined_text = "".join(texts)
            if len(combined_key.split()) >= 2 and len(combined_text) == len(combined_key.split()):
                self.learner.record(combined_key, combined_text)

    def _mine_cross_compound(self, key: str, char: str):
        """Mine compounds from adjacent single-character selections.

        When user selects single chars in sequence (e.g., '相' then '机'),
        pairs them to form '相机' and boosts it for 'xiang ji'.
        Only considers the immediately preceding single-char selection.
        """
        if len(self._single_char_history) < 1:
            return
        import pinyin_map as pm
        prev_key, prev_char = self._single_char_history[-1]
        # Check: prev_char + current_char (e.g. 相+机→相机)
        compound = prev_char + char
        combined_pinyin = f"{prev_key} {key}"
        if combined_pinyin in pm.WORD_MAP and compound in pm.WORD_MAP[combined_pinyin]:
            self._update_llm_scores(combined_pinyin, compound, 0.6)
            self.learner.record(combined_pinyin, compound)

    def _recombine_compound(self, word1: str, word2: str, key1: str, key2: str):
        """Recombine characters from two multi-character words into valid compounds.

        E.g. 首相 (shou xiang) + 机器 (ji qi) → 相机 (xiang ji), 手机 (shou ji)
        Only boosts compounds that exist in WORD_MAP.
        Called when user navigated (arrows/backspace) between selections.
        """
        import pinyin_map as pm
        syl1 = key1.split()
        syl2 = key2.split()
        for i, c1 in enumerate(word1):
            for j, c2 in enumerate(word2):
                # Forward: c1 + c2 (e.g. 相+机→相机)
                compound = c1 + c2
                pk = f"{syl1[i]} {syl2[j]}"
                if pk in pm.WORD_MAP and compound in pm.WORD_MAP[pk]:
                    self._update_llm_scores(pk, compound, 0.5)
                    self.learner.record(pk, compound)
                # Reverse: c2 + c1 (e.g. 机+相→机相 - unlikely but check)
                compound2 = c2 + c1
                pk2 = f"{syl2[j]} {syl1[i]}"
                if pk2 in pm.WORD_MAP and compound2 in pm.WORD_MAP[pk2]:
                    self._update_llm_scores(pk2, compound2, 0.3)
                    self.learner.record(pk2, compound2)

    def _generate_predictions(self) -> list[str]:
        """Generate next-word predictions from current context.

        After user commits text, predicts likely next characters/words
        based on learned bigrams (personalized) and common Chinese function words.
        Returns up to 6 candidate strings.
        """
        ctx = self._context.strip()
        if not ctx:
            return []

        # Common Chinese next-word candidates (valid after almost any word)
        COMMON_NEXT = ["是", "的", "了", "在", "有", "不", "和", "就", "也",
                       "都", "要", "会", "能", "去", "来", "好", "很", "太",
                       "更", "最", "这", "那", "一", "多", "大", "小", "上",
                       "下", "里", "个", "些", "次", "人", "事", "时", "可以",
                       "什么", "怎么", "应该", "需要", "没有"]

        # Bigram data (personalized, from user's typing history)
        bigram_hits: list[tuple[str, int]] = []
        prev_bigram = self._bigrams.get(self._last_committed_text, {})
        if prev_bigram:
            bigram_hits = sorted(prev_bigram.items(), key=lambda x: -x[1])

        # Merge: bigram first (personalized), then common words (fallback), deduped
        result = []
        for w, _ in bigram_hits:
            if w not in result:
                result.append(w)
        for w in COMMON_NEXT:
            if w not in result:
                result.append(w)
        return result[:6]

    def learn_late_llm_result(self, pinyin: str, refined: list[str]):
        if not refined:
            return
        top = refined[0]
        syllables = segment_pinyin(pinyin.strip())
        key = " ".join(syllables) if len(syllables) > 1 else pinyin.strip()
        weights = self.learner._phrase.setdefault(key, {})
        if top not in weights or weights[top] < 0.5:
            weights[top] = 0.5
            self.learner._dirty = True

    # ── Properties ──

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def context(self) -> str:
        return self._context

    @context.setter
    def context(self, value: str):
        self._context = value

    @staticmethod
    def get_token_stats() -> dict:
        """Return cumulative token usage stats (prompt_tokens, completion_tokens, calls)."""
        import token_stats
        return token_stats.get_stats()

    def reset_context(self):
        self._context = ""

    def request_deepseek_refine(self, pinyin: str, candidates: list[str],
                                 exclude: list[str], context: str = ""):
        """User-triggered re-rank: submit to DeepSeek with exclusions.
        Results come back via poll_results() as source 'deepseek_refine'.
        """
        if not self._use_deepseek or not self.llm.available:
            return
        self._futures["deepseek_refine"] = self._executor.submit(
            self._do_refine, pinyin, candidates, exclude, context,
        )

    def _do_refine(self, pinyin: str, candidates: list[str],
                   exclude: list[str], context: str,
    ) -> tuple[str, str, list[str], float, int, int] | None:
        """Run DeepSeek refine in thread pool."""
        try:
            reordered, _, elapsed = self.llm.refine_with_exclusion(
                pinyin, candidates, exclude, context,
            )
            if not reordered:
                return None
            pt = self.llm._last_prompt_tokens
            ct = self.llm._last_completion_tokens
            return "deepseek_refine", pinyin, reordered, elapsed, pt, ct
        except Exception:
            return None

    def save_learner(self):
        self.learner.save()
        self.save_llm_scores()
        self.save_bigrams()
