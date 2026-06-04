"""
Layer 2: Local vocabulary learning cache with persistence.

Learns user's character/word preferences per context,
and saves/loads to JSON for continuity between sessions.
"""

import json
import os
import re
from collections import defaultdict
from typing import Optional

from config import DEFAULT_CONFIG


def normalize_pinyin(pinyin: str) -> str:
    return re.sub(r"\s+", " ", pinyin.strip().lower())


def extract_context_key(context: str, window: int) -> str:
    """Extract context fingerprint from preceding text."""
    if not context or not context.strip():
        return ""
    cleaned = re.sub(r"[^\u4e00-\u9fff]", "", context.strip())
    return cleaned[-window:] if cleaned else ""


class Cache:
    def __init__(self, context_window: int = None, save_path: str = None):
        cfg = DEFAULT_CONFIG["cache"]
        self.context_window = context_window or cfg["context_window"]
        self.save_path = save_path or cfg["save_path"]
        # _store[(norm_pinyin, context_key)] = {word: count}
        self._store: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.stats = {"learns": 0, "hits": 0, "misses": 0}

    # ── Learning ──

    def learn(self, pinyin: str, context: str, selected: str):
        pn = normalize_pinyin(pinyin)
        ck = extract_context_key(context, self.context_window)

        # Exact key
        key = (pn, ck)
        self._store[key][selected] += 1

        # n-gram suffix keys for fuzzy matching
        if ck:
            for start in range(len(ck) - 1):
                suffix = ck[start:]
                sk = (pn, suffix)
                if sk != key:
                    self._store[sk][selected] += 1

        self.stats["learns"] += 1

    # ── Retrieval ──

    def suggest(self, pinyin: str, context: str) -> Optional[str]:
        pn = normalize_pinyin(pinyin)
        ck = extract_context_key(context, self.context_window)

        # 1) Exact
        key = (pn, ck)
        if key in self._store:
            self.stats["hits"] += 1
            return max(self._store[key], key=self._store[key].get)

        # 2) Suffix
        for end in range(len(ck) - 1, 1, -1):
            sk = (pn, ck[-end:])
            if sk in self._store:
                self.stats["hits"] += 1
                return max(self._store[sk], key=self._store[sk].get)

        # 3) No-context fallback
        key_nc = (pn, "")
        if key_nc in self._store:
            self.stats["hits"] += 1
            return max(self._store[key_nc], key=self._store[key_nc].get)

        self.stats["misses"] += 1
        return None

    # ── Ranking helper ──

    def boost_candidates(self, candidates: list[tuple[str, float]],
                         pinyin: str, context: str) -> list[tuple[str, float]]:
        """
        Re-rank candidates: if cache has a preference, boost it.
        Each candidate is (text, score). Returns re-ranked list.
        """
        preferred = self.suggest(pinyin, context)
        if preferred is None:
            return candidates

        # Move preferred to front, boost its score
        boosted = []
        pref_found = False
        for text, score in candidates:
            if text == preferred:
                boosted.append((text, score + 10.0))
                pref_found = True
            else:
                boosted.append((text, score))
        if not pref_found:
            boosted.insert(0, (preferred, 100.0))
        boosted.sort(key=lambda x: -x[1])
        return boosted

    # ── Persistence ──

    def save(self, path: str = None):
        path = path or self.save_path
        # Convert tuple keys to string keys for JSON
        serializable = {}
        for (pn, ck), words in self._store.items():
            key_str = f"{pn}||{ck}"
            serializable[key_str] = dict(words)
        data = {
            "version": 1,
            "context_window": self.context_window,
            "stats": self.stats,
            "entries": serializable,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str = None):
        path = path or self.save_path
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.context_window = data.get("context_window", self.context_window)
        self.stats = data.get("stats", self.stats)
        for key_str, words in data.get("entries", {}).items():
            if "||" in key_str:
                pn, ck = key_str.split("||", 1)
                key = (pn, ck)
            else:
                # Legacy format — treat entire key_str as pinyin, no context
                key = (key_str, "")
            self._store[key] = defaultdict(int, words)
        # Rebuild stats
        self.stats["learns"] = sum(sum(w.values()) for w in self._store.values())

    def size(self) -> int:
        return sum(len(words) for words in self._store.values())

    def clear(self):
        self._store.clear()
        self.stats = {"learns": 0, "hits": 0, "misses": 0}
