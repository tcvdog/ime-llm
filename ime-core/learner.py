"""
User preference learning — records selections to adjust pinyin_map weights.

Architecture:
  Per-syllable tracking:  {syllable: {char: count}}
    "ni": {"你": 5, "尼": 1}  → "你" gets +0.42 boost on future "ni" queries

  Per-phrase tracking:  {pinyin_key: {phrase: count}}
    "xia mian": {"虾面": 3, "下面": 1}  → "虾面" moves to front on WORD_MAP lookup

  save() → JSON (lossless, human-readable)
  load() ← JSON

No LLM involvement. Pure frequency-based learning from user selections.
"""

import json
import os

_MAX_ENTRIES = 5000


class UserPreferenceDB:
    def __init__(self, save_path: str | None = None):
        self._save_path = save_path or os.path.expanduser(
            "~/.cache/ime-llm/user_weights.json"
        )
        self._syllable: dict[str, dict[str, float]] = {}
        self._phrase: dict[str, dict[str, int]] = {}
        self._shortcut: dict[str, dict[str, int]] = {}
        self._dirty = False
        self.load()

    # ── Public API ──

    def record(self, pinyin: str, text: str):
        """Record a user selection. pinyin is the segmented pinyin (space-separated)."""
        syllables = pinyin.strip().split()
        if len(syllables) == 1:
            self._record_syllable(syllables[0], text)
        else:
            key = " ".join(syllables)
            self._record_phrase(key, text)
        self._dirty = True

    def get_syllable_weights(self) -> dict[str, dict[str, float]]:
        """Return all syllable-level weights for pinyin_map to consume."""
        return self._syllable

    def get_top_phrase(self, pinyin_key: str) -> str | None:
        """Return the most-selected phrase for a multi-syllable pinyin key."""
        weights = self._phrase.get(pinyin_key, {})
        if not weights:
            return None
        return max(weights, key=weights.get)

    # ── Shortcut (abbreviation) preference ──

    def record_shortcut(self, abbr: str, word: str):
        """Record a user selection for a pinyin abbreviation (e.g. 'bbmm'→'爸爸妈妈')."""
        weights = self._shortcut.setdefault(abbr, {})
        weights[word] = weights.get(word, 0) + 1
        self._dirty = True

    def get_top_shortcut(self, abbr: str) -> str | None:
        """Return the most-selected word for an abbreviation, or None."""
        weights = self._shortcut.get(abbr, {})
        if not weights:
            return None
        return max(weights, key=weights.get)

    def get_shortcut_weights(self, abbr: str) -> dict[str, int]:
        """Return all preference counts for an abbreviation."""
        return dict(self._shortcut.get(abbr, {}))

    def save(self):
        """Persist weights to JSON."""
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
        self._trim()
        data = {"syllable": self._syllable, "phrase": self._phrase, "shortcut": self._shortcut}
        try:
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._dirty = False
        except OSError:
            pass

    def load(self):
        """Load weights from JSON."""
        if not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._syllable = data.get("syllable", {})
            self._phrase = data.get("phrase", {})
            self._shortcut = data.get("shortcut", {})
            self._dirty = False
        except (json.JSONDecodeError, OSError, IOError):
            pass

    def clear(self):
        self._syllable.clear()
        self._phrase.clear()
        self._shortcut.clear()
        self._dirty = True

    def stats(self) -> dict:
        return {
            "syllable_entries": len(self._syllable),
            "phrase_entries": len(self._phrase),
            "shortcut_entries": len(self._shortcut),
            "dirty": self._dirty,
            "save_path": self._save_path,
        }

    # ── Internal ──

    def _record_syllable(self, syl: str, char: str):
        """Increment weight for a single-syllable → character selection."""
        weights = self._syllable.setdefault(syl, {})
        weights[char] = weights.get(char, 0.0) + 1.0
        # Decay all weights in this syllable to prevent lock-in
        total = sum(weights.values())
        if total > 10:
            for ch in list(weights.keys()):
                weights[ch] *= 0.95
                if weights[ch] < 0.01:
                    del weights[ch]

    def _record_phrase(self, key: str, phrase: str):
        """Increment count for a multi-syllable → phrase selection."""
        weights = self._phrase.setdefault(key, {})
        weights[phrase] = weights.get(phrase, 0) + 1

    def _trim(self):
        """Prevent unbounded growth beyond _MAX_ENTRIES."""
        while len(self._syllable) > _MAX_ENTRIES:
            self._syllable.pop(next(iter(self._syllable)))
        while len(self._phrase) > _MAX_ENTRIES:
            self._phrase.pop(next(iter(self._phrase)))
        while len(self._shortcut) > _MAX_ENTRIES:
            self._shortcut.pop(next(iter(self._shortcut)))
