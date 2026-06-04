"""
LLM backend for pinyin→character conversion (single LLM provider).

The LLM is the PRIMARY disambiguation engine. It receives:
  - The pinyin input (mixed 简拼/全拼/continuous)
  - A baseline candidate list from pinyin_map
  - The preceding text context

And returns a re-ranked candidate list based on its Chinese language knowledge.
"""

import json
import time
import urllib.request
import urllib.error
from collections import OrderedDict
from typing import Optional

from config import DEFAULT_CONFIG

# Cache capacity (LRU eviction)
_MAX_CACHE = 64

RANK_PROMPT = (
    "You are the disambiguation engine for a Chinese pinyin IME. "
    "Your job is to re-rank the candidate list so the most likely "
    "character(s) come first.\n\n"
    "Input includes:\n"
    "  - The pinyin the user typed (may mix 简拼 initials and 全拼 full syllables)\n"
    "  - A candidate list from the pinyin map (ordered by character frequency)\n"
    "  - The preceding context (typed so far)\n\n"
    "Rules:\n"
    "  1. Re-rank candidates from MOST to LEAST likely given the context.\n"
    "  2. Output EXACT candidates from the list — do NOT modify or create new ones.\n"
    "  3. Use your knowledge of Chinese vocabulary, grammar, and common usage.\n"
    "  4. Context is critical — after '我吃了' the next word is very different "
    "from after '我想去'.\n"
    "  5. If no context, rely on word frequency and common collocations.\n"
    "  6. Output format: candidate1,candidate2,candidate3,...\n"
    "     (comma-separated, no extra text, no English, no explanation)"
)

_CONV_PROMPT = (
    "You are a Chinese pinyin-to-hanzi converter for a smart IME. "
    "Input is pinyin (space-separated or continuous) with optional context.\n\n"
    "CRITICAL: The input may contain 简拼 (shorthand initials). "
    "Interpret single letters as common Chinese abbreviations:\n"
    "  w → 我, n → 你/那, t → 他/她/它, d → 的/地/得,\n"
    "  l → 了/里, h → 和/好, s → 是/上/说, m → 吗/没/么,\n"
    "  g → 个/过/高, y → 有/也/又, b → 不/把/被,\n"
    "  j → 就/几/叫, x → 想/下/小, z → 在/做/走,\n"
    "  c → 吃/从/才, zh → 这/着/只, ch → 出/吃/长,\n"
    "  sh → 是/说/上\n\n"
    "For multi-letter abbreviations, each letter is an initial of a syllable:\n"
    "  xh → xi huan (喜欢), wm → wo men (我们), jr → jin tian (今天),\n"
    "  zj → zai jian (再见), mj → ming jian (明天), etc.\n\n"
    "Output ONLY the converted Chinese text — no explanation, no punctuation "
    "(unless required by the original meaning), no extra text, no English. "
    "If multiple conversions are plausible, choose the one best matching the context."
)


def _cache_key(pinyin: str, context: str) -> str:
    return f"{pinyin}||{context}"


class LLMBackend:
    def __init__(self, endpoint: str = None, model: str = None, api_key: str = None,
                 timeout: int = None):
        cfg = DEFAULT_CONFIG["llm"]
        self.endpoint = (endpoint or cfg["endpoint"]).rstrip("/")
        self.model = model or cfg["model"]
        self.api_key = api_key or cfg["api_key"]
        self.timeout = timeout if timeout is not None else cfg.get("timeout", 15)

        self._cache: OrderedDict[str, tuple[list[str], float]] = OrderedDict()
        self._last_result: Optional[str] = None

    @property
    def available(self) -> bool:
        if self.api_key:
            return True
        for local_prefix in ("http://localhost", "http://127.0.0.1", "http://0.0.0.0"):
            if self.endpoint.startswith(local_prefix):
                return True
        return False

    def _call(self, system: str, user: str, max_tokens: int = 256
              ) -> tuple[Optional[str], float]:
        """Generic LLM call. Returns (text, latency_seconds)."""
        url = f"{self.endpoint}/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        start = time.time()

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None, time.time() - start

        elapsed = time.time() - start

        try:
            output = body["choices"][0]["message"]["content"].strip()
            self._last_result = output
            return output, elapsed
        except (KeyError, IndexError, TypeError):
            return None, elapsed

    def _parse_rank_output(self, raw: str, candidates: list[str]) -> list[str]:
        """Parse LLM rank output into reordered list."""
        parsed: list[str] = []
        seen: set[str] = set()
        for token in raw.replace("，", ",").split(","):
            t = token.strip().strip("\"'「」『』（）()")
            if t and t in candidates and t not in seen:
                parsed.append(t)
                seen.add(t)

        # Append any candidates the LLM missed (preserve original order)
        for c in candidates:
            if c not in seen:
                parsed.append(c)
                seen.add(c)

        return parsed

    # ── Public API ──

    def convert(self, pinyin: str, context: str = "",
                user_hints: str = "") -> tuple[Optional[str], float]:
        """LLM picks the single best conversion for a pinyin string."""
        if not self.available:
            return None, 0.0

        parts = []
        if context:
            parts.append(f"上下文：{context}")
        if user_hints:
            parts.append(user_hints)
        parts.append(f"拼音：{pinyin}")
        user_msg = "\n".join(parts)
        return self._call(_CONV_PROMPT, user_msg, max_tokens=64)

    def suggest_sentence(
        self, pinyin: str, context: str = ""
    ) -> tuple[Optional[str], float]:
        """LLM suggests the full sentence conversion."""
        if not self.available:
            return None, 0.0

        sent_prompt = (
            "You are a Chinese IME for full-sentence pinyin input. "
            "Convert the ENTIRE pinyin string to Chinese characters. "
            "Output ONLY the Chinese text — no explanation, no punctuation. "
            "Use context to disambiguate. Output must be the same length "
            "(in characters) as the number of syllables in the pinyin."
        )
        user_msg = f"上下文：{context}\n拼音：{pinyin}" if context else f"拼音：{pinyin}"
        return self._call(sent_prompt, user_msg, max_tokens=128)

    def rank(
        self,
        pinyin: str,
        candidates: list[str],
        context: str = "",
        user_hints: str = "",
    ) -> tuple[Optional[list[str]], None, float]:
        """Re-rank candidates by contextual likelihood.

        Args:
            pinyin: the pinyin input string
            candidates: candidate list to re-rank
            context: preceding text context
            user_hints: optional user preference data (e.g. "用户历史：虾面(5次)")

        Returns:
            (reordered_list | None, None, latency_seconds)
        """
        if not self.available or not candidates:
            return None, None, 0.0

        # Check cache
        ck = _cache_key(pinyin, context)
        if ck in self._cache:
            cached_reordered, _ = self._cache.pop(ck)
            self._cache[ck] = (cached_reordered, time.time())
            return cached_reordered, None, 0.0

        # Build user message
        candidate_str = "，".join(candidates)
        parts = [f"拼音：{pinyin}", f"候选词：{candidate_str}"]
        if context:
            parts.insert(0, f"上下文：{context}")
        if user_hints:
            parts.append(user_hints)
        user_msg = "\n".join(parts)

        raw, elapsed = self._call(RANK_PROMPT, user_msg, max_tokens=512)
        if raw is None:
            return None, None, elapsed

        parsed = self._parse_rank_output(raw, candidates)

        # Cache the result
        if parsed:
            self._cache[ck] = (parsed, time.time())
            while len(self._cache) > _MAX_CACHE:
                self._cache.popitem(last=False)

        return parsed, None, elapsed

    def clear_cache(self):
        self._cache.clear()

    def stats(self) -> dict:
        return {
            "cached": len(self._cache),
        }
