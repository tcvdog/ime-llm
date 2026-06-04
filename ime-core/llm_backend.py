"""
Layer 3: LLM backend for async pinyin→character refinement.

Supports three modes:
  - convert():   single best guess (legacy)
  - rank():      re-rank a list of candidates + optional character weight adjustments
  - suggest_sentence(): full-sentence conversion from continuous pinyin

Features:
  - Result caching (per pinyin+context) to avoid redundant API calls
  - Viterbi baseline included in prompt for more informed LLM ranking
  - Compact 'adjust' block for character frequency learning
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
    "You are a Chinese IME ranking assistant. "
    "Given pinyin input, a list of candidate conversions, and the preceding context, "
    "re-rank the candidates from most to least likely.\n\n"
    'Output format:\n'
    'candidate1,candidate2,candidate3,...\n'
    '{"adjust": {"字": +0.5, "词": -0.3}}\n\n'
    "Rules:\n"
    "  - Separate candidates with commas.\n"
    "  - Output EXACT candidates from the list; do not modify or create new ones.\n"
    "  - Use the preceding context to pick the most natural continuation.\n"
    "    For example, after '我' prefer '是' over '十'; after '今天' prefer '天气'.\n"
    "  - If no context, rely on word frequency and common collocations.\n"
    "  - Optional: append a JSON object with 'adjust' dict for character weight "
    "changes (+0.1 to +2.0 boost, -0.1 to -2.0 reduce). Small changes (0.1-0.5) "
    "for subtle shifts, larger (1.0-2.0) for strong preferences. "
    "Only include characters you're confident about.\n"
    "  - If no adjustments needed, omit the JSON object.\n"
    '  - Keep JSON compact on one line: {"adjust":{"字":0.5}}\n'
    '  - Do NOT output anything before the candidate list.'
)

_CONV_PROMPT = (
    "You are a Chinese pinyin-to-hanzi converter. "
    "Input is pinyin syllables (space-separated or continuous) with optional context. "
    "Output ONLY the converted Chinese text — no explanation, no punctuation "
    "(unless required by the original meaning), no extra text, no English. "
    "If multiple conversions are plausible, choose the one best matching the context."
)


def _cache_key(pinyin: str, context: str) -> str:
    return f"{pinyin}||{context}"


def _json_brace_end(text: str, start: int) -> int:
    """Find the matching closing brace starting from start position."""
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return i
    return -1


class LLMBackend:
    def __init__(self, endpoint: str = None, model: str = None, api_key: str = None,
                 timeout: int = None):
        cfg = DEFAULT_CONFIG["llm"]
        self.endpoint = (endpoint or cfg["endpoint"]).rstrip("/")
        self.model = model or cfg["model"]
        self.api_key = api_key or cfg["api_key"]
        self.timeout = timeout if timeout is not None else cfg.get("timeout", 15)

        # Result cache: (pinyin, context) → (reordered, weight_adj, timestamp)
        self._cache: OrderedDict[str, tuple[list[str], Optional[dict], float]] = (
            OrderedDict()
        )
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

    def _parse_rank_output(
        self, raw: str, candidates: list[str]
    ) -> tuple[list[str], Optional[dict[str, float]]]:
        """Parse LLM rank output into reordered list + optional weight adjustments."""
        weight_adj: dict[str, float] = {}
        text_part = raw

        # Extract JSON adjustment block at end: {"adjust": {...}}
        json_start = raw.rfind('{"adjust"')
        if json_start >= 0:
            brace_end = _json_brace_end(raw, json_start)
            if brace_end > json_start:
                try:
                    json_str = raw[json_start:brace_end + 1]
                    adj_data = json.loads(json_str)
                    raw_adj = adj_data.get("adjust", {})
                    for k, v in raw_adj.items():
                        if isinstance(k, str) and isinstance(v, (int, float)):
                            weight_adj[k] = float(v)
                    text_part = raw[:json_start].strip()
                except (json.JSONDecodeError, ValueError):
                    pass

        # Parse candidate list (comma or Chinese comma separated)
        parsed: list[str] = []
        seen: set[str] = set()
        for token in text_part.replace("，", ",").split(","):
            t = token.strip().strip("\"'「」『』（）()")
            if t and t in candidates and t not in seen:
                parsed.append(t)
                seen.add(t)

        # Append any candidates the LLM missed (preserve original order)
        for c in candidates:
            if c not in seen:
                parsed.append(c)
                seen.add(c)

        adj_out = weight_adj if weight_adj else None
        return parsed, adj_out

    # ── Public API ──

    def convert(self, pinyin: str, context: str = "") -> tuple[Optional[str], float]:
        """LLM picks the single best conversion for a pinyin string."""
        if not self.available:
            return None, 0.0

        user_msg = f"上下文：{context}\n拼音：{pinyin}" if context else f"拼音：{pinyin}"
        return self._call(_CONV_PROMPT, user_msg, max_tokens=64)

    def suggest_sentence(
        self, pinyin: str, context: str = ""
    ) -> tuple[Optional[str], float]:
        """LLM suggests the full sentence conversion (single best guess).

        Unlike convert(), this explicitly asks for sentence-level conversion
        and is intended for longer continuous pinyin input.
        """
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
        baseline: Optional[list[str]] = None,
    ) -> tuple[Optional[list[str]], Optional[dict[str, float]], float]:
        """Re-rank candidates by contextual likelihood.

        Args:
            pinyin: the pinyin input string
            candidates: candidate list to re-rank
            context: preceding text context
            baseline: optional baseline ranking (e.g. Viterbi output) for reference

        Returns:
            (reordered_list | None, weight_adjustments | None, latency_seconds)
        """
        if not self.available or not candidates:
            return None, None, 0.0

        # Check cache
        ck = _cache_key(pinyin, context)
        if ck in self._cache:
            cached_reordered, cached_adj, _ = self._cache.pop(ck)
            # Move to end (most recently used)
            self._cache[ck] = (cached_reordered, cached_adj, time.time())
            return cached_reordered, cached_adj, 0.0

        # Build user message
        candidate_str = "，".join(candidates)
        parts = [f"拼音：{pinyin}", f"候选词：{candidate_str}"]
        if context:
            parts.insert(0, f"上下文：{context}")
        if baseline:
            base_str = "，".join(baseline[:6])
            parts.append(f"基线排序（Viterbi）：{base_str}")
        user_msg = "\n".join(parts)

        raw, elapsed = self._call(RANK_PROMPT, user_msg, max_tokens=512)
        if raw is None:
            return None, None, elapsed

        parsed, adj_out = self._parse_rank_output(raw, candidates)

        # Cache the result
        if parsed:
            self._cache[ck] = (parsed, adj_out, time.time())
            while len(self._cache) > _MAX_CACHE:
                self._cache.popitem(last=False)

        return parsed, adj_out, elapsed

    def clear_cache(self):
        self._cache.clear()

    def stats(self) -> dict:
        return {
            "cached": len(self._cache),
        }
