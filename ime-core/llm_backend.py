"""
Layer 3: LLM backend for async pinyin→character refinement.

Supports two modes:
  - convert():  single best guess (legacy)
  - rank():     re-rank a list of candidates by contextual likelihood
"""

import json
import time
import urllib.request
import urllib.error
from typing import Optional, Callable

from config import DEFAULT_CONFIG


CONVERT_PROMPT = (
    "你是一个拼音转汉字的转换器。"
    "输入是一个或多个汉语拼音音节（用空格隔开或连续），以及可选的上下文。"
    "只输出转换后的汉字文本，不要任何解释、标点（除非原文需要）、不要多余内容。"
    "如果有多个合理的转换结果，选择最符合上下文语境的。"
)

RANK_PROMPT = (
    "你是一个中文输入法的候选词排序器。"
    "我会给你：\n"
    "1. 当前输入的拼音\n"
    "2. 可选的上下文（前面已输入的文字）\n"
    "3. 候选词列表（用逗号分隔）\n\n"
    "请根据上下文语义、词语搭配概率，把候选词从最可能到最不可能重新排序。\n"
    "只输出重新排序后的候选词，用逗号分隔，不要序号、不要任何解释、不要多余内容。\n"
    "保持候选词原样，不要修改文字。\n"
    "如果某候选词明显不符合上下文，可以排在最后。"
)


class LLMBackend:
    def __init__(self, endpoint: str = None, model: str = None, api_key: str = None):
        cfg = DEFAULT_CONFIG["llm"]
        self.endpoint = (endpoint or cfg["endpoint"]).rstrip("/")
        self.model = model or cfg["model"]
        self.api_key = api_key or cfg["api_key"]
        self.timeout = cfg.get("timeout", 15)
        self._last_result: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

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

    def convert(self, pinyin: str, context: str = "") -> tuple[Optional[str], float]:
        """Legacy: LLM picks the single best conversion."""
        if not self.available:
            return None, 0.0

        if context:
            user_msg = f"上下文：{context}\n拼音：{pinyin}"
        else:
            user_msg = f"拼音：{pinyin}"

        return self._call(CONVERT_PROMPT, user_msg, max_tokens=64)

    def rank(self, pinyin: str, candidates: list[str], context: str = ""
             ) -> tuple[Optional[list[str]], float]:
        """
        Re-rank candidates by contextual likelihood.
        Returns (reordered_list, latency_seconds) or (None, elapsed) on failure.
        """
        if not self.available or not candidates:
            return None, 0.0

        candidate_str = "，".join(candidates)
        if context:
            user_msg = f"上下文：{context}\n拼音：{pinyin}\n候选词：{candidate_str}"
        else:
            user_msg = f"拼音：{pinyin}\n候选词：{candidate_str}"

        raw, elapsed = self._call(RANK_PROMPT, user_msg, max_tokens=256)
        if raw is None:
            return None, elapsed

        # Parse: split by comma/space, keep valid candidates in order
        parsed: list[str] = []
        seen: set[str] = set()
        for token in raw.replace("，", ",").split(","):
            t = token.strip().strip("\"'「」『』")
            if t and t in candidates and t not in seen:
                parsed.append(t)
                seen.add(t)

        # Append any candidates the LLM didn't mention (in original order)
        for c in candidates:
            if c not in seen:
                parsed.append(c)
                seen.add(c)

        return parsed, elapsed
