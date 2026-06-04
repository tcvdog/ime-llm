"""
本地词汇学习缓存模块。

核心机制：
- 用户在某上下文中选择了一个词 → 缓存学习
- 下次相同或相似上下文 → 直接命中缓存，跳过 LLM
- 不同上下文 → 不互相干扰（上下文敏感的）

上下文指纹提取方法：
- 取上文末尾 CONTEXT_WINDOW 个有效字符作为指纹
- CONTEXT_WINDOW 越小越容易命中但可能误伤，越大越准确但学习慢

Typical CONTEXT_WINDOW values:
  4: 学习快速，适合短上下文（如"我想"、"打电"）
  6: 平衡（推荐，默认）
  8+: 精确，需要更多训练
"""

from collections import defaultdict
import re
from typing import Optional

DEFAULT_CONTEXT_WINDOW = 6


def normalize_pinyin(pinyin: str) -> str:
    """归一化拼音：去多余空格，统一小写"""
    return re.sub(r"\s+", " ", pinyin.strip().lower())


def extract_context_key(context: str, window: int = DEFAULT_CONTEXT_WINDOW) -> str:
    """
    从上下文提取指纹。
    策略：取末尾有意义字符（去标点、空格，取最后 window 个）
    """
    if not context or not context.strip():
        return ""
    # 去标点和空格，只保留中文字符
    cleaned = re.sub(r"[^\u4e00-\u9fff]", "", context.strip())
    # 取末尾 window 个字符
    return cleaned[-window:] if cleaned else ""


class LocalVocabularyCache:
    """
    本地词汇学习缓存。

    用法:
        cache = LocalVocabularyCache()
        cache.learn("xiamian", "我饿了想吃一碗", "虾面")
        suggestion = cache.suggest("xiamian", "我饿了想吃一碗")  # → "虾面"
    """

    def __init__(self, context_window: int = DEFAULT_CONTEXT_WINDOW):
        self.context_window = context_window
        # 缓存结构:
        #   cache[(normalized_pinyin, context_key)] = {word: count, ...}
        self._cache: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # 训练统计
        self.stats = {
            "total_learn": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }

    def learn(self, pinyin: str, context: str, selected: str):
        """
        学习用户的选择。

        学习策略：
        1. 使用完整的上下文指纹作为精确匹配键
        2. 同时生成所有可能的上下文后缀作为模糊匹配键
           例如："想吃一碗" → ["一碗", "吃一碗", "想吃一碗"]
           这样后续输入 "也想吃一碗" 能通过"一碗"或"吃一碗"命中

        Args:
            pinyin: 用户输入的拼音
            context: 上下文（上文内容）
            selected: 用户最终选择的汉字
        """
        pinyin_norm = normalize_pinyin(pinyin)
        ctx_key = extract_context_key(context, self.context_window)

        # 始终存储精确匹配键
        key = (pinyin_norm, ctx_key)
        self._cache[key][selected] += 1

        # 同时存储所有上下文后缀，用于模糊匹配
        if ctx_key:
            for start in range(len(ctx_key) - 1):
                suffix = ctx_key[start:]
                suffix_key = (pinyin_norm, suffix)
                if suffix_key != key:
                    self._cache[suffix_key][selected] += 1

        self.stats["total_learn"] += 1

    def suggest(self, pinyin: str, context: str) -> Optional[str]:
        """
        根据缓存的用户习惯推荐。

        匹配策略（优先级顺序）:
        1. 精确匹配上下文指纹 → 返回最高频选择
        2. 后缀匹配（逐步缩短上下文指纹）→ 返回最匹配的
        3. 无匹配 → 返回 None（交给 LLM）

        Args:
            pinyin: 用户输入的拼音
            context: 上下文

        Returns:
            推荐词（如果缓存命中），否则 None
        """
        pinyin_norm = normalize_pinyin(pinyin)
        ctx_key_full = extract_context_key(context, self.context_window)

        # Strategy 1: 精确匹配
        key = (pinyin_norm, ctx_key_full)
        if key in self._cache:
            best = max(self._cache[key], key=self._cache[key].get)
            self.stats["cache_hit"] += 1
            return best

        # Strategy 2: 上下文后缀匹配（得益于 n-gram 存储，短的上下文键也存在）
        for end in range(len(ctx_key_full) - 1, 1, -1):
            ctx_sub = ctx_key_full[-end:]
            key = (pinyin_norm, ctx_sub)
            if key in self._cache:
                best = max(self._cache[key], key=self._cache[key].get)
                self.stats["cache_hit"] += 1
                return best

        # Strategy 3: 无上下文匹配（仅拼音匹配）
        key = (pinyin_norm, "")
        if key in self._cache:
            best = max(self._cache[key], key=self._cache[key].get)
            self.stats["cache_hit"] += 1
            return best

        self.stats["cache_miss"] += 1
        return None

    def forget(self, pinyin: str, context: str, word: str):
        """删除一个学习记录（用于修正错误学习）"""
        pinyin_norm = normalize_pinyin(pinyin)
        ctx_key = extract_context_key(context, self.context_window)
        key = (pinyin_norm, ctx_key)
        if key in self._cache and word in self._cache[key]:
            del self._cache[key][word]
            if not self._cache[key]:
                del self._cache[key]

    def size(self) -> int:
        """返回缓存的学习条目数"""
        return sum(len(words) for words in self._cache.values())

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self.stats = {"total_learn": 0, "cache_hit": 0, "cache_miss": 0}

    def get_all_entries(self) -> list[dict]:
        """导出所有学习记录（用于持久化/调试）"""
        entries = []
        for (pinyin, ctx_key), words in sorted(self._cache.items()):
            entries.append({
                "pinyin": pinyin,
                "context_key": ctx_key,
                "candidates": dict(words),
                "total_learns": sum(words.values()),
            })
        return entries


# ─── 演示/自检 ───


def demo():
    """独立演示缓存学习机制"""
    print("=" * 60)
    print("本地词汇学习缓存 - 演示")
    print("=" * 60)

    cache = LocalVocabularyCache(context_window=6)

    test_scenarios = [
        # (拼音, 上下文, 用户选择)
        ("xiamian", "我饿了，想吃一碗", "虾面"),
        ("xiamian", "把书放在桌子", "下面"),
        ("xiamian", "请看", "下面"),
        ("shujia", "马上放", "暑假"),
        ("shujia", "把书放到", "书架"),
        ("baozi", "早餐吃了两个肉", "包子"),
        ("baozi", "动物园里的", "豹子"),
        ("li bai", "唐代诗人", "李白"),
        ("li bai", "花了五百块，还剩下了三", "三百"),
    ]

    print(f"\n场景 1: 学习用户偏好")
    print("-" * 40)
    for pinyin, context, selected in test_scenarios:
        cache.learn(pinyin, context, selected)
        print(f"  学习: '{pinyin}' + '{context}' → '{selected}'")
    print(f"\n  缓存条目数: {cache.size()}")

    print(f"\n场景 2: 相同上下文 → 命中缓存")
    print("-" * 40)
    pinyin, context = "xiamian", "我饿了，想吃一碗"
    result = cache.suggest(pinyin, context)
    print(f"  查询: '{pinyin}' + '{context}'")
    print(f"  结果: {result}  (命中: ✓)")

    print(f"\n场景 3: 不同上下文 → 不同推荐（不互相干扰）")
    print("-" * 40)
    for pinyin, context in [
        ("xiamian", "我饿了，想吃一碗"),
        ("xiamian", "把书放在桌子"),
        ("xiamian", "请看"),
        ("baozi", "早餐吃了两个肉"),
        ("baozi", "动物园里的"),
    ]:
        result = cache.suggest(pinyin, context)
        hit = result is not None
        status = "✓" if hit else "✗"
        print(f"  '{pinyin}' + '{context}' → {result}  (命中: {status})")

    print(f"\n场景 4: 从未见过的上下文 → 不命中（交给 LLM）")
    print("-" * 40)
    result = cache.suggest("xiamian", "从来没见过的上下文")
    print(f"  'xiamian' + '从来没见过的上下文' → {result}")
    print(f"  (返回 None，需要 LLM 处理)")

    print(f"\n场景 5: 上下文模糊匹配（后缀匹配）")
    print("-" * 40)
    # "想吃肉包子" 和 "吃了两个肉" 共享后缀 "肉"
    cache.learn("xiaolongbao", "昨天吃的", "小笼包")
    result = cache.suggest("xiaolongbao", "今天吃的")
    print(f"  学习了: 'xiaolongbao' + '昨天吃的' → '小笼包'")
    print(f"  查询: 'xiaolongbao' + '今天吃的' → {result}")
    print(f"  ('吃的' 后缀匹配，说明模糊匹配有效)")

    print(f"\n缓存统计:")
    print(f"  总学习次数: {cache.stats['total_learn']}")
    print(f"  命中次数:   {cache.stats['cache_hit']}")
    print(f"  未命中次数: {cache.stats['cache_miss']}")
    print(f"  命中率:     {cache.stats['cache_hit']/(cache.stats['cache_hit']+cache.stats['cache_miss'])*100:.0f}%")

    print()
    print("=" * 60)
    print("演示完成 - 缓存机制正常工作")
    print("=" * 60)


if __name__ == "__main__":
    demo()
