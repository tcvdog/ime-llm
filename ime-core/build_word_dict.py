#!/usr/bin/env python3
"""
Build a comprehensive Chinese word dictionary using Ollama.

Generates a properly weighted WORD_MAP replacement so that
multi-syllable pinyin produces real words (not character combinations).

Strategy:
  1. Query Ollama for common Chinese 2-character words grouped by pinyin
  2. Build WORD_MAP from the results with frequency weights
  3. Update pinyin_map.py's WORD_MAP at runtime
  4. Save to a JSON file for persistence

Usage:
  python3 build_word_dict.py
  python3 build_word_dict.py --incremental  (append to existing)
"""

import json
import os
import re
import sys
import time

# Add ime-core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
from llm_backend import LLMBackend

OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "ime-core", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "word_dict.json")

INCREMENTAL = "--incremental" in sys.argv
DRY_RUN = "--dry-run" in sys.argv


# ── Ollama helper ───────────────────────────────────────────────

def _make_ollama() -> LLMBackend:
    return LLMBackend(
        endpoint=OLLAMA_ENDPOINT,
        model=OLLAMA_MODEL,
        api_key="",
        timeout=OLLAMA_TIMEOUT,
    )


SYSTEM_PROMPT = (
    "你是一个中文词汇专家。请列举中文里最常见的双字词（2-character words），"
    "按拼音分组输出。每个词附带使用频率分数（1-100，100=最常用）。\n\n"
    "输出格式（每行一个拼音组合）：\n"
    "pinyin: word1(score), word2(score), ...\n\n"
    "例如：\n"
    "shi hou: 时候(95), 侍候(5)\n"
    "xiang tong: 相同(90), 相通(30), 想通(15), 相统(2)\n"
    "zhi dao: 知道(95), 指导(60), 直达(10), 直到(40)\n\n"
    "要求：\n"
    "- 只输出格式化的数据行，不要任何解释\n"
    "- 覆盖所有常见拼音组合\n"
    "- score 反映真实使用频率\n"
    "- 每组至少包含1个词，最多10个\n"
    "- 优先输出真正常用的词\n"
    "- 如果某个拼音没有常见词，跳过"
)

# Common pinyin syllable pairs that form real 2-character words
# This focuses the LLM on productive combinations rather than all 168K pairs
COMMON_SYLLABLE_PAIRS = [
    # Common first syllables
    "shi", "yi", "bu", "ren", "da", "xiao", "shang", "xia", "zhong",
    "guo", "jia", "nian", "yue", "tian", "di", "shan", "shui", "feng",
    "yu", "xue", "dong", "xi", "nan", "bei", "qian", "hou", "zuo",
    "you", "li", "wai", "kai", "guan", "jin", "chu", "hui", "lai",
    "qu", "zou", "pao", "tiao", "kan", "ting", "shuo", "du", "xie",
    "sheng", "huo", "gong", "zuo", "fa", "zhan", "jing", "ji", "jiao",
    "tong", "fang", "xiang", "zhi", "de", "le", "zhe", "he", "you",
    "zai", "er", "ta", "wo", "ni", "zi", "ji", "mei", "hao", "chang",
    "xin", "wen", "hua", "xue", "liao", "qing", "ai", "gan", "jue",
    "jian", "she", "yong", "dian", "ming", "cheng", "fang", "an",
    "zhuang", "liang", "duo", "shao", "kuai", "man", "yuan", "jin",
    "wan", "ti", "wen", "ti", "mu", "bei", "kan", "jian", "dao",
    "jie", "shu", "gu", "shi", "yun", "dong", "yang", "jian", "zhu",
    "dang", "ran", "ru", "guo", "sui", "ran", "dan", "tong", "shi",
    "ci", "wai", "ling", "jie", "tie", "fou", "ze", "zong", "bao",
    "hu", "zhu", "chi", "liang", "ban", "bian", "guan", "xiang",
    "xiang", "zhen", "xi", "wang", "shi", "jian", "li", "bian",
    "hai", "neng", "rang", "gei", "dui", "cong", "ba", "bi", "rang",
    "zhuo", "qiang", "nan", "ruan", "ying", "ku", "tian", "kuai",
    "le", "tong", "ku", "nan", "guan", "yu", "yong", "gan", "qing",
    "zhen", "jia", "xu", "shi", "fei", "chang", "duan", "gao", "ai",
    "pang", "bian", "jian", "dan", "fu", "za", "jian", "yi", "nan",
    "zhu", "ming", "ping", "fan", "zheng", "que", "cuo", "hui", "yi",
    "xiang", "fa", "ban", "fang", "yuan", "ze", "gui", "ding", "tiao",
    "li", "jie", "jue", "ding", "suan", "ji", "hua", "celue", "fang",
    "zhen", "chuang", "xin", "yan", "fa", "tu", "po", "li", "cheng",
    "pu", "tong", "te", "bie", "zhu", "yao", "ci", "ji", "xian",
    "dian", "nao", "wang", "luo", "shu", "ju", "shu", "ma", "zi",
    "lian", "jie", "she", "bei", "an", "quan", "bao", "zhang", "jian",
    "kong", "zhi", "guan", "li", "jing", "ying", "tou", "zi", "xian",
    "mu", "biao", "ji", "chu", "chuan", "bo", "yun", "shu", "cun",
    "chu", "diao", "du", "jie", "pai", "mai", "gou", "wu", "liu",
    "cai", "wu", "zi", "jin", "yin", "hang", "dai", "kuan", "li",
    "xi", "shui", "shou", "ru", "cheng", "ben", "xiao", "yi",
    "tong", "xuan", "chuan", "tou", "ming", "bai", "an", "shi",
]

# Also include all single characters that the user might use
# and generate WORD_MAP entries for common 3-4 syllable phrases
COMMON_PHRASES_3 = [
    "wei shen me", "zen me ban", "zen me yang", "bu ke yi",
    "mei guan xi", "dui bu qi", "mei wen ti", "you yi si",
    "zhi dao le", "ming bai le", "zen me le", "shen me shi",
    "shen me yang", "na li hui", "zen me hui", "mei ban fa",
    "kan jian le", "ting jian le", "gan jue dao", "yi jing le",
    "da jia hao", "xie xie ni", "ni hao ma", "wo bu shi",
    "zhe ge shi", "na ge shi", "ji ge ren", "ji dian le",
    "qu nar le", "zai nar ya", "bu zhi dao", "wo xiang yao",
    "wo yao qu", "ni yao ma", "zhe shi shen me",
    "kan dian shi", "shang wang le", "da dian hua",
    "shou dao le", "fa gei wo", "gei wo kan",
    "yi qi qu", "zai jian le", "ming tian jian",
    "jin tian hao", "zuo tian de", "kai xin ma",
    "e le ma", "chi le ma", "shui jiao le",
    "shang ban qu", "xia ban le", "hui jia le",
    "qu xue xiao", "zai jia li", "deng yi xia",
    "lai kan kan", "qu kan kan", "shuo yi shuo",
    "zuo yi zuo", "xiang yi xiang", "ting yi ting",
    "tian qi hao", "bu cuo o", "fei chang hao",
    "hen zhong yao", "bu yao jin", "mei shi er",
]

COMMON_PHRASES_4 = [
    "yi wu suo you", "xi huan ni", "ai shang ni",
    "qu shi jie", "yi tian dao wan", "man man lai",
]


def _ollama_call(system: str, user: str, max_tokens: int = 4096) -> str | None:
    """Call Ollama with a prompt and return the text response."""
    ollama = _make_ollama()
    raw, elapsed = ollama._call(system, user, max_tokens=max_tokens)
    if raw:
        print(f"  [Ollama] {len(raw)} chars in {elapsed:.1f}s")
    return raw


def parse_word_list(text: str) -> dict[str, list[tuple[str, int]]]:
    """Parse Ollama output into {pinyin: [(word, score), ...]}."""
    result: dict[str, list[tuple[str, int]]] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        # Parse: "shi hou: 时候(95), 侍候(5)"
        match = re.match(r"^([a-z\s]+):\s*(.*)", line)
        if not match:
            continue
        pinyin = " ".join(match.group(1).strip().split())
        words_str = match.group(2).strip()
        entries: list[tuple[str, int]] = []
        for part in words_str.split(","):
            part = part.strip()
            wm = re.match(r"^(.+?)\((\d+)\)", part)
            if wm:
                word = wm.group(1).strip()
                score = min(100, max(1, int(wm.group(2))))
                if word and len(word) >= 2:
                    entries.append((word, score))
        if entries:
            # Sort by score descending
            entries.sort(key=lambda x: -x[1])
            result[pinyin] = entries
    return result


def build_word_dict_via_ollama() -> dict[str, list[tuple[str, int]]]:
    """Step 1: Ask Ollama for common 2-character words by pinyin."""
    print("\n=== 第1步: 查询 Ollama 获取常用双字词 ===")
    print(f"   模型: {OLLAMA_MODEL}, 目标拼音组合: {len(COMMON_SYLLABLE_PAIRS)} 个首音节")

    # Batch: send pairs of syllables for efficiency
    # Generate a prompt with many syllable combinations
    pairs_text = []
    for s1 in COMMON_SYLLABLE_PAIRS:
        pairs_text.append(f"{s1} {s1}")  # same syllable as placeholder

    # Actually let's be smarter - generate combinations with common 2nd syllables
    second_syllables = [
        "shi", "yi", "li", "nian", "tian", "ren", "jian", "hua", "ji",
        "fang", "tong", "xing", "da", "guo", "jia", "chang", "feng",
        "dong", "xi", "nan", "bei", "shui", "shan", "yuan", "jin",
        "xian", "yong", "gong", "zuo", "zhong", "sheng", "ming",
        "xin", "wen", "fa", "zhan", "jiao", "xue", "hui", "qing",
        "ai", "gan", "jue", "dao", "lai", "qu", "shang", "xia",
        "qian", "hou", "zuo", "you", "li", "wai", "kai", "guan",
        "jin", "chu", "ding", "cai", "liao", "yang", "bian", "bao",
        "zhu", "chi", "dai", "zhi", "cheng", "fen", "shou", "lu",
        "an", "quan", "guan", "wang", "luo", "dian", "tong", "bu",
        "men", "ge", "wei", "zai", "you", "guo", "ke", "neng",
        "dui", "cong", "zhe", "de", "le", "ma", "ba", "me",
    ]

    # Generate focused lyricist-style combinations
    combo_list = []
    seen_pairs = set()
    for s1 in COMMON_SYLLABLE_PAIRS[:60]:  # Top 60 first syllables
        for s2 in second_syllables:
            pair = f"{s1} {s2}"
            if pair not in seen_pairs:
                combo_list.append(pair)
                seen_pairs.add(pair)
        if len(combo_list) >= 2000:
            break

    print(f"   生成了 {len(combo_list)} 个拼音组合等待 Ollama 评估...")

    # Split into batches for the LLM
    batch_size = 300
    master_dict: dict[str, list[tuple[str, int]]] = {}

    for i in range(0, len(combo_list), batch_size):
        batch = combo_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(combo_list) + batch_size - 1) // batch_size

        print(f"\n  batch {batch_num}/{total_batches} ({len(batch)} 个组合)...")

        user_msg = "请为以下每个拼音组合列出最常见的双字词（带频率分数）：\n\n"
        for pair in batch:
            user_msg += f"{pair}\n"
        user_msg += (
            "\n输出格式（每行一个）：\n"
            "pinyin: word1(score), word2(score)\n\n"
            "要求：\n"
            "- 只输出有真实词语的组合\n"
            "- 没有常见词的组合直接跳过，不输出\n"
            "- score 范围 1-100\n"
            "- 每组最多5个词"
        )

        raw = _ollama_call(SYSTEM_PROMPT, user_msg, max_tokens=8192)
        if raw:
            parsed = parse_word_list(raw)
            master_dict.update(parsed)
            print(f"   本批获得 {len(parsed)} 个拼音组合")
        else:
            print(f"   [警告] batch {batch_num} 无响应")

        # Small delay to avoid overwhelming Ollama
        if batch_num < total_batches:
            time.sleep(1)

    print(f"\n   Ollama 返回了 {len(master_dict)} 个拼音组合")
    return master_dict


def add_3_4_syllable_phrases() -> dict[str, list[tuple[str, int]]]:
    """Step 2: Get 3-4 syllable common phrases from Ollama."""
    print("\n=== 第2步: 获取3-4音节常用短语 ===")

    all_phrases = list(COMMON_PHRASES_3) + list(COMMON_PHRASES_4)
    result: dict[str, list[tuple[str, int]]] = {}

    batch_size = 50
    for i in range(0, len(all_phrases), batch_size):
        batch = all_phrases[i:i + batch_size]

        user_msg = "请为以下每个拼音组合列出最可能的词语（带频率分数）：\n\n"
        for phrase in batch:
            user_msg += f"{phrase}\n"
        user_msg += (
            "\n输出格式：\n"
            "pinyin: word1(score), word2(score)\n\n"
            "例如：\n"
            "wei shen me: 为什么(99)\n"
            "example, only list real common phrases."
        )

        raw = _ollama_call(SYSTEM_PROMPT, user_msg, max_tokens=4096)
        if raw:
            parsed = parse_word_list(raw)
            result.update(parsed)
        time.sleep(0.5)

    print(f"   获得 {len(result)} 个多音节短语")
    return result


def build_word_dict_file(
    word_dict: dict[str, list[tuple[str, int]]],
    phrase_dict: dict[str, list[tuple[str, int]]],
):
    """Save the word dictionary to JSON."""
    word_dict.update(phrase_dict)

    # Convert to WORD_MAP-compatible format: {pinyin: [word1, word2, ...]}
    word_map: dict[str, list[str]] = {}
    for pinyin, entries in sorted(word_dict.items()):
        words = [w for w, _ in entries]
        if words:
            word_map[pinyin] = words

    data = {
        "word_map": word_map,
        "word_scores": {pinyin: dict(entries) for pinyin, entries in word_dict.items()},
    }

    if not DRY_RUN:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {OUTPUT_PATH}")
        print(f"  WORD_MAP: {len(word_map)} 个拼音组合, {sum(len(v) for v in word_map.values())} 个词语")

    return word_map


def update_pinyin_map(word_map: dict[str, list[str]]):
    """Update pinyin_map.py's WORD_MAP in memory."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
    import importlib
    import pinyin_map as pm

    old_count = len(pm.WORD_MAP)
    pm.WORD_MAP.clear()
    pm.WORD_MAP.update(word_map)

    # Also update the bigram frequency data
    pm._BIGRAM_FREQ.clear()
    for words in pm.WORD_MAP.values():
        for word in words:
            for j in range(len(word) - 1):
                bg = word[j:j+2]
                pm._BIGRAM_FREQ[bg] = pm._BIGRAM_FREQ.get(bg, 0) + 1

    new_count = len(pm.WORD_MAP)
    print(f"  WORD_MAP 已更新: {old_count} → {new_count} 个拼音组合")

    # Reload the module to persist changes if running as script
    importlib.reload(pm)


def main():
    start_time = time.time()
    print("=" * 60)
    print("  中文词组词典构建工具")
    print(f"  Ollama: {OLLAMA_ENDPOINT} / {OLLAMA_MODEL}")
    print("=" * 60)

    word_dict = {}
    phrase_dict = {}

    # Load existing if incremental
    if INCREMENTAL and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            data = json.load(f)
        word_map = data.get("word_map", {})
        word_scores = data.get("word_scores", {})
        word_dict = {k: [(w, word_scores.get(k, {}).get(w, 50)) for w in v]
                     for k, v in word_map.items()}
        print(f"已加载现有词典: {len(word_dict)} 个拼音组合")

    # Step 1: 2-character words
    if not word_dict:
        word_dict = build_word_dict_via_ollama()

    # Step 2: 3-4 syllable phrases
    phrase_dict = add_3_4_syllable_phrases()

    # Step 3: Save to file
    word_map = build_word_dict_file(word_dict, phrase_dict)

    # Step 4: Update pinyin_map
    if not DRY_RUN:
        update_pinyin_map(word_map)

    elapsed = time.time() - start_time
    print(f"\n总计耗时: {elapsed/60:.1f} 分钟")
    print("=" * 60)


if __name__ == "__main__":
    main()
