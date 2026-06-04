#!/usr/bin/env python3
"""
使用搜狗词库 + pypinyin 构建带权重的拼音词库。

数据来源：
  - word-master/src/main/resources/dic.txt  → 64万词
  - word-master/src/main/resources/bigram.txt  → 词频数据

输出：
  - data/word_dict.json  → 带权重的 WORD_MAP
"""

import json
import os
import sys
from collections import defaultdict

# Add ime-core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
from pinyin_map import SYLLABLE_MAP

DIC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "word-master", "src", "main", "resources", "dic.txt",
)
BIGRAM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "word-master", "src", "main", "resources", "bigram.txt",
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ime-core", "data", "word_dict.json")

print("=" * 60)
print("  基于搜狗词库构建拼音词库")
print("=" * 60)

# Step 1: Load pypinyin
print("\n加载 pypinyin...")
from pypinyin import pinyin as py_pinyin, Style

# Step 2: Load word list
print(f"加载词库: {DIC_PATH}")
all_words = open(DIC_PATH, "r", encoding="utf-8").read().splitlines()
print(f"  总词数: {len(all_words)}")

# Step 3: Load bigram frequency data
print(f"加载词频: {BIGRAM_PATH}")
word_freq: dict[str, int] = defaultdict(int)
for line in open(BIGRAM_PATH, "r", encoding="utf-8"):
    # Format: "word1:word2 count"
    if ":" not in line:
        continue
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    pair = parts[0]
    try:
        count = int(parts[1])
    except ValueError:
        continue
    if ":" in pair:
        w1, w2 = pair.split(":", 1)
        word_freq[w1] += count
        word_freq[w2] += count

print(f"  有频率数据的词: {len(word_freq)}")

# Step 4: Extract 2-character words, get pinyin, validate
print("\n提取双字词并映射拼音...")

word_dict: dict[str, list[tuple[str, int]]] = {}

processed = 0
valid = 0
skipped_no_syllable = 0
skipped_no_match = 0
skipped_not_in_dict = 0
skipped_wrong_pinyin = 0

for word in all_words:
    if len(word) != 2:
        continue

    processed += 1

    # Get pinyin using pypinyin
    try:
        pinyins = py_pinyin(word, style=Style.NORMAL)
        if len(pinyins) != 2:
            skipped_wrong_pinyin += 1
            continue
        syl1 = pinyins[0][0].lower() if pinyins[0] else ""
        syl2 = pinyins[1][0].lower() if pinyins[1] else ""
        if not syl1 or not syl2:
            skipped_wrong_pinyin += 1
            continue
    except Exception:
        skipped_wrong_pinyin += 1
        continue

    # Validate both syllables exist in SYLLABLE_MAP
    if syl1 not in SYLLABLE_MAP:
        skipped_no_syllable += 1
        continue
    if syl2 not in SYLLABLE_MAP:
        skipped_no_syllable += 1
        continue

    # Validate both characters are in their syllable's character list
    ch1, ch2 = word[0], word[1]
    if ch1 not in SYLLABLE_MAP[syl1]:
        skipped_no_match += 1
        continue
    if ch2 not in SYLLABLE_MAP[syl2]:
        skipped_no_match += 1
        continue

    # Check if word appears in frequency data
    freq = word_freq.get(word, 0)
    if freq == 0:
        skipped_not_in_dict += 1
        continue

    pinyin_key = f"{syl1} {syl2}"
    if pinyin_key not in word_dict:
        word_dict[pinyin_key] = []
    word_dict[pinyin_key].append((word, freq))
    valid += 1

print(f"  处理双字词: {processed}")
print(f"  验证通过: {valid}")
print(f"  跳过(音节不在表中): {skipped_no_syllable}")
print(f"  跳过(字符不匹配音节): {skipped_no_match}")
print(f"  跳过(无频率数据): {skipped_not_in_dict}")
print(f"  跳过(拼音异常): {skipped_wrong_pinyin}")

# Step 5: Sort by frequency and build output
print("\n按频率排序...")
word_map: dict[str, list[str]] = {}
for pk, entries in word_dict.items():
    entries.sort(key=lambda x: -x[1])  # high freq first
    word_map[pk] = [w for w, _ in entries]

print(f"  拼音组合: {len(word_map)}")
total_words = sum(len(v) for v in word_map.values())
print(f"  总词数: {total_words}")

# Step 6: Save
print(f"\n保存到: {OUTPUT_PATH}")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"word_map": word_map}, f, ensure_ascii=False, indent=2)

# Step 7: Verify key words
print("\n验证关键词语:")
for test_key in [
    "hai shi", "xiang tong", "zhi dao", "gong zuo", "jian she",
    "fang bian", "suo yi", "dan shi", "yin wei", "ru guo",
]:
    words = word_map.get(test_key, [])
    if words:
        print(f"  {test_key}: {words[:5]}")
    else:
        print(f"  {test_key}: 缺失")

print(f"\n完成! 共 {len(word_map)} 拼音组合, {total_words} 词")
