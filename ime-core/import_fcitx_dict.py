#!/usr/bin/env python3
"""
从系统已安装的 fcitx5/libpinyin 词库导出拼音词库。

来源: /usr/share/libime/sc.dict (209,269 词)
格式: word pinyin score
输出: data/word_dict.json
"""

import json
import os
import re
import sys
import subprocess
from collections import defaultdict

IME_DIR = os.path.join(os.path.dirname(__file__), "ime-core")
DATA_DIR = os.path.join(IME_DIR, "data")
SC_DICT = "/usr/share/libime/sc.dict"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "word_dict.json")

print("=" * 60)
print("  从 fcitx5/libpinyin 导入词库")
print("=" * 60)

# Step 1: Use libime_pinyindict to dump the dictionary to text
print(f"\n导出 sc.dict...")
txt_path = "/tmp/sc_dict.txt"
result = subprocess.run(
    ["libime_pinyindict", "-d", SC_DICT, txt_path],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"导出失败: {result.stderr}")
    sys.exit(1)

# Step 2: Parse the text format
print(f"解析 {txt_path}...")
word_dict: dict[str, list[tuple[str, float]]] = defaultdict(list)
total = 0
valid = 0
skipped = 0

# Load SYLLABLE_MAP for validation
sys.path.insert(0, IME_DIR)
from pinyin_map import SYLLABLE_MAP

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1

        # Parse: "word pinyin score"
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            skipped += 1
            continue
        word_pinyin = parts[0].strip()
        try:
            score = float(parts[1])
        except ValueError:
            skipped += 1
            continue

        # Split word and pinyin: "大 da" or "大巴 da'ba"
        wp_parts = word_pinyin.split(" ", 1)
        if len(wp_parts) != 2:
            skipped += 1
            continue

        word = wp_parts[0].strip()
        raw_pinyin = wp_parts[1].strip()

        # Convert pinyin format: "da'ba" → "da ba"
        pinyin = raw_pinyin.replace("'", " ")
        syllables = pinyin.split()

        if len(word) != len(syllables) or len(word) < 2:
            skipped += 1
            continue

        # Validate both characters exist in SYLLABLE_MAP
        ok = True
        for ch, syl in zip(word, syllables):
            if syl in SYLLABLE_MAP and ch not in SYLLABLE_MAP[syl]:
                ok = False
                break
        if not ok:
            skipped += 1
            continue

        # Normalize score: negative means very rare, min 0.1
        if score <= 0:
            score = 0.1
        elif score > 100:
            score = 100.0

        word_dict[pinyin].append((word, score))
        valid += 1

print(f"  总行: {total}")
print(f"  有效: {valid}")
print(f"  跳过: {skipped}")
print(f"  拼音组合: {len(word_dict)}")

# Step 3: Sort by score within each pinyin group
print("\n排序...")
word_map: dict[str, list[str]] = {}
for pk, entries in word_dict.items():
    entries.sort(key=lambda x: -x[1])
    word_map[pk] = [w for w, _ in entries]

print(f"  WORD_MAP: {len(word_map)} 组合")
print(f"  总词数: {sum(len(v) for v in word_map.values())}")

# Step 4: Save
print(f"\n保存到: {OUTPUT_PATH}")
os.makedirs(DATA_DIR, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"word_map": word_map}, f, ensure_ascii=False, indent=2)

# Step 5: Verify
print("\n验证:")
from engine import Engine
from config import load_config
from pinyin_map import WORD_MAP

e = Engine(config=load_config())
for test in ['haishi', 'xiangtong', 'zhi dao', 'gong zuo', 'jian she',
             'fang bian', 'yin wei', 'suo yi', 'dan shi', 'huo zhe']:
    words = WORD_MAP.get(test, [])
    if words:
        print(f"  {test}: {words[:5]}")
    else:
        r = e.process(test.replace(' ', ''))
        print(f"  {test}: {r[0][0] if r else '?'} (from generate_candidates)")

print(f"\n完成! {len(word_map)} 拼音组合, {sum(len(v) for v in word_map.values())} 词")
