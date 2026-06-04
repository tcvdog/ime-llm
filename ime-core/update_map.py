#!/usr/bin/env python3
"""
IME Map Updater — uses Ollama to improve pinyin map character rankings.

Processes:
  1. Single syllables: re-rank all characters by usage frequency via Ollama
  2. WORD_MAP multi-syllable entries: validate & re-rank
  3. Common 2/3/4-syllable combinations: generate n-gram frequency data
  4. Updates pinyin_map_ext.json with Ollama-informed weights
  5. Generates updated bigram/unigram frequency data

Usage:
  python3 update_map.py                     # process ALL (may take 5-15 min)
  python3 update_map.py --syllables-only    # only single syllables (faster)
  python3 update_map.py --words-only        # only WORD_MAP entries
  python3 update_map.py --dry-run           # print what would be done

Config:
  Set OLLAMA_ENDPOINT / OLLAMA_MODEL env vars, or edit defaults below.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add ime-core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
from llm_backend import LLMBackend

# ── Config ──────────────────────────────────────────────────────
OLLAMA_ENDPOINT = os.environ.get(
    "OLLAMA_ENDPOINT", "http://localhost:11434/v1",
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "10"))
BATCH_SIZE = 8  # concurrent Ollama requests
DRY_RUN = "--dry-run" in sys.argv

DATA_DIR = os.path.join(os.path.dirname(__file__), "ime-core", "data")
PINYIN_MAP_PATH = os.path.join(DATA_DIR, "pinyin_map_ext.json")
BIGRAM_PATH = os.path.join(DATA_DIR, "bigram_counts.json")
UNIGRAM_PATH = os.path.join(DATA_DIR, "unigram_counts.json")


# ── Ollama helper ───────────────────────────────────────────────

def _make_ollama() -> LLMBackend:
    return LLMBackend(
        endpoint=OLLAMA_ENDPOINT,
        model=OLLAMA_MODEL,
        api_key="",
        timeout=OLLAMA_TIMEOUT,
    )


RANK_SYLLABLE_PROMPT = (
    "You are a Chinese language frequency expert. I will give you a pinyin syllable "
    "and a list of Chinese characters that map to it. "
    "Rank the characters by how frequently they are used in modern written Chinese, "
    "from MOST common to LEAST common.\n\n"
    "Output format: comma-separated list of characters ONLY. No explanations, no numbers."
)


def _rank_syllable(ollama: LLMBackend, syllable: str, chars: list[str]) -> list[str] | None:
    """Ask Ollama to rank characters for a single pinyin syllable."""
    user_msg = f"拼音：{syllable}\n候选字：{'，'.join(chars)}"
    raw, elapsed = ollama._call(RANK_SYLLABLE_PROMPT, user_msg, max_tokens=512)
    if not raw:
        return None
    # Parse comma-separated characters
    parsed = []
    seen = set()
    for tok in raw.replace("，", ",").split(","):
        t = tok.strip()
        # Single Chinese character expected
        if t and len(t) == 1 and '\u4e00' <= t <= '\u9fff' and t not in seen:
            if t in chars:
                parsed.append(t)
                seen.add(t)
    # Append any characters Ollama missed (preserve original order)
    for ch in chars:
        if ch not in seen:
            parsed.append(ch)
            seen.add(ch)
    return parsed


RANK_PHRASE_PROMPT = (
    "You are a Chinese IME expert. I will give you a pinyin sequence and "
    "a list of possible Chinese word/phrase candidates. "
    "Rank the candidates by how likely/natural they are in real Chinese text, "
    "from MOST likely to LEAST likely.\n\n"
    "Output format: comma-separated list of candidates ONLY. No explanations."
)


def _rank_phrase(ollama: LLMBackend, pinyin: str, candidates: list[str]) -> list[str] | None:
    """Ask Ollama to rank multi-character candidates for a pinyin sequence."""
    if not candidates:
        return None
    user_msg = f"拼音：{pinyin}\n候选词：{'，'.join(candidates)}"
    raw, elapsed = ollama._call(RANK_PHRASE_PROMPT, user_msg, max_tokens=256)
    if not raw:
        return None
    parsed = []
    seen = set()
    for tok in raw.replace("，", ",").split(","):
        t = tok.strip().strip("\"'「」『』（）()")
        if t and t in candidates and t not in seen:
            parsed.append(t)
            seen.add(t)
    for c in candidates:
        if c not in seen:
            parsed.append(c)
            seen.add(c)
    return parsed


# ── Update single-syllable rankings ────────────────────────────

def process_syllables(data: dict) -> dict:
    """Re-rank all single-syllable character lists using Ollama."""
    total = len(data)
    print(f"\n{'='*60}")
    print(f"处理单音节声母 ({total} 个, batch={BATCH_SIZE})")
    print(f"{'='*60}")

    updated = {}
    processed = 0
    batch_num = 0

    syllables = sorted(data.keys())
    ollama = _make_ollama()

    if not ollama.available:
        print("  [跳过] Ollama 不可用，检查 OLLAMA_ENDPOINT")
        return data

    for i in range(0, len(syllables), BATCH_SIZE):
        batch = syllables[i:i + BATCH_SIZE]
        batch_num += 1

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
            futures = {}
            for syl in batch:
                chars = data[syl]
                if len(chars) <= 2:
                    # 1-2 chars, not worth asking Ollama
                    updated[syl] = chars
                    processed += 1
                    continue
                futures[pool.submit(_rank_syllable, ollama, syl, chars)] = syl

            for future in as_completed(futures):
                syl = futures[future]
                try:
                    reordered = future.result()
                    if reordered:
                        updated[syl] = reordered
                    else:
                        updated[syl] = data[syl]
                except Exception as e:
                    print(f"  [错误] {syl}: {e}")
                    updated[syl] = data[syl]
                processed += 1

        pct = processed / total * 100
        done_chars = sum(len(updated.get(s, data.get(s, []))) for s in batch)
        print(f"  batch {batch_num}: {processed}/{total} ({pct:.0f}%) — 示例: {batch[0] if batch else '?'}", end="")
        # Show first 5 chars comparison for first syllable in batch
        if batch:
            orig_top = ''.join(data[batch[0]][:5])
            new_top = ''.join(updated.get(batch[0], data[batch[0]])[:5])
            if orig_top != new_top:
                print(f"  [{batch[0]}] {orig_top} → {new_top}")
            else:
                print()

    # Handle syllables that weren't processed (<=2 chars)
    for syl in syllables:
        if syl not in updated:
            updated[syl] = data[syl]

    changed = sum(1 for s in syllables if updated.get(s) != data.get(s))
    print(f"\n单音节处理完成: {processed}/{total}, 排序变化: {changed} 个音节")
    return updated


# ── Update WORD_MAP multi-syllable entries ─────────────────────

def process_word_map(word_map_path: str, data: dict) -> dict:
    """Re-rank WORD_MAP multi-syllable entries using Ollama."""
    # Import WORD_MAP from pinyin_map.py
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pinyin_map",
        os.path.join(os.path.dirname(__file__), "ime-core", "pinyin_map.py"),
    )
    pinyin_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pinyin_map)
    WORD_MAP = pinyin_map.WORD_MAP

    entries = [(k, v) for k, v in WORD_MAP.items() if len(v) > 1]
    total = len(entries)
    print(f"\n{'='*60}")
    print(f"处理多音节词汇 (WORD_MAP, {total} 个有歧义项)")
    print(f"{'='*60}")

    ollama = _make_ollama()
    if not ollama.available:
        print("  [跳过] Ollama 不可用")
        return data

    updated_word_map = dict(WORD_MAP)
    processed = 0

    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
            futures = {}
            for pinyin_key, candidates in batch:
                futures[pool.submit(_rank_phrase, ollama, pinyin_key, candidates)] = pinyin_key

            for future in as_completed(futures):
                pk = futures[future]
                try:
                    reordered = future.result()
                    if reordered:
                        updated_word_map[pk] = reordered
                    else:
                        updated_word_map[pk] = [w for w, _ in batch if w[0] == pk[0]] if False else dict(WORD_MAP).get(pk, [])
                        updated_word_map[pk] = dict(WORD_MAP).get(pk, [])
                except Exception:
                    pass
                processed += 1

        pct = processed / total * 100
        if batch:
            pk = batch[0][0]
            orig = dict(WORD_MAP).get(pk, [])
            new = updated_word_map.get(pk, [])
            changed = " ✓" if orig != new else ""
            print(f"  batch {i//BATCH_SIZE + 1}: {processed}/{total} ({pct:.0f}%) — {pk}: {orig[:3]} → {new[:3]}{changed}")

    changed_count = sum(1 for k in updated_word_map if updated_word_map[k] != WORD_MAP.get(k, []))
    print(f"\n多音节处理完成: {processed}/{total}, 排序变化: {changed_count} 个")

    # Update pinyin_map module's WORD_MAP
    pinyin_map.WORD_MAP.clear()
    pinyin_map.WORD_MAP.update(updated_word_map)

    return data


# ── Generate frequency data ────────────────────────────────────

def generate_freq_data(data: dict, word_map_path: str = None):
    """Generate unigram and bigram frequency data from updated map."""
    print(f"\n{'='*60}")
    print(f"生成频率数据")
    print(f"{'='*60}")

    # Unigram: position-based frequency for each syllable's characters
    unigram = {}
    for syl, chars in data.items():
        total = len(chars)
        for pos, ch in enumerate(chars):
            # Exponential decay: first = 100, last ~= 0.1
            weight = 100.0 * (0.5 ** (pos / (total * 0.3))) if total > 0 else 1.0
            unigram[ch] = max(unigram.get(ch, 0), round(weight, 2))

    print(f"  单字频率: {len(unigram)} 个字符")

    # Bigram: from WORD_MAP common pairs
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pinyin_map",
        os.path.join(os.path.dirname(__file__), "ime-core", "pinyin_map.py"),
    )
    pinyin_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pinyin_map)

    bigram = {}
    for words in pinyin_map.WORD_MAP.values():
        for word in words:
            for j in range(len(word) - 1):
                bg = word[j:j+2]
                bigram[bg] = bigram.get(bg, 0) + 1

    print(f"  双字组合 (bigram): {len(bigram)} 个")

    if not DRY_RUN:
        with open(UNIGRAM_PATH, "w", encoding="utf-8") as f:
            json.dump(unigram, f, ensure_ascii=False, indent=2)
        print(f"  -> 已保存: {UNIGRAM_PATH}")

        with open(BIGRAM_PATH, "w", encoding="utf-8") as f:
            json.dump(bigram, f, ensure_ascii=False, indent=2)
        print(f"  -> 已保存: {BIGRAM_PATH}")

    return unigram, bigram


# ── Save updated pinyin map ────────────────────────────────────

def save_pinyin_map(data: dict):
    """Save updated pinyin_map_ext.json."""
    if DRY_RUN:
        print(f"\n[DRY RUN] 跳过保存 pinyin_map_ext.json")
        return

    with open(PINYIN_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存: {PINYIN_MAP_PATH} ({len(data)} 个音节)")


# ── Main ───────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Load current data
    if not os.path.exists(PINYIN_MAP_PATH):
        print(f"错误: 找不到 {PINYIN_MAP_PATH}")
        sys.exit(1)

    with open(PINYIN_MAP_PATH, "r") as f:
        data = json.load(f)

    print(f"IME 地图更新器")
    print(f"  Ollama: {OLLAMA_ENDPOINT} / {OLLAMA_MODEL}")
    print(f"  数据: {PINYIN_MAP_PATH} ({len(data)} 音节, {sum(len(v) for v in data.values())} 字符)")
    if DRY_RUN:
        print(f"  模式: DRY RUN (不会保存任何文件)")

    only_syllables = "--syllables-only" in sys.argv
    only_words = "--words-only" in sys.argv

    # Step 1: Single syllable re-ranking
    if not only_words:
        data = process_syllables(data)

    # Step 2: WORD_MAP multi-syllable re-ranking
    if not only_syllables:
        # Also check if we have WORD_MAP access
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
        try:
            data = process_word_map(None, data)
        except Exception as e:
            print(f"  [警告] 多音节处理失败: {e}")

    # Step 3: Generate frequency data
    if not only_words and not only_syllables:
        generate_freq_data(data)

    # Step 4: Save updated map
    if not only_words:
        save_pinyin_map(data)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"完成! 耗时 {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
