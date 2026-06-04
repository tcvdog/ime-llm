#!/usr/bin/env python3
"""Integration test for the IME engine pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Engine
from pinyin_map import segment_pinyin, generate_candidates

engine = Engine()
pass_count = 0
fail_count = 0


def check(name, actual, expected):
    global pass_count, fail_count
    ok = actual == expected
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: {repr(actual)}")
    if ok:
        pass_count += 1
    else:
        fail_count += 1
        print(f"     期望: {repr(expected)}")


print("=" * 54)
print("  IME Engine Integration Test")
print("=" * 54)

# ── 1. Pinyin segmentation ──
print("\n[1] Pinyin segmentation")
check("space-separated", segment_pinyin("ni hao"), ["ni", "hao"])
check("continuous short", segment_pinyin("xiamian"), ["xia", "mian"])
check("continuous long", segment_pinyin("jintianqizhenhao"), ["jin", "tian", "qi", "zhen", "hao"])
check("continuous medium", segment_pinyin("womenyinggaizenmeban"), ["wo", "men", "ying", "gai", "zen", "me", "ban"])
check("empty string", segment_pinyin(""), [])
check("single syllable", segment_pinyin("ni"), ["ni"])

# ── 2. Candidate generation ──
print("\n[2] Candidate generation")
single_cands = generate_candidates("ni")
print(f"  single 'ni' candidates ({len(single_cands)}): {single_cands[:5]}")
assert len(single_cands) > 0, "Empty candidates for 'ni'"

multi_cands = generate_candidates("xiamian")
print(f"  multi 'xiamian' candidates ({len(multi_cands)}): {multi_cands[:5]}")
assert len(multi_cands) > 0, "Empty candidates for 'xiamian'"

# Check specific known mappings
known_cands = generate_candidates("ni hao")
print(f"  'ni hao' candidates ({len(known_cands)}): {known_cands[:5]}")

# ── 3. Engine pipeline ──
print("\n[3] Engine process pipeline")
result = engine.process("ni hao")
print(f"  'ni hao' → {len(result)} candidates")
for text, score, source in result[:5]:
    print(f"    [{source}] {text} (score={score:.1f})")

engine.reset_context()

# With context - should show basic candidates
result_context = engine.process("xiamian")
print(f"\n  'xiamian' → {len(result_context)} candidates")
for text, score, source in result_context[:5]:
    print(f"    [{source}] {text} (score={score:.1f})")

# ── 4. Cache learning ──
print("\n[4] Cache learning")
engine.reset_context()
engine.process("xiamian")
engine.select("虾面", "xiamian")
print(f"  Learned: 'xiamian' → '虾面' (context: '{engine.context}')")

engine.reset_context()
engine.context = "我饿了想吃一碗"
result_cached = engine.process("xiamian")
print(f"  Same context candidates:")
source_shown = set()
for text, score, source in result_cached[:5]:
    source_shown.add(source)
    print(f"    [{source}] {text} (score={score:.1f})")

# Expect cache to boost "虾面" (even though context differs slightly)
cached_items = [(t, s) for t, s, src in result_cached if src == "cache"]
if cached_items:
    print(f"  ✓ Cache boosted: {cached_items}")
else:
    print(f"  Note: Cache did not boost (context context differs)")

# ── 5. Different context = different result ──
print("\n[5] Context sensitivity")
engine.reset_context()
engine.context = "把书放在桌子"
result_diff = engine.process("xiamian")
print(f"  Context '把书放在桌子':")
first_source = result_diff[0][2] if result_diff else "none"
print(f"  Top: [{first_source}] {result_diff[0][0] if result_diff else 'N/A'}")

# ── 6. LLM backend (if configured) ──
print("\n[6] LLM backend (non-blocking refinement)")
if engine.llm.available:
    print("  LLM configured, testing convert...")
    text, elapsed = engine.llm.convert("xiamian", "我饿了想吃一碗")
    if text:
        print(f"  LLM: '{text}' ({elapsed*1000:.0f}ms)")
    else:
        print(f"  LLM call failed (elapsed: {elapsed*1000:.0f}ms)")
else:
    print("  LLM not configured (set LLM_API_KEY to test)")

# ── 7. Cache persistence ──
print("\n[7] Cache persistence")
engine.save_cache("/tmp/test_ime_cache.json")
cache_size_before = engine.cache.size()

engine2 = Engine()
engine2.load_cache("/tmp/test_ime_cache.json")
cache_size_after = engine2.cache.size()
check("save/load roundtrip", cache_size_after, cache_size_before)

# Cleanup
import os as _os
if _os.path.exists("/tmp/test_ime_cache.json"):
    _os.remove("/tmp/test_ime_cache.json")

# ── Summary ──
print("\n" + "=" * 54)
total = pass_count + fail_count
print(f"  Results: {pass_count}/{total} passed", end="")
if fail_count > 0:
    print(f", {fail_count} failed ❌")
else:
    print(" ✅")
print("=" * 54)
