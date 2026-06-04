#!/usr/bin/env python3
"""Integration test for the IME engine pipeline (LLM-only mode)."""

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
print("  IME Engine Integration Test (LLM-only)")
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

known_cands = generate_candidates("ni hao")
print(f"  'ni hao' candidates ({len(known_cands)}): {known_cands[:5]}")

# ── 3. Engine pipeline ──
print("\n[3] Engine process pipeline")
result = engine.process("ni hao")
print(f"  'ni hao' → {len(result)} candidates")
for text, score, source in result[:5]:
    print(f"    [{source}] {text} (score={score:.1f})")

# ── 4. Context tracking ──
print("\n[4] Context tracking")
engine.reset_context()
engine.process("xiamian")
engine.select("虾面", "xiamian")
print(f"  Selected '虾面' → context: '{engine.context}'")

engine.context = "我饿了想吃一碗"
result_ctx = engine.process("xiamian")
print(f"  Context '我饿了想吃一碗':")
for text, score, source in result_ctx[:5]:
    print(f"    [{source}] {text}")
# In LLM mode, source should be "map" (cache is removed)
assert all(src == "map" for _, _, src in result_ctx[:5]), "Expected all map sources"

# ── 5. LLM backend (if configured) ──
print("\n[5] LLM backend (non-blocking refinement)")
if engine.llm.available:
    print("  LLM configured, testing convert...")
    text, elapsed = engine.llm.convert("xiamian", "我饿了想吃一碗")
    if text:
        print(f"  LLM: '{text}' ({elapsed*1000:.0f}ms)")
    else:
        print(f"  LLM call failed (elapsed: {elapsed*1000:.0f}ms)")
else:
    print("  LLM not configured (set LLM_API_KEY to test)")

# ── 6. Engine has no cache persistence (removed) ──
print("\n[6] Local cache persistence — removed (LLM-only)")
print("  ✓ No cache layer to save/load")
assert not hasattr(engine, 'cache'), "Cache should not exist in LLM-only mode"
assert not hasattr(engine, 'freq_db'), "CharFrequencyDB should not exist in LLM-only mode"
assert not hasattr(engine, 'save_cache'), "save_cache should not exist"

# ── Summary ──
print("\n" + "=" * 54)
total = pass_count + fail_count
print(f"  Results: {pass_count}/{total} passed", end="")
if fail_count > 0:
    print(f", {fail_count} failed ❌")
else:
    print(" ✅")
print("=" * 54)
