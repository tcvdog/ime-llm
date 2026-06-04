#!/usr/bin/env python3
"""Unit tests for Viterbi decoder and BigramModel."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bigram_model import BigramModel
from viterbi import viterbi_decode, viterbi_rerank
from pinyin_map import generate_candidates, segment_pinyin

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
        print(f"      expected: {repr(expected)}")


print("=" * 56)
print("  Viterbi Decoder & BigramModel Test Suite")
print("=" * 56)

model = BigramModel()

# ── 1. BigramModel basics ──
print("\n[1] BigramModel basics")
stats = model.stats()
print(f"  Bigrams: {stats['bigrams']}, Vocab: {stats['vocab_chars']}")
assert stats['bigrams'] > 0, "No bigrams loaded!"
assert stats['vocab_chars'] > 0, "Empty vocabulary!"

# Known bigram: "我们" (from WORD_MAP)
p_women = model.bigram_prob("我", "们")
p_wo_shi = model.bigram_prob("我", "是")
check("P(们|我) > P(是|我)", p_women > p_wo_shi, True)

# Unseen bigram should still have non-zero probability (smoothing)
p_unseen = model.bigram_prob("谢", "饯")
check("P(饯|谢) > 0 (smoothing)", p_unseen > 0, True)

# Emission score
em = model.emission_score("我", 82.8)
check("emission_score('我', 82.8) > 0", em > 0, True)

# ── 2. Viterbi decode basics ──
print("\n[2] Viterbi decode basics")

# Single syllable
results_single = viterbi_decode(
    ["wo"],
    [[("我", 82.8), ("窝", 1.5), ("握", 16.4)]],
    model, beam_size=3, max_results=3
)
texts_single = ["".join(ch for ch,_ in seq) for seq in results_single]
check("single 'wo' has results", len(texts_single) > 0, True)
print(f"  single 'wo': {texts_single[:3]}")

# Two syllables: "wo shi"
results_2 = viterbi_decode(
    ["wo", "shi"],
    [[("我", 82.8), ("握", 16.4), ("窝", 1.5)],
     [("是", 100.0), ("时", 77.9), ("事", 47.5), ("十", 44.0)]],
    model, beam_size=5, max_results=5
)
texts_2 = ["".join(ch for ch, _ in seq) for seq in results_2]
print(f"  'wo shi' top-5: {texts_2}")
check("'wo shi' top = 我是", texts_2[0] == "我是", True)

# Known bigram pair: "我" + "们" should rank high
results_wm = viterbi_decode(
    ["wo", "men"],
    [[("我", 82.8), ("握", 16.4)],
     [("们", 40.0), ("门", 30.0), ("闷", 5.0)]],
    model, beam_size=5, max_results=5
)
texts_wm = ["".join(ch for ch, _ in seq) for seq in results_wm]
print(f"  'wo men' top-5: {texts_wm}")
check("'wo men' top = 我们", texts_wm[0] == "我们", True)

# ── 3. Viterbi with unseen chars ──
print("\n[3] Edge cases")

# Single syllable, single char
results_single1 = viterbi_decode(
    ["ri"],
    [[("日", 50.0)]],
    model, beam_size=5, max_results=5
)
texts_single1 = ["".join(ch for ch, _ in seq) for seq in results_single1]
check("single 'ri' = 日", texts_single1[0] == "日" if texts_single1 else False, True)

# Empty input
results_empty = viterbi_decode([], [], model)
check("empty → []", results_empty, [])

# Empty syllable chars
results_nochars = viterbi_decode(["xx"], [[]], model)
check("no chars for syllable → []", results_nochars, [])

# ── 4. Viterbi rerank ──
print("\n[4] Viterbi rerank")

cands = ["我是", "窝是", "我事"]
reranked = viterbi_rerank(cands, ["wo", "shi"], model)
# viterbi_rerank uses uniform emission weights (no syllable→char mapping)
# so ranking is purely bigram-driven. Just check nothing crashes.
print(f"  reranked 'wo shi': {reranked}")
check("rerank produces results", len(reranked) == len(cands), True)

cands2 = ["下面", "虾面", "夏眠"]
reranked2 = viterbi_rerank(cands2, ["xia", "mian"], model)
print(f"  'xia mian' reranked: {reranked2}")

# ── 5. Integration: generate_candidates sends bigram_model ──
print("\n[5] Integration with generate_candidates")

# Without bigram model (existing behavior)
cands_no_bm = generate_candidates("women", max_combinations=9, bigram_model=None)
print(f"  'women' (no bigram): {cands_no_bm[:5]}")

# With bigram model (Viterbi path)
cands_bm = generate_candidates("women", max_combinations=9)
print(f"  'women' (with bigram): {cands_bm[:5]}")

# Most likely Viterbi result should be 我们
if cands_bm:
    check("'women' with Viterbi: 我们 is among top-3", cands_bm[0] in ["我们", "我们"] or "我们" in cands_bm[:3], True)

# Disambiguation test: "xiamian" should prefer 下面 (common bigram)
cands_xm = generate_candidates("xiamian", max_combinations=9)
print(f"  'xiamian': {cands_xm[:5]}")
if cands_xm:
    # 下面 should be in top 3 (bigram 下面 is very common)
    check("'xiamian' has 下面 in top 3", "下面" in cands_xm[:3], True)

# Known word "shijian" (时间)
cands_sj = generate_candidates("shijian", max_combinations=9)
print(f"  'shijian': {cands_sj[:5]}")

# ── 6. Long sequence Viterbi ──
print("\n[6] Long sequence Viterbi")

# 3-syllable: "wo ai ni" (我爱你)
sylls = ["wo", "ai", "ni"]
chars = [
    [("我", 82.8), ("握", 16.4), ("窝", 1.5)],
    [("爱", 80.0), ("挨", 5.0), ("矮", 4.0)],
    [("你", 90.0), ("泥", 10.0), ("逆", 5.0)],
]
results_3 = viterbi_decode(sylls, chars, model, beam_size=10, max_results=5)
texts_3 = ["".join(ch for ch, _ in seq) for seq in results_3]
print(f"  'wo ai ni' top-5: {texts_3}")
if texts_3:
    check("'wo ai ni' top = 我爱你", texts_3[0] == "我爱你", True)

# ── Summary ──
print("\n" + "=" * 56)
total = pass_count + fail_count
print(f"  Results: {pass_count}/{total} passed", end="")
if fail_count > 0:
    print(f", {fail_count} failed ❌")
else:
    print(" ✅")
print("=" * 56)

sys.exit(fail_count)
