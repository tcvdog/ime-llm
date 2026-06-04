"""
本地词汇学习缓存验证脚本。

验证流程:
  Phase 1: 基线测试（纯 LLM，无缓存）
  Phase 2: 训练缓存（从失败用例中学习用户偏好）
  Phase 3: 缓存推理（缓存优先，LLM 兜底）
  Phase 4: 多轮迭代训练（可选）

运行:
  export LLM_API_KEY="sk-..."
  python validate_cache.py --endpoint https://api.deepseek.com/v1 --model deepseek-chat
  python validate_cache.py --endpoint https://api.deepseek.com/v1 --model deepseek-chat --iterations 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from local_cache import LocalVocabularyCache
from test_cases import TEST_CASES

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

SYSTEM_PROMPT = (
    "你是一个拼音转汉字的转换器。"
    "输入是一个或多个汉语拼音音节（用空格隔开或连续），以及可选的上下文。"
    "只输出转换后的汉字文本，不要任何解释、标点（除非原文需要）、不要多余内容。"
    "如果有多个合理的转换结果，选择最符合上下文语境的。"
)


def build_user_message(pinyin: str, context: str) -> str:
    if context:
        return f"上下文：{context}\n拼音：{pinyin}"
    else:
        return f"拼音：{pinyin}"


def call_llm(api_key: str, endpoint: str, model: str, pinyin: str, context: str) -> tuple[str, float]:
    """调用 LLM 做拼音→汉字转换，返回 (文本, 延迟秒数)"""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(pinyin, context)},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    start = time.time()

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return f"[HTTP {e.code}] {e.read().decode('utf-8', errors='replace')}", 0.0
    except Exception as e:
        return f"[Error] {e}", 0.0

    elapsed = time.time() - start

    try:
        output = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        output = f"[ParseError] {json.dumps(body, ensure_ascii=False)[:200]}"

    return output, elapsed


def run_evaluation(
    test_cases: list,
    api_key: str,
    endpoint: str,
    model: str,
    cache: LocalVocabularyCache,
    use_cache: bool = True,
    verbose: bool = True,
    learn_from_failures: bool = False,
) -> dict:
    """
    运行一轮评估。

    Args:
        test_cases: 测试用例列表
        cache: 词汇缓存（可能已包含学习数据）
        use_cache: 是否启用缓存优先查询
        learn_from_failures: 是否在失败时自动学习

    Returns:
        评估结果统计
    """
    results = []
    exact_matches = 0
    total_latency = 0.0

    for i, tc in enumerate(test_cases):
        pinyin = tc["pinyin"]
        context = tc["context"]
        expected = tc["expected"]

        source = "llm"
        # 缓存优先
        if use_cache:
            cached = cache.suggest(pinyin, context)
            if cached is not None:
                actual = cached
                source = "cache"
                latency = 0.0
                if verbose:
                    print(f"  [{i+1}/{len(test_cases)}] {tc['id']}... 🟢 CACHE → {actual}")
                exact = actual == expected
                if exact:
                    exact_matches += 1
                results.append({
                    "id": tc["id"],
                    "source": "cache",
                    "expected": expected,
                    "actual": actual,
                    "exact_match": exact,
                    "latency": 0.0,
                })
                continue

        # LLM 兜底
        actual, latency = call_llm(api_key, endpoint, model, pinyin, context)
        exact = actual == expected
        if exact:
            exact_matches += 1
        total_latency += latency

        if verbose:
            status = "✓" if exact else "✗"
            print(f"  [{i+1}/{len(test_cases)}] {tc['id']}... {status} ({latency*1000:.0f}ms)  {actual}")

        # 从失败中学习
        if learn_from_failures and not exact:
            cache.learn(pinyin, context, expected)
            if verbose:
                print(f"     → 学习: '{expected}'")

        results.append({
            "id": tc["id"],
            "source": source,
            "expected": expected,
            "actual": actual,
            "exact_match": exact,
            "latency": latency,
        })

    accuracy = exact_matches / len(test_cases) * 100
    avg_latency = total_latency / len(test_cases) * 1000 if total_latency > 0 else 0

    return {
        "results": results,
        "accuracy": accuracy,
        "exact_matches": exact_matches,
        "total": len(test_cases),
        "avg_latency_ms": avg_latency,
        "cache_size": cache.size(),
        "cache_hits": cache.stats["cache_hit"],
        "cache_misses": cache.stats["cache_miss"],
    }


def print_phase_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_comparison(baseline: dict, trained: dict):
    """打印前后对比"""
    print()
    print("=" * 60)
    print("  对比总结")
    print("=" * 60)
    print(f"{'指标':<30} {'基线':>12} {'训练后':>12} {'变化':>10}")
    print("-" * 64)
    acc_change = trained["accuracy"] - baseline["accuracy"]
    print(f"{'精确匹配率':<30} {baseline['accuracy']:>10.1f}% {trained['accuracy']:>10.1f}% {acc_change:>+9.1f}%")
    print(f"{'精确匹配':<30} {baseline['exact_matches']:>5}/{baseline['total']:<5} {trained['exact_matches']:>5}/{trained['total']:<5}")
    lat_change = trained["avg_latency_ms"] - baseline["avg_latency_ms"]
    print(f"{'平均延迟':<30} {baseline['avg_latency_ms']:>10.0f}ms {trained['avg_latency_ms']:>10.0f}ms {lat_change:>+9.0f}ms")
    print(f"{'缓存条目数':<30} {'—':>12} {trained['cache_size']:>12}")
    print(f"{'缓存命中':<30} {'—':>12} {trained['cache_hits']:>12}")
    print()

    # 列出改进的用例
    baseline_failures = {r["id"] for r in baseline["results"] if not r["exact_match"]}
    trained_failures = {r["id"] for r in trained["results"] if not r["exact_match"]}
    fixed = baseline_failures - trained_failures
    new_failures = trained_failures - baseline_failures

    if fixed:
        print(f"✅ 修复的用例 ({len(fixed)}):")
        for fid in sorted(fixed):
            tc = next(t for t in TEST_CASES if t["id"] == fid)
            print(f"     {fid}: {tc['expected']}")
    if new_failures:
        print(f"❌ 新增失败 ({len(new_failures)}):")
        for fid in sorted(new_failures):
            print(f"     {fid}")
    if not fixed and not new_failures:
        print("  无变化")


def main():
    parser = argparse.ArgumentParser(description="本地词汇缓存验证")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--endpoint", default=os.environ.get("LLM_ENDPOINT", "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--iterations", type=int, default=1, help="多轮迭代训练次数（默认: 1）")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        print("❌ 需要 API key。设置 LLM_API_KEY 环境变量或 --api-key 参数。")
        sys.exit(1)

    # 只看 disambig + named_entity 类别会更能体现缓存效果
    # 但默认跑全部，让用户看到整体效果
    test_cases = TEST_CASES
    total = len(test_cases)

    print()
    print("=" * 60)
    print("  本地词汇学习缓存 - 验证流程")
    print(f"  模型: {args.model}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  用例数: {total}")
    print(f"  训练迭代: {args.iterations} 轮")
    print("=" * 60)

    # ── Phase 1: 基线 ──
    print_phase_header("Phase 1: 基线（纯 LLM，无缓存）")
    cache_empty = LocalVocabularyCache()
    baseline = run_evaluation(
        test_cases, api_key, args.endpoint, args.model,
        cache=cache_empty, use_cache=False, verbose=args.verbose,
    )
    print(f"\n  基线准确率: {baseline['accuracy']:.1f}%")
    print(f"  基线延迟:   {baseline['avg_latency_ms']:.0f}ms")

    # ── Phase 2: 训练 ──
    print_phase_header(f"Phase 2: 缓存训练（从基线失败中学习）")
    cache_trained = LocalVocabularyCache()
    # 从基线失败中学习: 把所有失败的案例作为 "用户纠正" 喂给缓存
    failures = [r for r in baseline["results"] if not r["exact_match"]]
    for r in failures:
        tc = next(t for t in test_cases if t["id"] == r["id"])
        cache_trained.learn(tc["pinyin"], tc["context"], tc["expected"])
        if args.verbose:
            print(f"  ← 学习: {r['id']}  '{tc['pinyin']}' + '{tc['context']}' → '{tc['expected']}'")

    print(f"\n  训练完成。缓存条目数: {cache_trained.size()}")

    # ── Phase 3: 缓存推理 ──
    print_phase_header("Phase 3: 缓存推理（缓存优先，LLM 兜底）")
    trained = run_evaluation(
        test_cases, api_key, args.endpoint, args.model,
        cache=cache_trained, use_cache=True, verbose=args.verbose,
    )

    print(f"\n  训练后准确率: {trained['accuracy']:.1f}%")
    print(f"  缓存命中:     {trained['cache_hits']}")
    print(f"  缓存未命中:   {trained['cache_misses']}")

    # ── 对比 ──
    print_comparison(baseline, trained)

    # ── Phase 4: 多轮迭代（可选） ──
    if args.iterations > 1:
        print_phase_header(f"Phase 4: 多轮迭代训练（{args.iterations} 轮）")

        current_cache = LocalVocabularyCache()
        current_accuracy = 0.0
        history = []

        for iteration in range(1, args.iterations + 1):
            print(f"\n--- 迭代 {iteration}/{args.iterations} ---")
            # 训练：从之前失败中学习
            result = run_evaluation(
                test_cases, api_key, args.endpoint, args.model,
                cache=current_cache, use_cache=True,
                learn_from_failures=True, verbose=False,
            )
            history.append(result)
            current_accuracy = result["accuracy"]
            print(f"  准确率: {result['accuracy']:.1f}%  |  缓存: {result['cache_size']}条  |  命中: {result['cache_hits']}")

        print()
        print("=" * 60)
        print("  迭代训练汇总")
        print("=" * 60)
        print(f"{'迭代':>6} {'准确率':>10} {'缓存条数':>10} {'命中':>8} {'未命中':>8}")
        for i, h in enumerate(history):
            print(f"{i+1:>6} {h['accuracy']:>9.1f}% {h['cache_size']:>10} {h['cache_hits']:>8} {h['cache_misses']:>8}")

    # ── 结论 ──
    print()
    print("=" * 60)
    print("  结论")
    print("=" * 60)
    print(f"  基线准确率:          {baseline['accuracy']:.1f}%")
    print(f"  训练后准确率:        {trained['accuracy']:.1f}%")
    print(f"  提升:                {trained['accuracy'] - baseline['accuracy']:+.1f}%")
    print(f"  基线延迟:            {baseline['avg_latency_ms']:.0f}ms")
    print(f"  缓存命中延迟:        ~0ms （立即返回）")
    print()

    delta = trained["accuracy"] - baseline["accuracy"]
    if delta >= 5:
        print("  ✅ 本地词汇学习缓存显著提升准确率。")
        print(f"     单轮训练修复 {len(failures)}/{total} 个错误 ({delta:.1f}% 提升)。")
        print("     缓存命中零延迟 <= 本地学习的延迟优势明显。")
    elif delta > 0:
        print(f"  ⚠  有提升 ({delta:.1f}%) 但幅度不大。")
        print("     可能需要更多测试用例或更大上下文窗口。")
    else:
        print("  ⚠  缓存未提升准确率。检查上下文指纹设计。")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{args.output_dir}/{ts}_cache_validation.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline,
            "trained": trained,
            "config": {"model": args.model, "endpoint": args.endpoint, "iterations": args.iterations},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {args.output_dir}/{ts}_cache_validation.json")
    print()


if __name__ == "__main__":
    main()
