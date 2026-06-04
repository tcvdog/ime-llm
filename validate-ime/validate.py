"""
拼音→汉字 LLM 转换验证脚本

用法:
  # 设置 API 密钥（任选其一）
  export OPENAI_API_KEY="sk-..."
  # 或 openai compatible endpoint
  export LLM_API_KEY="..."  
  export LLM_ENDPOINT="https://your-endpoint/v1"
  export LLM_MODEL="gpt-4o-mini"

  # 运行
  python validate.py [--model gpt-4o-mini] [--endpoint https://api.openai.com/v1]

  # 指定自定义 endpoint（兼容 openai 格式）
  python validate.py --endpoint https://api.deepseek.com/v1 --model deepseek-chat

输出:
  - 终端打印准确率汇总 + 详细结果表
  - results/{timestamp}_results.json  完整结果
  - results/{timestamp}_summary.txt   文本摘要
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional

# 尝试导入 openai 库，如果没有则用 urllib 兜底
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from test_cases import TEST_CASES

# ─── 配置 ───────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ENDPOINT = "https://api.openai.com/v1"
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


# ─── LLM 调用（openai 库） ──────────────────────────────


def call_llm_openai(api_key: str, endpoint: str, model: str, pinyin: str, context: str) -> tuple[str, float]:
    """返回 (输出文本, 延迟秒数)"""
    client = OpenAI(api_key=api_key, base_url=endpoint)
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(pinyin, context)},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    elapsed = time.time() - start
    output = resp.choices[0].message.content.strip()
    return output, elapsed


# ─── LLM 调用（urllib 兜底，无外部依赖） ─────────────────


def call_llm_urllib(api_key: str, endpoint: str, model: str, pinyin: str, context: str) -> tuple[str, float]:
    """用 urllib 调用 openai-compatible API"""
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


# ─── 评估 ───────────────────────────────────────────────


def compute_char_accuracy(expected: str, actual: str) -> float:
    """简单字符级准确率：正确字符数 / 总字符数"""
    if not expected:
        return 0.0
    correct = sum(1 for a, b in zip(expected, actual) if a == b)
    return correct / len(expected)


def compute_exact_match(expected: str, actual: str) -> bool:
    return expected == actual


# ─── 报告 ───────────────────────────────────────────────


def print_separator(char="=", width=72):
    print(char * width)


def print_result_table(results: list[dict]):
    """打印详细结果表"""
    header = f"{'ID':<20} {'Category':<16} {'准确率':>6} {'延迟(ms)':>10} {'期望→实际':<40}"
    print_separator()
    print("详细结果")
    print_separator()
    print(header)
    print("-" * len(header))

    for r in results:
        acc_pct = f"{r['accuracy']*100:.0f}%"
        lat_ms = f"{r['latency']*1000:.0f}"
        exp_act = f"{r['expected']} → {r['actual']}"
        flag = " ✓" if r["exact_match"] else " ✗"
        print(
            f"{r['id']:<20} {r['category']:<16} {acc_pct:>6} {lat_ms:>8}ms  {exp_act:<40}{flag}"
        )


def print_summary(results: list[dict]):
    """打印汇总统计"""
    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"])
    avg_accuracy = sum(r["accuracy"] for r in results) / total if total else 0
    avg_latency = sum(r["latency"] for r in results) / total if total else 0
    exact_rate = exact_matches / total * 100 if total else 0

    # 按类别统计
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "exact": 0, "acc_sum": 0.0, "lat_sum": 0.0}
        categories[cat]["total"] += 1
        categories[cat]["exact"] += 1 if r["exact_match"] else 0
        categories[cat]["acc_sum"] += r["accuracy"]
        categories[cat]["lat_sum"] += r["latency"]

    print_separator("=")
    print("汇总")
    print_separator("=")
    print(f"总用例:     {total}")
    print(f"精确匹配:   {exact_matches}/{total} ({exact_rate:.1f}%)")
    print(f"平均字符准确率: {avg_accuracy*100:.1f}%")
    print(f"平均延迟:     {avg_latency*1000:.0f}ms")
    print()

    # 按类别
    print_separator("-")
    print(f"{'Category':<20} {'总数':>5} {'精确匹配':>10} {'准确率':>8} {'延迟(ms)':>10}")
    print("-" * 55)
    for cat, stats in sorted(categories.items()):
        acc = stats["acc_sum"] / stats["total"] * 100
        lat = stats["lat_sum"] / stats["total"] * 1000
        exact = f"{stats['exact']}/{stats['total']}"
        print(f"{cat:<20} {stats['total']:>5} {exact:>10} {acc:>7.1f}% {lat:>8.0f}ms")
    print()

    # 列出所有失败
    failures = [r for r in results if not r["exact_match"]]
    if failures:
        print_separator("-")
        print(f"失败用例 ({len(failures)})")
        print_separator("-")
        for r in failures:
            print(f"  [{r['id']}] ({r['category']})")
            print(f"    拼音:   {r['pinyin']}")
            print(f"    上下文: '{r['context']}'")
            print(f"    期望:   {r['expected']}")
            print(f"    实际:   {r['actual']}")
            print(f"    准确率: {r['accuracy']*100:.0f}%")
            print()


def save_results(results: list[dict], dir: str):
    os.makedirs(dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(dir, f"{timestamp}_results.json")
    txt_path = os.path.join(dir, f"{timestamp}_summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 存文本摘要
    lines = []
    lines.append("=" * 60)
    lines.append("拼音→汉字 LLM 转换验证结果")
    lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型: {results[0]['model'] if results else 'N/A'}")
    lines.append(f"Endpoint: {results[0]['endpoint'] if results else 'N/A'}")
    lines.append("=" * 60)
    lines.append("")

    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    avg_acc = sum(r["accuracy"] for r in results) / total if total else 0
    avg_lat = sum(r["latency"] for r in results) / total if total else 0
    lines.append(f"总用例: {total}  精确匹配: {exact}/{total} ({exact/total*100:.1f}%)")
    lines.append(f"平均字符准确率: {avg_acc*100:.1f}%  平均延迟: {avg_lat*1000:.0f}ms")
    lines.append("")

    failures = [r for r in results if not r["exact_match"]]
    if failures:
        lines.append(f"--- 失败用例 ({len(failures)}) ---")
        for r in failures:
            lines.append(f"  [{r['id']}] ({r['category']})")
            lines.append(f"    拼音:   {r['pinyin']}")
            lines.append(f"    上下文: '{r['context']}'")
            lines.append(f"    期望:   {r['expected']}")
            lines.append(f"    实际:   {r['actual']}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  完整结果已保存: {json_path}")
    print(f"  摘要已保存:     {txt_path}")


# ─── 主流程 ─────────────────────────────────────────────


def run_validation(
    api_key: str,
    endpoint: str,
    model: str,
    use_openai_lib: bool,
    test_cases: list,
    verbose: bool = True,
) -> list[dict]:
    results = []

    for i, tc in enumerate(test_cases):
        if verbose:
            print(f"[{i+1}/{len(test_cases)}] {tc['id']}... ", end="", flush=True)

        if use_openai_lib:
            actual, latency = call_llm_openai(api_key, endpoint, model, tc["pinyin"], tc["context"])
        else:
            actual, latency = call_llm_urllib(api_key, endpoint, model, tc["pinyin"], tc["context"])

        accuracy = compute_char_accuracy(tc["expected"], actual)
        exact_match = compute_exact_match(tc["expected"], actual)
        status = "✓" if exact_match else "✗"

        if verbose:
            print(f"{status}  ({latency*1000:.0f}ms)  {actual}")

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "pinyin": tc["pinyin"],
            "context": tc["context"],
            "expected": tc["expected"],
            "actual": actual,
            "accuracy": round(accuracy, 4),
            "exact_match": exact_match,
            "latency": round(latency, 4),
            "model": model,
            "endpoint": endpoint,
        })

    return results


def get_api_key():
    """从环境变量或 ~/.opencode/llm_config.json 读取 API key"""
    # 优先环境变量
    for env_var in ["LLM_API_KEY", "OPENAI_API_KEY"]:
        val = os.environ.get(env_var)
        if val:
            return val, env_var

    # 尝试从 opencode 配置读取
    config_paths = [
        os.path.expanduser("~/.opencode/llm_config.json"),
        os.path.expanduser("~/.opencode.json"),
        os.path.expanduser("~/.opencode.jsonc"),
    ]
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw = f.read()
                # 可能带注释
                if path.endswith(".jsonc"):
                    import re
                    raw = re.sub(r"//.*", "", raw)
                cfg = json.loads(raw)
                key = cfg.get("api_key") or cfg.get("llm", {}).get("api_key") or cfg.get("openai", {}).get("api_key")
                if key:
                    return key, path
            except Exception:
                continue

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="拼音→汉字 LLM 转换验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python validate.py
  python validate.py --model gpt-4o-mini
  python validate.py --endpoint https://api.deepseek.com/v1 --model deepseek-chat
  python validate.py --endpoint http://localhost:1234/v1 --model local-model
  python validate.py --model qwen2.5:7b  # 本地 ollama
        """,
    )
    parser.add_argument("--model", default=None, help=f"模型名称（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--endpoint", default=None, help=f"API endpoint（默认: {DEFAULT_ENDPOINT}）")
    parser.add_argument("--api-key", default=None, help="API 密钥（默认从环境变量读取）")
    parser.add_argument("--output-dir", default="results", help="结果输出目录（默认: results）")
    parser.add_argument("--verbose", action="store_true", default=True, help="显示详细进度")
    parser.add_argument("--dry-run", action="store_true", help="不调 API，只打印请求内容")
    args = parser.parse_args()

    # ── 读取配置 ──
    model = args.model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    endpoint = args.endpoint or os.environ.get("LLM_ENDPOINT", DEFAULT_ENDPOINT)

    if args.dry_run:
        print()
        print("=== Dry Run（仅展示请求内容，无需 API key）===")
        for tc in TEST_CASES:
            print()
            print(f"[{tc['id']}] ({tc['category']})")
            print(f"  System: {SYSTEM_PROMPT}")
            print(f"  User:   {build_user_message(tc['pinyin'], tc['context'])}")
            print(f"  Expect: {tc['expected']}")
        return

    api_key = args.api_key
    key_source = "命令行参数"
    if not api_key:
        api_key, source = get_api_key()
        if api_key:
            key_source = source

    if not api_key:
        print("=" * 60)
        print("未找到 API 密钥！")
        print("=" * 60)
        print()
        print("请通过以下任一方式设置：")
        print()
        print("  1. 环境变量:")
        print(f"     export OPENAI_API_KEY=\"sk-...\"")
        print(f"     export LLM_API_KEY=\"...\"   # 通用")
        print()
        print("  2. 命令行参数:")
        print(f"     python validate.py --api-key sk-...")
        print()
        print("  3. 配置 ~/.opencode/llm_config.json")
        print()
        print("如果要测试本地模型（ollama 等），不需要 API key：")
        print(f"     python validate.py --endpoint http://localhost:11434/v1 --model qwen2.5:7b")
        print(f"     python validate.py --endpoint http://localhost:1234/v1 --model local-model")
        sys.exit(1)

    # ── 打印配置 ──
    print()
    print_separator("=")
    print("  拼音→汉字 LLM 转换验证")
    print_separator("=")
    print(f"  模型:     {model}")
    print(f"  Endpoint: {endpoint}")
    print(f"  API Key:  {api_key[:8]}...{api_key[-4:]} (来源: {key_source})")
    print(f"  用例数:   {len(TEST_CASES)}")
    print(f"  测试范围:")
    from test_cases import count_cases
    for cat, cnt in count_cases().items():
        print(f"    {cat}: {cnt}")
    print_separator("=")
    print()

    # ── dry run ──
    if args.dry_run:
        print("=== Dry Run（仅展示请求内容）===")
        for tc in TEST_CASES:
            print()
            print(f"[{tc['id']}]")
            print(f"  System: {SYSTEM_PROMPT}")
            print(f"  User:   {build_user_message(tc['pinyin'], tc['context'])}")
            print(f"  Expect: {tc['expected']}")
        return

    # ── 执行 ──
    use_openai_lib = HAS_OPENAI
    if not use_openai_lib:
        print("⚠  openai 库未安装，使用 urllib 兜底（功能正常但无流式支持）")
        print("   建议: pip install openai")
        print()

    print(f"开始验证 ({len(TEST_CASES)} 个用例)...")
    print()

    results = run_validation(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        use_openai_lib=use_openai_lib,
        test_cases=TEST_CASES,
        verbose=True,
    )

    # ── 打印结果 ──
    print()
    print_result_table(results)
    print()
    print_summary(results)

    # ── 保存 ──
    save_results(results, args.output_dir)
    print()

    # ── 结论 ──
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    exact_rate = exact / total * 100
    avg_lat = sum(r["latency"] for r in results) / total * 1000

    print_separator("=")
    print("结论")
    print_separator("=")
    print(f"  精确匹配率: {exact_rate:.1f}%")
    print(f"  平均延迟:   {avg_lat:.0f}ms")

    if exact_rate >= 90 and avg_lat < 500:
        print(f"  ✅ LLM 方案可行！准确率和延迟都达标。可以继续推进原型。")
    elif exact_rate >= 80:
        print(f"  ⚠  准确率尚可 ({exact_rate:.1f}%)，但延迟 {'偏高' if avg_lat > 500 else '可接受'}。")
        print(f"     可能需要本地小模型/缓存优化。")
    else:
        print(f"  ❌ 准确率偏低 ({exact_rate:.1f}%)。需要评估 prompt 设计或换模型。")
    print()


if __name__ == "__main__":
    main()
