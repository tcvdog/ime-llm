#!/usr/bin/env python3
"""
IME 日志分析器 — 从操作日志中提取使用模式。
运行方式: python3 analyze_ime_log.py
输出: 打印分析报告
"""

import os
import re
import json
from collections import Counter, defaultdict
from datetime import datetime

LOG_DIR = os.path.expanduser("~/.cache/ime-llm/log")
LOG_PATH = os.path.join(LOG_DIR, "ime.log")
STATE_PATH = os.path.join(LOG_DIR, "analysis_state.json")


def load_state() -> dict:
    """加载上次分析到的位置。"""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_line": 0, "run_count": 0}


def save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def analyze():
    if not os.path.exists(LOG_PATH):
        print("日志文件不存在")
        return
    
    state = load_state()
    with open(LOG_PATH) as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    if total_lines <= state["last_line"]:
        print(f"无新增日志（上次分析到第 {state['last_line']} 行）")
        return
    
    new_lines = all_lines[state["last_line"]:]
    state["last_line"] = total_lines
    state["run_count"] += 1
    
    # ── 解析日志 ──
    commits = []          # OUT 记录
    keys_total = 0
    out_bksp_count = 0
    correction_patterns = []  # (错误拼音, 最终选中词)
    
    i = 0
    while i < len(new_lines):
        line = new_lines[i].strip()
        if "|OUT|" in line:
            m = re.search(r'py=(\S+)', line)
            m2 = re.search(r'OUT\|(.+?)\|', line)
            py = m.group(1) if m else ""
            word = m2.group(1) if m2 else ""
            commits.append((py, word))
            
            # 看后面几行有没有退格
            for j in range(i+1, min(i+8, len(new_lines))):
                if "Bksp" in new_lines[j]:
                    out_bksp_count += 1
                    # 收集退格后重输的拼音序列
                    retype_keys = []
                    for k in range(j+1, min(j+30, len(new_lines))):
                        kl = new_lines[k].strip()
                        if "OUT|" in kl:
                            # 找到了新的提交
                            m3 = re.search(r'OUT\|(.+?)\|', kl)
                            m4 = re.search(r'py=(\S+)', kl)
                            if m3 and m4:
                                correction_patterns.append((py, m3.group(1), m4.group(1)))
                            break
                        if "Bksp" in kl or "Esc" in kl:
                            break
                    break
        
        if "|KEY|" in line:
            keys_total += 1
        
        i += 1
    
    # ── 找所有退格之前的拼音状态 ──
    bksp_pinyins = []
    for i, line in enumerate(new_lines):
        if "Bksp" in line:
            # 往前找最近的 py= 状态
            for j in range(i-1, max(i-5, -1), -1):
                m = re.search(r'py=(\S+)', new_lines[j])
                if m:
                    bksp_pinyins.append(m.group(1))
                    break
    
    # ── 统计 ──
    bksp_counter = Counter(bksp_pinyins)
    
    # ── 报告 ──
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"IME 日志分析报告 — {now}（第 {state['run_count']} 次）")
    print(f"分析行数: {len(new_lines)} 行（累计 {total_lines} 行）")
    print()
    print(f"按键总次数: {keys_total}")
    print(f"输出汉字数: {len(commits)}")
    print(f"提交后退格次数: {out_bksp_count}")
    print()
    
    if bksp_counter:
        print("=== 退格前的拼音（可能输入有误） ===")
        for py, cnt in bksp_counter.most_common(10):
            print(f"  {py:15s}  {cnt} 次")
        print()
    
    if correction_patterns:
        print("=== 提交→退格→重输 纠错模式 ===")
        for err_py, correct_word, correct_py in correction_patterns[-8:]:
            print(f"  错误: {err_py:15s} → 纠正为: {correct_word:8s} ({correct_py})")
        print()
    
    if commits:
        print("=== 最近输出 ===")
        for py, word in commits[-10:]:
            print(f"  {word:8s} ← {py}")
        print()
    
    print(f"--- 分析状态已保存到 {STATE_PATH} ---")
    
    save_state(state)


if __name__ == "__main__":
    analyze()
