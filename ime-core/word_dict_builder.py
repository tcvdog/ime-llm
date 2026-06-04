#!/usr/bin/env python3
"""
词库后台构建器 — 持续在后台使用 Ollama 更新字库。

处理 1~6 音节的拼音组合，生成带权重的词库：
  1字词: 对411个音节，重新排序列出字符频率
  2字词: 覆盖常用双字词组合
  3字词: 常用三字短语
  4-6字词: 成语、常用短语、习惯用语

运行后持续在后台工作，每完成一批自动保存。
可以随时停止（Ctrl+C），下次运行自动从断点继续。

Usage:
  python3 word_dict_builder.py               # 正常运行
  python3 word_dict_builder.py --quick        # 快速模式(只处理高频组合)
  python3 word_dict_builder.py --status       # 查看进度
"""

import json
import os
import re
import sys
import time
import signal

# Add ime-core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
from llm_backend import LLMBackend

OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "ime-core", "data")
STATE_PATH = os.path.join(DATA_DIR, "word_dict_build_state.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "word_dict.json")

running = True


def sigint_handler(sig, frame):
    global running
    print("\n[中断] 保存进度后退出...")
    running = False


signal.signal(signal.SIGINT, sigint_handler)


# ── Progress state ──────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"phase": 0, "completed": [], "word_dict": {}, "phrases": {}}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def save_word_dict(state: dict):
    """Save the full word dictionary to JSON."""
    word_map = {}
    word_scores = {}

    for d in [state["word_dict"], state["phrases"]]:
        for pinyin, entries in d.items():
            if pinyin not in word_map:
                word_map[pinyin] = [w for w, _ in entries]
                word_scores[pinyin] = {w: s for w, s in entries}

    data = {"word_map": word_map, "word_scores": word_scores}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in word_map.values())
    print(f"[保存] WORD_MAP: {len(word_map)} 拼音, {total} 词")


# ── Ollama helper ───────────────────────────────────────────────

def _call_ollama(system: str, user: str, max_tokens: int = 4096) -> str | None:
    ollama = LLMBackend(
        endpoint=OLLAMA_ENDPOINT,
        model=OLLAMA_MODEL,
        api_key="",
        timeout=OLLAMA_TIMEOUT,
    )
    raw, elapsed = ollama._call(system, user, max_tokens=max_tokens)
    if raw:
        return raw
    return None


def parse_word_entries(text: str) -> dict[str, list[tuple[str, int]]]:
    """Parse: xiang tong: 相同(95), 相通(30)"""
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        m = re.match(r"^([a-z\s]+):\s*(.*)", line)
        if not m:
            continue
        pinyin = " ".join(m.group(1).strip().split())
        entries = []
        for part in m.group(2).split(","):
            part = part.strip()
            wm = re.match(r"^(.+?)\((\d+)\)", part)
            if wm:
                word = wm.group(1).strip()
                score = min(100, max(1, int(wm.group(2))))
                if word:
                    entries.append((word, score))
        if entries:
            entries.sort(key=lambda x: -x[1])
            result[pinyin] = entries
    return result


# ── Phase 1: Single syllable character ranking ──────────────────

PHASE1_SYLLABLES = [
    "a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao",
    "bei", "ben", "beng", "bi", "bian", "biao", "bie", "bin", "bing",
    "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce", "cen", "ceng",
    "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng",
    "chi", "chong", "chou", "chu", "chua", "chuai", "chuan", "chuang",
    "chui", "chun", "chuo", "ci", "cong", "cou", "cu", "cuan", "cui",
    "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "dei",
    "deng", "di", "dia", "dian", "diao", "die", "ding", "diu", "dong",
    "dou", "du", "duan", "dui", "dun", "duo", "e", "en", "er", "fa",
    "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu", "ga",
    "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong",
    "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo",
    "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng",
    "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun",
    "huo", "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing",
    "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai", "kan",
    "kang", "kao", "ke", "ken", "keng", "kong", "kou", "ku", "kua",
    "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la", "lai", "lan",
    "lang", "lao", "le", "lei", "leng", "li", "lia", "lian", "liang",
    "liao", "lie", "lin", "ling", "liu", "long", "lou", "lu", "lv",
    "luan", "lue", "lun", "luo", "ma", "mai", "man", "mang", "mao",
    "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min",
    "ming", "miu", "mo", "mou", "mu", "na", "nai", "nan", "nang",
    "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao",
    "nie", "nin", "ning", "niu", "nong", "nou", "nu", "nv", "nuan",
    "nue", "nuo", "o", "ou", "pa", "pai", "pan", "pang", "pao",
    "pei", "pen", "peng", "pi", "pian", "piao", "pie", "pin", "ping",
    "po", "pou", "pu", "qi", "qia", "qian", "qiang", "qiao", "qie",
    "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun", "ran",
    "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou", "ru",
    "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang", "sao",
    "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao",
    "she", "shei", "shen", "sheng", "shi", "shou", "shu", "shua",
    "shuai", "shuan", "shuang", "shui", "shun", "shuo", "si", "song",
    "sou", "su", "suan", "sui", "sun", "suo", "ta", "tai", "tan",
    "tang", "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting",
    "tong", "tou", "tu", "tuan", "tui", "tun", "tuo", "wa", "wai",
    "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia",
    "xian", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu",
    "xu", "xuan", "xue", "xun", "ya", "yan", "yang", "yao", "ye",
    "yi", "yin", "ying", "yo", "yong", "you", "yu", "yuan", "yue",
    "yun", "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen",
    "zeng", "zha", "zhai", "zhan", "zhang", "zhao", "zhe", "zhen",
    "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan",
    "zhuang", "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu",
    "zuan", "zui", "zun", "zuo",
]


def phase1_single_chars(state: dict) -> dict:
    """Rank characters for each single syllable using Ollama."""
    completed = set(state.get("completed", []))
    word_dict = state.get("word_dict", {})

    print("\n=== 阶段1: 单音节字符排序 ===")
    total = len(PHASE1_SYLLABLES)
    done = len([s for s in PHASE1_SYLLABLES if s in completed])

    for i, syl in enumerate(PHASE1_SYLLABLES):
        if not running:
            break
        if syl in completed:
            continue

        sys_prompt = "你是一个中文频率专家。只输出逗号分隔的汉字列表。"
        user_msg = (
            f"拼音 \"{syl}\" 对应哪些汉字？请按使用频率从高到低排列。"
            f"只输出汉字，用逗号分隔，不要任何说明。"
        )
        raw = _call_ollama(sys_prompt, user_msg, max_tokens=512)
        if raw:
            chars = []
            for tok in raw.replace("，", ",").split(","):
                ch = tok.strip()
                if ch and len(ch) == 1 and '\u4e00' <= ch <= '\u9fff':
                    chars.append(ch)
            if chars:
                entries = [(ch, max(1, 100 - j * 3)) for j, ch in enumerate(chars[:30])]
                word_dict[syl] = entries
                completed.add(syl)

        done += 1
        if i % 20 == 0 or done == total:
            pct = done / total * 100
            print(f"  单音节: {done}/{total} ({pct:.0f}%)")
            state["completed"] = list(completed)
            state["word_dict"] = word_dict
            save_state(state)
            save_word_dict(state)

    print(f"  单音节完成: {len(word_dict)} 个")
    return word_dict


# ── Phase 2: 2-character common words ──────────────────────────

PHASE2_2CHAR_SYLLABLES = [
    "shi", "yi", "bu", "ren", "da", "xiao", "shang", "xia", "zhong",
    "guo", "jia", "nian", "yue", "tian", "di", "shan", "shui", "feng",
    "yu", "dong", "xi", "nan", "bei", "qian", "hou", "zuo", "you",
    "li", "wai", "kai", "guan", "jin", "chu", "hui", "lai", "qu",
    "zou", "kan", "ting", "shuo", "du", "xie", "sheng", "huo",
    "gong", "zuo", "fa", "zhan", "jing", "ji", "jiao", "tong",
    "fang", "xiang", "zhi", "de", "he", "you", "zai", "er",
    "ta", "wo", "ni", "zi", "mei", "hao", "chang", "xin", "wen",
    "hua", "xue", "qing", "ai", "gan", "jue", "jian", "she",
    "yong", "dian", "ming", "cheng", "an", "zhuang", "liang",
    "duo", "shao", "kuai", "man", "yuan", "jin", "wan", "ti",
    "wen", "mu", "bei", "jie", "shu", "gu", "yun", "dong",
    "yang", "zhu", "dang", "ran", "ru", "sui", "dan", "ci",
    "wai", "ling", "fou", "ze", "zong", "bao", "hu", "zhu",
    "chi", "ban", "bian", "guan", "xiang", "zhen", "xi", "wang",
    "hai", "neng", "rang", "gei", "dui", "cong", "ba", "bi",
    "qiang", "nan", "ruan", "ying", "ku", "tian", "le", "tong",
    "guan", "yu", "qing", "zhen", "jia", "xu", "fei", "chang",
    "duan", "gao", "ai", "bian", "fu", "yi", "ping", "fan",
    "zheng", "que", "cuo", "hui", "xiang", "fa", "ban",
    "yuan", "ze", "gui", "ding", "tiao", "jue", "suan", "ji",
    "hua", "chuang", "xin", "yan", "tu", "po", "pu", "te",
    "bie", "zhu", "yao", "ci", "xian", "dian", "nao", "wang",
    "luo", "shu", "ju", "ma", "zi", "lian", "she", "bei",
    "an", "quan", "bao", "jian", "kong", "zhi", "guan", "li",
    "jing", "ying", "tou", "xian", "biao", "chu", "chuan",
    "bo", "yun", "shu", "cun", "diao", "pai", "mai", "gou",
    "wu", "liu", "cai", "jin", "yin", "hang", "dai", "kuan",
    "xi", "shou", "ru", "cheng", "xiao", "xuan", "tou",
    "ming", "bai",
]


def phase2_2char_words(state: dict) -> dict:
    """Generate 2-character word dictionary using Ollama."""
    completed = set(state.get("completed_2char", []))
    word_dict = state.get("word_dict", {})

    print("\n=== 阶段2: 双字词 ===")
    print(f"  源音节: {len(PHASE2_2CHAR_SYLLABLES)} 个")

    # Build comprehensive combination list
    all_pairs = []
    for s1 in PHASE2_2CHAR_SYLLABLES:
        for s2 in PHASE2_2CHAR_SYLLABLES:
            all_pairs.append(f"{s1} {s2}")

    total_pairs = len(all_pairs)
    done = len([p for p in all_pairs if p in completed])
    batch_size = 200

    for i in range(0, len(all_pairs), batch_size):
        if not running:
            break

        batch = all_pairs[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_pairs + batch_size - 1) // batch_size

        # Skip if all in this batch are done
        if all(p in completed for p in batch):
            continue

        sys_prompt = (
            "你是中文词汇专家。对每个拼音组合，列出最常见的2字词及频率分数(1-100)。"
            "如果某个组合没有常见词，跳过不输出。只输出格式化数据。"
        )
        user_msg = "列出以下拼音组合对应的最常见的2字词：\n\n"
        for pair in batch:
            user_msg += f"{pair}\n"
        user_msg += (
            "\n输出格式（每行一个）：\n"
            "xiang tong: 相同(95), 相通(30)\n"
            "要求：只输出有真实词语的组合，没有就跳过，分数1-100"
        )

        raw = _call_ollama(sys_prompt, user_msg, max_tokens=8192)
        if raw:
            parsed = parse_word_entries(raw)
            for pk, entries in parsed.items():
                word_dict[pk] = entries
                completed.add(pk)

        # Mark all pairs in batch as completed (even if no result)
        for p in batch:
            completed.add(p)

        pct = len([p for p in all_pairs if p in completed]) / total_pairs * 100
        print(f"  批 {batch_num}/{total_batches}: {pct:.0f}% ({len(word_dict)} 拼音组合)")

        state["completed_2char"] = list(completed)
        state["word_dict"] = word_dict
        save_state(state)
        save_word_dict(state)

        if batch_num < total_batches:
            time.sleep(1)

    print(f"  双字词完成: {len(word_dict)} 个拼音组合")
    return word_dict


# ── Phase 3: 3-6 character phrases ─────────────────────────────

PHASE3_PHRASES = [
    # 3-syllable common phrases
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
    "you shi hou", "na yi tian", "zhe yi ci",
    "xia yi ge", "shang yi ci", "di yi ci",
    "zui hou yi", "kai shi le", "jie shu le",
    "zhun bei hao", "kai xin de", "nan guo de",
    "gao xing de", "sheng qi de", "zhu yao shi",
    "qi shi shi", "dan shi wo", "suo yi wo",
    "er qie ye", "huo zhe shi", "hai you yi",
    "ru guo ni", "yin wei wo", "sui ran wo",
    # 4-syllable
    "yi wu suo you", "xi huan ni", "ai shang ni",
    "qu shi jie", "yi tian dao wan", "man man lai",
    "hao jiu bu jian", "ni zen me yang", "wo hen hao",
    "ti qian shuo hao", "fang xin hao le", "bie dan xin",
    "zhu ni hao yun", "gong xi fa cai", "xin nian kuai le",
    "sheng ri kuai le", "jie ri kuai le", "wan shi ru yi",
    "yi lu ping an", "shen ti jian kang", "xue xi jin bu",
    "gong zuo shun li", "ming tian jian",
    # 5-syllable
    "zhe shi zen me le", "ni zai gan shen me",
    "wo bu zhi dao a", "zhe ge hen zhong yao",
    "wo ting bu dong", "ni neng bang wo ma",
    "wo hui qu de", "qing gei wo kan kan",
    "bu yong dan xin", "wo yi jing zhi dao le",
    "hen gao xing ren shi ni", "xie xie ni de bang zhu",
    # 6-syllable
    "wo yi zhi zai deng ni", "ni zai na li ya",
    "zhe ge dong xi hen hao", "wo men yi qi qu ba",
    "tian qi zhen hao a", "ni jin tian zen me yang",
    "wo ming tian zhao ni", "rang wo men yi qi lai",
]


def phase3_phrases(state: dict) -> dict:
    """Generate 3-6 syllable phrase dictionary."""
    completed = set(state.get("completed_phrases", []))
    phrases = state.get("phrases", {})

    print("\n=== 阶段3: 多音节短语 (3-6字) ===")
    print(f"  短语总数: {len(PHASE3_PHRASES)}")

    for i, pinyin in enumerate(PHASE3_PHRASES):
        if not running:
            break
        if pinyin in completed:
            continue

        sys_prompt = "你是一个中文短语专家。只输出格式化数据。"
        user_msg = (
            f"拼音 \"{pinyin}\" 对应的中文短语是什么？"
            f"如有多个可能，按使用频率从高到低排列，附带频率分数(1-100)。\n"
            f"输出格式：{pinyin}: 短语1(分数), 短语2(分数)"
        )
        raw = _call_ollama(sys_prompt, user_msg, max_tokens=512)
        if raw:
            parsed = parse_word_entries(raw)
            if parsed:
                phrases.update(parsed)
                completed.add(pinyin)
            else:
                # No parseable result, mark as done anyway
                completed.add(pinyin)
        else:
            completed.add(pinyin)

        if i % 20 == 0 or i == len(PHASE3_PHRASES) - 1:
            pct = (i + 1) / len(PHASE3_PHRASES) * 100
            print(f"  短语: {i+1}/{len(PHASE3_PHRASES)} ({pct:.0f}%) - {len(phrases)} 个已解析")
            state["completed_phrases"] = list(completed)
            state["phrases"] = phrases
            save_state(state)
            save_word_dict(state)

        time.sleep(0.3)

    print(f"  短语完成: {len(phrases)} 个")
    return phrases


# ── Update pinyin_map.py WORD_MAP ──────────────────────────────

def update_runtime_word_map(state: dict):
    """Update the running pinyin_map.py WORD_MAP with our dictionary."""
    try:
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ime-core"))
        pinyin_map = importlib.import_module("pinyin_map")

        # Build combined word map
        word_map = {}
        for d in [state.get("word_dict", {}), state.get("phrases", {})]:
            for pk, entries in d.items():
                words = [w for w, _ in entries]
                if words:
                    word_map[pk] = words

        old_count = len(pinyin_map.WORD_MAP)
        pinyin_map.WORD_MAP.clear()
        pinyin_map.WORD_MAP.update(word_map)

        # Update bigram freq
        pinyin_map._BIGRAM_FREQ.clear()
        for words in pinyin_map.WORD_MAP.values():
            for word in words:
                for j in range(len(word) - 1):
                    bg = word[j:j+2]
                    pinyin_map._BIGRAM_FREQ[bg] = pinyin_map._BIGRAM_FREQ.get(bg, 0) + 1

        new_count = len(pinyin_map.WORD_MAP)
        print(f"[更新] WORD_MAP: {old_count} → {new_count} 拼音组合")

        # Also save to the deployed system dir
        DEPLOY_DIR = "/usr/share/ibus-ime-llm"
        if os.path.exists(DEPLOY_DIR):
            deploy_pm = os.path.join(DEPLOY_DIR, "pinyin_map.py")
            data_path = os.path.join(DEPLOY_DIR, "data", "word_dict.json")
            if os.path.exists(OUTPUT_PATH):
                import shutil
                os.makedirs(os.path.join(DEPLOY_DIR, "data"), exist_ok=True)
                shutil.copy2(OUTPUT_PATH, data_path)
                print(f"[部署] 词库已复制到 {data_path}")
    except Exception as e:
        print(f"[警告] WORD_MAP 更新失败: {e}")


# ── Status ──────────────────────────────────────────────────────

def print_status():
    if not os.path.exists(STATE_PATH):
        print("状态: 尚未开始")
        return
    state = load_state()
    wd = state.get("word_dict", {})
    ph = state.get("phrases", {})
    c1 = len(state.get("completed", []))
    c2 = len(state.get("completed_2char", []))
    c3 = len(state.get("completed_phrases", []))
    total_2char = len(PHASE2_2CHAR_SYLLABLES) ** 2
    print(f"阶段1 单音节: {c1}/{len(PHASE1_SYLLABLES)}")
    print(f"阶段2 双字词:  {len(wd)} 拼音组合 ({c2}/{total_2char} 已扫描)")
    print(f"阶段3 多音节:  {len(ph)} 短语 ({c3}/{len(PHASE3_PHRASES)} 已处理)")
    total_words = sum(len(v) for v in wd.values()) + sum(len(v) for v in ph.values())
    print(f"词库总计: {len(wd)+len(ph)} 拼音组合, {total_words} 个词")


# ── Main ────────────────────────────────────────────────────────

def main():
    if "--status" in sys.argv:
        print_status()
        return

    quick = "--quick" in sys.argv
    print("=" * 60)
    print("  词库后台构建器")
    print(f"  Ollama: {OLLAMA_ENDPOINT} / {OLLAMA_MODEL}")
    print(f"  {'快速模式' if quick else '完整模式'}")
    print("=" * 60)

    state = load_state()
    phases = state.get("completed_phases", [])
    first_run = not state.get("word_dict") and not state.get("phrases")

    if first_run:
        print("首次运行，从零开始构建词库...")
    else:
        wd = len(state.get("word_dict", {}))
        ph = len(state.get("phrases", {}))
        print(f"恢复断点: word_dict={wd}, phrases={ph}")

    # Phase 1: Single characters
    if 1 not in phases and running and not quick:
        word_dict = phase1_single_chars(state)
        state["word_dict"] = word_dict
        phases.append(1)
        state["completed_phases"] = phases
        save_state(state)

    # Phase 2: 2-character words
    if 2 not in phases and running:
        word_dict = phase2_2char_words(state)
        state["word_dict"] = word_dict
        phases.append(2)
        state["completed_phases"] = phases
        save_state(state)

    # Phase 3: Multi-syllable phrases
    if 3 not in phases and running and not quick:
        phrases = phase3_phrases(state)
        state["phrases"] = phrases
        phases.append(3)
        state["completed_phases"] = phases
        save_state(state)

    # Update runtime
    if running:
        update_runtime_word_map(state)
        print("\n✅ 构建完成!")
    else:
        print("\n⏸ 已中断，下次继续")

    print_status()


if __name__ == "__main__":
    main()
