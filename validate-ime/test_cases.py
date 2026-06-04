"""
拼音→汉字 验证测试用例集

设计思路：
- Category 1 (basic): 无歧义/高概率匹配，测试基础转换能力
- Category 2 (disambig): 核心验证目标——有歧义的拼音组合，测试 LLM 能否利用上下文消歧
- Category 3 (long_sentence): 长句连续拼音（不分词），模拟真实输入场景
- Category 4 (named_entity): 人名、地名、专有名词
- Category 5 (edge): 生僻字、多音字、简写
"""

import os

TEST_CASES = [
    # =====================================================
    # Category 1: 基础无歧义
    # =====================================================
    {
        "id": "basic_01",
        "pinyin": "ni hao",
        "context": "",
        "expected": "你好",
        "category": "basic",
        "note": "基本问候，无歧义",
    },
    {
        "id": "basic_02",
        "pinyin": "xie xie",
        "context": "",
        "expected": "谢谢",
        "category": "basic",
        "note": "感谢，无歧义",
    },
    {
        "id": "basic_03",
        "pinyin": "da jia dou shuo ni hen bang",
        "context": "",
        "expected": "大家都说你很棒",
        "category": "basic",
        "note": "简单完整句子",
    },
    {
        "id": "basic_04",
        "pinyin": "zhe ge dian nao bu cuo",
        "context": "",
        "expected": "这个电脑不错",
        "category": "basic",
        "note": "日常陈述",
    },

    # =====================================================
    # Category 2: 歧义消解（核心验证项）
    # =====================================================
    # --- xiamian: 虾面 vs 下面 vs 下面 ---
    {
        "id": "disambig_xiamian_food",
        "pinyin": "xiamian",
        "context": "我饿了，想吃一碗",
        "expected": "虾面",
        "category": "disambig",
        "note": "食物上下文 → 虾面",
    },
    {
        "id": "disambig_xiamian_position",
        "pinyin": "xiamian",
        "context": "把书放在桌子",
        "expected": "下面",
        "category": "disambig",
        "note": "位置上下文 → 下面",
    },
    {
        "id": "disambig_xiamian_next",
        "pinyin": "xiamian",
        "context": "请看",
        "expected": "下面",
        "category": "disambig",
        "note": "文章指引 → 下面（如「请看下面」）",
    },

    # --- shujia: 暑假 vs 书价 vs 书架 ---
    {
        "id": "disambig_shujia_vacation",
        "pinyin": "shujia",
        "context": "马上放",  # 放假 / 暑假
        "expected": "暑假",
        "category": "disambig",
        "note": "假期上下文 → 暑假",
    },
    {
        "id": "disambig_shujia_price",
        "pinyin": "shujia",
        "context": "这本定价太高，",
        "expected": "书价",
        "category": "disambig",
        "note": "价格上下文 → 书价",
    },
    {
        "id": "disambig_shujia_furniture",
        "pinyin": "shujia",
        "context": "把书放到",
        "expected": "书架",
        "category": "disambig",
        "note": "家具/位置上下文 → 书架",
    },

    # --- baozi: 包子 vs 豹子 vs 孢子 ---
    {
        "id": "disambig_baozi_food",
        "pinyin": "baozi",
        "context": "早餐吃了两个肉",
        "expected": "包子",
        "category": "disambig",
        "note": "食物上下文 → 包子",
    },
    {
        "id": "disambig_baozi_animal",
        "pinyin": "baozi",
        "context": "动物园里的",
        "expected": "豹子",
        "category": "disambig",
        "note": "动物上下文 → 豹子",
    },
    {
        "id": "disambig_baozi_bio",
        "pinyin": "baozi",
        "context": "真菌用繁殖",
        "expected": "孢子",
        "category": "disambig",
        "note": "生物上下文 → 孢子",
    },

    # --- dianhua: 电话 vs 点画 ---
    {
        "id": "disambig_dianhua_comm",
        "pinyin": "dianhua",
        "context": "打",
        "expected": "电话",
        "category": "disambig",
        "note": "通信上下文 → 电话",
    },

    # --- gongshi: 公式 vs 工事 vs 攻势 ---
    {
        "id": "disambig_gongshi_math",
        "pinyin": "gongshi",
        "context": "这个数学",
        "expected": "公式",
        "category": "disambig",
        "note": "数学上下文 → 公式",
    },
    {
        "id": "disambig_gongshi_military",
        "pinyin": "gongshi",
        "context": "敌人发起了猛烈的",
        "expected": "攻势",
        "category": "disambig",
        "note": "军事上下文 → 攻势",
    },

    # --- youxian: 优先 vs 有限 vs 有线 ---
    {
        "id": "disambig_youxian_order",
        "pinyin": "youxian",
        "context": "这个任务",
        "expected": "优先",
        "category": "disambig",
        "note": "排序上下文 → 优先",
    },
    {
        "id": "disambig_youxian_limit",
        "pinyin": "youxian",
        "context": "资源",
        "expected": "有限",
        "category": "disambig",
        "note": "数量上下文 → 有限",
    },

    # --- qizhong: 期中 vs 期终 vs 其中 ---
    {
        "id": "disambig_qizhong_exam",
        "pinyin": "qizhong",
        "context": "下周",
        "expected": "期中",
        "category": "disambig",
        "note": "考试上下文 → 期中（考试）",
    },
    {
        "id": "disambig_qizhong_among",
        "pinyin": "qizhong",
        "context": "这个班级共有30人，",
        "expected": "其中",
        "category": "disambig",
        "note": "范围上下文 → 其中",
    },

    # =====================================================
    # Category 3: 长句连续拼音（不分词）
    # =====================================================
    {
        "id": "long_sentence_01",
        "pinyin": "jintianqizhenhao",
        "context": "",
        "expected": "今天天气真好",
        "category": "long_sentence",
        "note": "7音节连续，无空格",
    },
    {
        "id": "long_sentence_02",
        "pinyin": "womenyinggaizenmeban",
        "context": "",
        "expected": "我们应该怎么办",
        "category": "long_sentence",
        "note": "7音节连续疑问句",
    },
    {
        "id": "long_sentence_03",
        "pinyin": "zuijinmang she nme ne",
        "context": "",
        "expected": "最近忙什么呢",
        "category": "long_sentence",
        "note": "混合空格（模拟用户输入不统一）",
    },
    {
        "id": "long_sentence_04",
        "pinyin": "zhe ge wentiwoxuyaokaolvyixia",
        "context": "",
        "expected": "这个问题我需要考虑一下",
        "category": "long_sentence",
        "note": "部分分词部分连续",
    },
    {
        "id": "long_sentence_05",
        "pinyin": "wobuzhidaogairenhezuo",
        "context": "",
        "expected": "我不知道该怎么办",
        "category": "long_sentence",
        "note": "口语句式连续拼音",
    },

    # =====================================================
    # Category 4: 人名/地名/专有名词
    # =====================================================
    {
        "id": "name_01",
        "pinyin": "li bai",
        "context": "唐代诗人",
        "expected": "李白",
        "category": "named_entity",
        "note": "历史人名",
    },
    {
        "id": "name_02",
        "pinyin": "li bai",
        "context": "今天我花了五百块，还剩下了三",
        "expected": "三百",
        "category": "named_entity",
        "note": "数字上下文 → 里百× → 三百",
    },
    {
        "id": "name_03",
        "pinyin": "beijing shi wo guo shou du",
        "context": "",
        "expected": "北京是我国首都",
        "category": "named_entity",
        "note": "地名",
    },
    {
        "id": "name_04",
        "pinyin": "ma yun",
        "context": "阿里巴巴创始人",
        "expected": "马云",
        "category": "named_entity",
        "note": "当代知名人物",
    },

    # =====================================================
    # Category 5: 边缘场景
    # =====================================================
    {
        "id": "edge_01",
        "pinyin": "le",
        "context": "我吃",
        "expected": "了",
        "category": "edge",
        "note": "单音节助词 \u2014\u2014 了 vs 乐",
    },
    {
        "id": "edge_02",
        "pinyin": "le",
        "context": "这件事让我很快",
        "expected": "乐",
        "category": "edge",
        "note": "单音节 → 快乐",
    },
    {
        "id": "edge_03",
        "pinyin": "xing",
        "context": "你贵",
        "expected": "姓",
        "category": "edge",
        "note": "多音字/礼貌用语 → 姓",
    },
    {
        "id": "edge_04",
        "pinyin": "xing",
        "context": "我今天很高",
        "expected": "兴",
        "category": "edge",
        "note": "多音字 → 高兴",
    },
    {
        "id": "edge_05",
        "pinyin": "zhi neng",
        "context": "人工智能",
        "expected": "智能",
        "category": "edge",
        "note": "常见技术词汇",
    },
    {
        "id": "edge_06",
        "pinyin": "zhi neng",
        "context": "我只",
        "expected": "只能",
        "category": "edge",
        "note": "副词：只能 vs 智能",
    },
    {
        "id": "edge_07",
        "pinyin": "xian zai ji dian",
        "context": "",
        "expected": "现在几点",
        "category": "edge",
        "note": "疑问句式",
    },
    {
        "id": "edge_08",
        "pinyin": "ni zhi dao ma",
        "context": "",
        "expected": "你知道吗",
        "category": "edge",
        "note": "口语疑问句",
    },
]


def count_cases() -> dict:
    counts = {}
    for tc in TEST_CASES:
        cat = tc["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def get_cases_by_category(category: str) -> list:
    return [tc for tc in TEST_CASES if tc["category"] == category]


# ─── JSON import/export ───


DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")


def export_json(filepath: str = DEFAULT_JSON_PATH):
    """将测试用例导出为 JSON 文件"""
    import json
    data = {
        "meta": {
            "total": len(TEST_CASES),
            "categories": count_cases(),
            "description": "拼音→汉字 LLM 转换验证测试用例集",
        },
        "test_cases": TEST_CASES,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已导出 {len(TEST_CASES)} 个用例到: {filepath}")


def load_from_json(filepath: str = DEFAULT_JSON_PATH) -> list[dict]:
    """Load test cases from JSON file; fall back to built-in on missing file"""
    import json
    if not os.path.exists(filepath):
        print(f"⚠  未找到 {filepath}，使用代码内测试用例")
        return TEST_CASES
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data if isinstance(data, list) else data.get("test_cases", TEST_CASES)
    print(f"从 {filepath} 加载了 {len(cases)} 个用例")
    return cases


if __name__ == "__main__":
    counts = count_cases()
    total = sum(counts.values())
    print(f"总测试用例数: {total}")
    print(f"分类: {counts}")
    for cat, cnt in counts.items():
        print(f"  {cat}: {cnt} 用例")
    print()
    export_json()
