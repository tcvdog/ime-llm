# IME 引擎 — 锚定摘要

## Goal
用大规模的频率加权字符库 + 实时 LLM 权重调整来丰富输入法

## 限制与偏好
- LLM 是多种算法之一 —— 即使没有远程 API key，权重调整和重排序也必须能运行
- 传统输入法基础功能必须可用（翻页、Enter=原始拼音、中文标点）
- 实现必须通过集成测试

## 进度

### 已完成
- 修复 `gui.py` `_add_llm_candidate()` bug —— 现在正确合并完整重排序列表，而不是用列表插入破坏数据
- 在 `pinyin_map.py` 中添加 `local_rank()` 双字频率重排序器（基于 WORD_MAP 的 529 条双字频率图）
- 将本地重排序接入 `engine.py` Step 3，作为 LLM 不可用时的回退
- 创建 `freq_db.py` —— `CharFrequencyDB` 类：
  - 从 `data/pinyin_map_ext.json` 加载 3512 个字符、407 个拼音音节
  - 基于排序位置计算频率得分（指数衰减，最高 ≈ 100，最低 ≈ 0.1）
  - 运行时权重调整（LLM 学习）+ 通过 JSON 文件持久化
  - 提供 `get_syllable_chars()`、`adjust_weight()`、`apply_llm_adjustments()`、save/load 方法
- 创建 `data/gen_freq_data.py` —— 从紧凑内联数据构建 `pinyin_map_ext.json` 的生成器脚本
- 更新 `pinyin_map.py` —— 添加 `get_weighted_characters()`（支持传入 FreqDB）；`generate_candidates()` 现在接受可选的 `freq_db` 参数，并使用频率加权排序
- 重写 `engine.py` —— 集成 `CharFrequencyDB` 进行评分；`_text_score()` 对字符权重取平均；LLM 精炼时调用 `apply_llm_adjustments()`；提供显式的 `save_freq()`/`load_freq()`；`select()` 对选中字符加分
- 更新 `llm_backend.py` —— `rank()` 现在返回三元组 `(list | None, dict | None, float)`，包含可选的 `{char: delta}` 权重调整；解析 LLM 响应中的 `{"adjust": {...}}` JSON 块
- 更新 `config.py` —— 添加 `freq_db.save_path` 配置项
- 修复 `freq_db.py` 中重复的 `stats()` 方法（使用了错误的属性名 `_syllable_chars`）
- 修复 `freq_db.py` `adjust_weight()` 的 ±50 钳位 bug —— 将钳位移至 `get_weight()`，防止 LLM 大幅调整后 `select()` 的 +0.05 增益被静默吞掉
- 端到端验证（11 项检查全部通过）：频率加权排序 → LLM 调整 → 字符重排序 → 持久化 → 重新加载 → select() 累积奖励 → engine.process() 反映调整后权重
- 集成测试：**7/7 通过**

### 已部署
- **IBus 引擎**以 root 身份部署至 `/usr/share/ibus-ime-llm/`，包含新增的 `freq_db.py` + `data/pinyin_map_ext.json`
- MD5 校验和全部匹配源码，IBus 守护进程已重启
- `ibus list-engine` 确认 `ime-llm` 已注册
- 运行时验证：`CharFrequencyDB` 加载 3433 字、407 音节，正确

### 已验证兼容
- `ibus_main.py` —— 正确使用 `Engine()` 和 `engine.llm_refine()`
- `main.py` —— 使用 `Engine()` 构造函数，兼容
- `gui.py` —— 使用 `Engine()` 构造函数，兼容

## 关键决策
- 字符频率数据存储为 `data/pinyin_map_ext.json`（44KB）的 JSON 格式，而非内联 Python，以避免文件大小问题
- 频率得分根据排序位置以指数方式计算——不硬编码——因此任何按音节的排序都能产生有效分数
- LLM 权重调整与基础频率分离，基础数据保持干净；调整结果通过单独 JSON 文件持久化
- `select()` 选中字符加 +0.05，在基础频率之上逐步实现长期学习
- LLM 的 rank 响应解析 JSON `{"adjust": {...}}` 块获取调整值——如果 LLM 不返回则优雅降级

## 下一步
1. ~~`llm_backend.py` 添加权重调整模式~~ ✅ 已完成
2. ~~`config.py` 更新 freq_db.save_path~~ ✅ 已完成
3. ~~`freq_db.py` 修复 adjust_weight() 钳位 bug~~ ✅ 已完成
4. ~~运行完整集成测试~~ ✅ 7/7 通过
5. ~~端到端权重调整验证~~ ✅ 11/11 通过
6. （可选）用实际 LLM 端点测试（设置 `LLM_API_KEY`）验证权重调整端到端流程

## 关键上下文
- `engine.py` 在 `__init__()` 中创建 `CharFrequencyDB`——启动时加载 `data/pinyin_map_ext.json`；如果 JSON 缺失，回退到 `SYLLABLE_MAP`
- `generate_candidates()` 传入 `freq_db` 参数——控制每个音节的候选字符及其顺序
- 没有 LLM 时，引擎使用 `local_rank()` 进行重排序 + 基础频率权重进行字符排序——两者完全离线可用
- 集成测试 `test_engine.py` 尚未测试 `CharFrequencyDB` —— 后续可添加

## 相关文件
- `/home/tcvdog/桌面/算法集合/ime-core/freq_db.py` —— `CharFrequencyDB` 类，从 JSON 预计算的模块级 `_CHAR_BASE` 和 `_SYLLABLE_IDX`
- `/home/tcvdog/桌面/算法集合/ime-core/data/pinyin_map_ext.json` —— 3512 个字符的频率排序映射（生成文件）
- `/home/tcvdog/桌面/算法集合/ime-core/data/gen_freq_data.py` —— 数据文件的生成器
- `/home/tcvdog/桌面/算法集合/ime-core/pinyin_map.py` —— `get_weighted_characters()`（第 943 行），更新后的 `generate_candidates()`（第 991 行）
- `/home/tcvdog/桌面/算法集合/ime-core/engine.py` —— 完全重写的 `Engine`，包含 `self.freq_db`、`_text_score()`、`select()` 加分、`save_freq()`
- `/home/tcvdog/桌面/算法集合/ime-core/llm_backend.py` —— `rank()` 返回 `(reordered, weight_adj, latency)` 三元组
- `/home/tcvdog/桌面/算法集合/ime-core/config.py` —— 添加了 `freq_db.save_path`
