# IME LLM — 自学习拼音输入法引擎

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个**三层并行、自学习**的拼音输入法引擎。融合本地词库、本地 Ollama 模型和远程 DeepSeek API，通过加权评分合并和多路径拼音切分提高输入准确性，并让系统持续从用户选择和 LLM 反馈中自适应优化。

支持独立 GUI 模式和 Linux IBus 桌面集成。

---

## 核心理念

**三个臭皮匠，顶个诸葛亮。**

普通输入法只有一本死词典——查词频硬排，不会变通。这个输入法让三个"大脑"同时工作：

| 大脑 | 延迟 | 优点 | 缺点 |
|------|------|------|------|
| 本地词库 (Map) | 0ms | 永远可用，不依赖网络 | 死记硬背，不会理解上下文 |
| Ollama 本地模型 | ~300ms | 理解上下文，隐私好 | 算力有限（消费级 GPU） |
| DeepSeek 云端 API | ~2s | 最聪明，知识最广 | 需要联网，按 token 计费 |

三者并行竞赛，谁先回来谁先占位，最终按加权评分合并——不偏袒单一模型。

---

## 技术路线与框架

### 架构总览

```
用户输入 (pinyin)
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  Layer 0: 拼音映射 (pinyin_map.py)                    │
│  ─ 117K+ 拼音组合 / 175K+ 词                          │
│  ─ 零延迟，始终可用                                    │
│  ─ 包含多路径 DP 切分算法                               │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Layer 1: Ollama 本地模型 (可选)                        │
│  ─ qwen2.5:7b, ~300ms 快速重排序                       │
│  ─ GPU 常驻显存 (RTX 3080 约 5.5GB)                    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Layer 2: DeepSeek 远程 API (可选)                      │
│  ─ deepseek-chat, ~2s 上下文感知重排序                   │
│  ─ 异步非阻塞，不卡主线程                                │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
              加权评分合并 (Weighted Merge)
              Map(1.0) + Ollama(1.5) + DeepSeek(2.0)
                       ↓
                  候选词输出
```

### 数据流

```
每按键 → process_map (0ms, 仅地图层)
         │
         ├──→ 立即显示候选词
         │
         └──→ 停止 300ms 后 → process (提交所有 LLM)
                                  │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
               Ollama(并)    DeepSeek(并)     Map + LLM分数
                    │              │
                    ▼              ▼
               加权合并 ←────┬─────┘
                    │
                    ▼
               更新界面候选
                    │
                    ▼
             用户选择 → learner记录 → llm_scores更新
```

### 核心模块

| 模块 | 职责 |
|------|------|
| `engine.py` | 多层编排器：拼音切分→候选生成→LLM提交→结果合并→学习 |
| `pinyin_map.py` | 音节→汉字映射 + 多路径DP切分算法 + 词库 |
| `llm_backend.py` | LLM API 客户端（OpenAI 兼容），含 LRU 缓存 |
| `learner.py` | 用户偏好数据库：持久化用户选择历史 |
| `token_stats.py` | Token 用量统计：分别记录输入/输出，跨会话持久化 |
| `config.py` | 配置管理：环境变量 > 用户配置 > 默认值 |
| `gui.py` | tkinter GUI：三徽章状态 + token 统计 |
| `ibus_main.py` | IBus 桌面引擎：辅助文字 + 候选列表 + 快捷键 |
| `settings.py` | 设置对话框：模式选择 + Ollama/DeepSeek 配置 |

---

## 关键技术决策

### 1. 三层并行竞赛架构

Map/Ollama/DeepSeek **同时提交**到 `ThreadPoolExecutor`（max_workers=2）。谁先返回谁占高位，迟到的降权插入。保证：
- 地图层始终零延迟
- LLM 结果到达时无感刷新
- 网络慢或模型卡顿时不阻塞打字

### 2. 多路径拼音切分 (DP)

使用动态规划生成所有合法拼音切分路径，按音节频率和切分次数评分，取 top 5 路径加权合并候选：

```
输入 "xian"
  ├─ 路径1: [xian]      → 现/先/县  (得分0.98)
  ├─ 路径2: [xi, an]    → 西安/西岸  (得分0.89)
  └─ 路径3: [xia, n]    → 瞎/n?     (得分0.47，被压制)
```

稀有音节和过多切分路径自动降权，有用歧义保持合理权重。

### 3. 加权评分合并

三个来源的候选按权重打分合并，避免单一模型同质化：

```
合并分数 = base_score + Σ(来源权重 / (排名位置 + 1))
```

来源权重：Map=1.0, Ollama=1.5, DeepSeek=2.0

### 4. 300ms 去抖机制

每按键只触发地图层查询，停止 300ms 后才提交 LLM。快速打字时零开销。

### 5. 场景感知学习权重

根据用户行为动态调整学习速率，6 种场景：

| 场景 | 权重 | 含义 |
|------|------|------|
| 用户直接选择 | 1.0 | 最高信任，用户亲自选 |
| LLM及时 + 用户同意 | 0.8 | LLM推荐 + 用户确认，很靠谱 |
| LLM及时 + 用户没选 | 0.4 | LLM推荐但用户无反馈，半信半疑 |
| LLM迟到 + 同意用户 | 0.5 | 网慢但英雄所见略同 |
| LLM迟到 + 不同意 | 0.15 | 来晚了还不准，几乎忽略 |
| 组合学习 (连续单字) | 1.0 | 自动发现用户常用词组 |

### 6. 组合学习

检测 3 秒窗口内的连续单字选择，自动识别用户逐字选出的词组并记忆。例如连续选"相"→"同"后，系统自动学习"相同"为一个词组。

### 7. LLM 总开关

设置界面或 `Ctrl+Shift+L`（IBus）一键关闭所有 LLM，退回纯本地词库模式。适合离线或不计费的场景。支持 `LLM_ENABLED=0` 环境变量。

### 8. Token 用量统计

独立模块 `token_stats.py` 累计记录 DeepSeek 的输入/输出 token 数，持久化到 `~/.cache/ime-llm/token_stats.json`。IBus 辅助文字区实时显示：

```
M✓ D✓               I:1234 O:56
```

---

## 项目结构

```
ime-llm/
├── ime-core/                   ← 核心引擎
│   ├── main.py                 ← 入口（GUI 模式）
│   ├── engine.py               ← 多层编排器
│   ├── pinyin_map.py           ← 音节→汉字映射 + 词库
│   ├── llm_backend.py          ← LLM API 客户端
│   ├── learner.py              ← 用户偏好学习
│   ├── token_stats.py          ← Token 用量统计
│   ├── config.py               ← 配置管理
│   ├── gui.py                  ← tkinter GUI
│   ├── ibus_main.py            ← IBus 桌面引擎
│   ├── settings.py             ← 设置对话框
│   ├── build_sogou_dict.py     ← 搜狗词库导入
│   ├── import_fcitx_dict.py    ← fcitx5/libpinyin 词库导入
│   ├── build_word_dict.py      ← Ollama 词库生成
│   ├── word_dict_builder.py    ← 后台连续词库构建
│   ├── update_map.py           ← 批量地图权重更新
│   └── data/
│       ├── word_dict.json      ← 词库 (117K+ 拼音组合)
│       ├── llm_scores.json     ← 累计 LLM 反馈分数
│       └── config.json         ← 词库频率配置
│
├── ime-ibus/                   ← IBus 桌面集成
│   ├── ime-llm.xml             ← IBus 组件注册
│   ├── ime-llm.svg             ← 引擎图标
│   └── install.sh              ← 系统安装脚本
│
├── validate-ime/               ← 测试套件
│   └── validate.py             ← LLM 转换准确率测试
│
├── README.md
└── anchor-summary.md
```

---

## 快速开始

```bash
# 运行 GUI 原型（纯 Python 标准库，零依赖）
cd ime-core
python3 main.py

# 安装为系统 IBus 引擎
sudo bash ime-ibus/install.sh
```

### 配置 LLM（可选）

```bash
# DeepSeek（远程，最准）
export LLM_API_KEY="sk-..."
export LLM_ENDPOINT="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"

# Ollama（本地，快速）
export OLLAMA_ENDPOINT="http://localhost:11434/v1"
export OLLAMA_MODEL="qwen2.5:7b"

# 关闭 LLM（纯本地）
export LLM_ENABLED=0
```

### 配置界面

齿轮按钮或 `Ctrl+Shift+L`（IBus）切换 LLM 开关。配置文件 `~/.config/ime-llm/config.json`：

```json
{
  "llm_enabled": true,
  "mode": "map_ollama_deepseek",
  "llm": {
    "endpoint": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key": ""
  },
  "ollama": {
    "endpoint": "http://localhost:11434/v1",
    "model": "qwen2.5:7b",
    "timeout": 5
  }
}
```

四种模式：`map_only` | `map_ollama` | `map_deepseek` | `map_ollama_deepseek`

---

## 持久化数据

| 文件 | 内容 |
|------|------|
| `data/word_dict.json` | 基础词库 + LLM 优化排序 |
| `data/llm_scores.json` | 累计 LLM 反馈加权分数 |
| `~/.cache/ime-llm/user_weights.json` | 用户选择偏好 |
| `~/.cache/ime-llm/token_stats.json` | Token 用量统计 |
| `~/.config/ime-llm/config.json` | 用户配置 |

---

## 前景预测与技术方向

### 近期（1-3 个月）

1. **模糊拼音支持** — zh/z, sh/s, ch/c, n/l, r/l 等常见混淆，大幅降低方言用户的纠错成本
2. **形码辅助** — 五笔/仓颉等形码作为第 4 层，解决"会写不会读"的痛点
3. **用户词库同步** — 多设备间同步学习数据，跨设备无缝体验
4. **更紧凑的 Token 统计** — 根据使用量自动切换 K/M 单位

### 中期（3-6 个月）

1. **纠错学习** — 用户修改候选后自动记住修正模式（"这次选错了，应该选第 3 个"）
2. **句子级输入优化** — 整句连打时，DeepSeek 做一次性整句转换而非逐词重排
3. **性能优化** — `generate_candidates` 递归效率优化，减少笛卡尔积爆炸
4. **分音节维护 llm_scores** — 不同切分路径的 LLM 反馈分开存储，更精细

### 远期（6-12 个月）

1. **个性化语言模型** — 基于用户历史微调小模型，在本地运行专属语言模型
2. **上下文记忆增强** — 记住用户在特定话题中的用词偏好（技术文档 vs 日常聊天）
3. **多模态输入** — 语音输入辅助纠错、手写混输
4. **联邦学习** — 用户词库匿名聚合，共享高频新词趋势（不泄露个人隐私）

### 技术演进路线

```
当前 ──────────────────────────────────────────→ 未来
                                                 
纯工程优化                     ML 驱动
─────────                     ────────
规则切分 (DP)                 可学习切分
固定权重合并                  自适应权重 (RL)
用户规则学习                  强化学习反馈
离线词库                      实时新词发现
手动配置                      自动参数调优
```

---

## Windows 版本路线图

当前版本基于 Linux IBus 框架开发。未来改造为 Windows 版本的技术路线：

### IME 框架层

| Linux (当前) | Windows (目标) |
|---|---|
| IBus Engine (D-Bus 服务) | Windows Text Services Framework (TSF) |
| `ibus_main.py` 作为后端 | `win_main.py` 作为后端，通过 TSF COM 接口注册 |
| GLib MainLoop 事件驱动 | Windows 消息循环 (PeekMessage/DispatchMessage) |
| `ibus-daemon` 管理生命周期 | `CTfMonitor`/COM 注册管理生命周期 |

### 引擎层（无需改动）

```
ime-core/
├── engine.py          ← 平台无关，直接复用
├── pinyin_map.py      ← 纯数据+算法，直接复用
├── llm_backend.py     ← HTTP API 调用，直接复用
├── learner.py         ← 文件 I/O，直接复用
├── token_stats.py     ← 文件 I/O，直接复用
├── config.py          ← 文件 I/O，直接复用
└── data/              ← 词库数据，直接复用
```

核心引擎 7 个模块全部平台无关，**零修改**迁移。

### 需要重写的模块（约 500 行）

| 模块 | Linux | Windows |
|------|-------|---------|
| 入口 | `ibus_main.py` | `win_main.py` (Python + ctypes + TSF) |
| GUI 原型 | `gui.py` (tkinter) | `gui.py` 可选 (tkinter 在 Windows 也可运行) |
| 安装 | `install.sh` | `install.ps1` 或 MSI 安装包 |
| 键盘钩子 | IBus key event | `SetWindowsHookEx(WH_KEYBOARD_LL)` + TSF key sink |
| 候选窗口 | IBus LookupTable | TSF `ITfCandidateListUIElement` 或自绘窗口 |

### 技术选型（Windows）

```
Windows TSF 输入法
      │
┌─────┴──────┐
│ Python 3.x  │ ← 与 Linux 版共用 ime-core 核心
│ ctypes TSF  │ ← 直接调用 COM 接口，无需 C++ 编译
│ 自绘候选窗  │ ← pywin32 / ctypes + GDI
└─────┬──────┘
      │
┌─────┴──────┐
│ Ollama/WSL │ ← 本地模型可选
│ DeepSeek   │ ← 远程 API，平台无关
└────────────┘
```

Windows 版本的核心策略：**用 Python 的 ctypes 直接调用 TSF COM 接口**，避免 C++ 编译和 DLL 注册的复杂性。用户只需 `pip install pywin32` 即可运行。

### 移植步骤

1. **阶段一** — TSF 骨架：注册 CLSID，实现 `ITfTextInputProcessor`，接收键盘事件
2. **阶段二** — 候选窗口：接管焦点窗口，弹出候选列表，支持键盘选择
3. **阶段三** — 引擎集成：对接 `ime-core` 核心模块，完成输入链路
4. **阶段四** — 安装包：`pip install ime-llm-windows` 一键安装
