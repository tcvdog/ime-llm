# IME LLM — AI-Powered Pinyin Input Method

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **self-learning pinyin input method engine** powered by local + cloud LLMs. Supports standalone GUI and Linux IBus integration.

---

## Architecture

```
User Input (pinyin)
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 0: Pinyin Map (pinyin_map.py)                 │
│  ─ 57,991 words词典 / 36,545 pinyin combos           │
│  ─ 175,017 words (imported from fcitx5/libpinyin)     │
│  ─ Zero latency, always available                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Layer 1: Ollama Local Model (optional)               │
│  ─ qwen2.5:7b, ~300-800ms fast re-ranking            │
│  ─ VRAM-persistent, real-time inference              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2: DeepSeek Remote API (optional)              │
│  ─ deepseek-chat, ~2s context-aware re-ranking       │
│  ─ Async, non-blocking                                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
         Weighted Merge — 3 sources combined
         Map(1.0) + Ollama(1.5) + DeepSeek(2.0)
         ↓
         Candidates Output
```

### Pipeline

```
Map(0ms) ──┬──→ 显示结果 ──→ 你选择 → learner记录 → 下次map提升
           │
           ├──→ Ollama(并行, ~0.5s) ──→ 加权合并 → 更新llm_scores
           │
           └──→ DeepSeek(并行, ~2s) ──→ 加权合并 → 更新llm_scores
```

Key design: **Map/Ollama/DeepSeek run in parallel**. Results are merged by weighted scoring as they arrive. The first to return doesn't "win" — all three contribute proportionally.

### Learning System

Three feedback loops continuously improve accuracy:

```
你选"相同" → user_weights: "xiang tong"→"相同"+1.0
           → phrase_boost: 下次地图排第一
           
Ollama返回 → llm_scores: 相同+1.5, 相通+0.75, 想通+0.5...
           → WORD_MAP reordered

DeepSeek返回 → llm_scores: 相同+2.0, 相通+1.0...
              → WORD_MAP reordered
```

All persisted across sessions:
- `data/word_dict.json` — base word dictionary (117K+ pinyin combos)
- `data/llm_scores.json` — accumulated LLM feedback weights
- `~/.cache/ime-llm/user_weights.json` — user selection history

---

## Project Structure

```
ime-llm/
├── ime-core/                   ← Core engine
│   ├── main.py                 ← Entry point (GUI mode)
│   ├── engine.py               ← Multi-layer orchestrator
│   ├── pinyin_map.py           ← Syllable→character map + word dictionary
│   ├── llm_backend.py          ← LLM API client (OpenAI-compatible)
│   ├── learner.py              ← User preference learning
│   ├── config.py               ← Configuration (env > file > defaults)
│   ├── gui.py                  ← tkinter GUI (3-badge status display)
│   ├── ibus_main.py            ← IBus engine (Linux desktop)
│   ├── settings.py             ← Settings dialog (mode/Ollama/DeepSeek)
│   ├── build_sogou_dict.py     ← Import Sogou word list
│   ├── import_fcitx_dict.py    ← Import fcitx5/libpinyin dictionary
│   ├── build_word_dict.py      ← Ollama-generated word list
│   ├── word_dict_builder.py    ← Background continuous word list builder
│   ├── update_map.py           ← Batch map weight updater
│   └── data/
│       ├── word_dict.json      ← Word dictionary (117K entries)
│       └── llm_scores.json     ← Accumulated LLM feedback
│
├── ime-ibus/                   ← IBus desktop integration
│   ├── ime-llm.xml             ← IBus component registration
│   ├── ime-llm.svg             ← Engine icon
│   └── install.sh              ← System install script
│
├── validate-ime/               ← Test suite
│   └── validate.py             ← LLM conversion accuracy test
│
├── settings.py                 ← Settings dialog
├── README.md
└── anchor-summary.md
```

---

## Quick Start

```bash
# Run GUI prototype (no dependencies needed)
cd ime-core
python3 main.py

# Install as system IBus engine
sudo bash ime-ibus/install.sh
```

### Optional: Enable LLM

```bash
# DeepSeek (remote, best accuracy)
export LLM_API_KEY="sk-..."
export LLM_ENDPOINT="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"

# Ollama (local, fast)
export OLLAMA_ENDPOINT="http://localhost:11434/v1"
export OLLAMA_MODEL="qwen2.5:7b"
```

### Configuration via Settings GUI

Click the **⚙** button in the GUI, or edit `~/.config/ime-llm/config.json`:

```json
{
  "mode": "map_ollama_deepseek",
  "llm": {
    "endpoint": "https://api.deepseek.com/v1",
    "model": "deepseek-chat"
  },
  "ollama": {
    "endpoint": "http://localhost:11434/v1",
    "model": "qwen2.5:7b",
    "timeout": 10
  }
}
```

Four modes: `map_only` | `map_ollama` | `map_deepseek` | `map_ollama_deepseek`

---

## Technical Highlights

| Feature | Detail |
|---------|--------|
| **Parallel LLM** | Ollama + DeepSeek run simultaneously, results merged by weighted scoring |
| **300ms debounce** | No LLM request on every keystroke — waits for typing pause |
| **Self-learning** | 3 feedback loops: user selection, Ollama, DeepSeek — all update map weights |
| **Compound detection** | Select "相" + "同" → system learns "相同" as a phrase |
| **Late LLM learning** | Even if you selected before LLM returned, result is learned with reduced weight |
| **Scenario-aware weights** | 6 scenarios with different learning rates (0.15~1.0) |
| **117K word dictionary** | Imported from fcitx5/libpinyin, validated against SYLLABLE_MAP |
| **Three-badge UI** | Shows Map ✓ / Ollama ✓ / DeepSeek ✓ status simultaneously |
| **VRAM-persistent** | qwen2.5:7b stays loaded in GPU memory (~0.3s inference) |
| **Zero deps GUI** | GUI mode uses only Python stdlib |

---

## Data Sharing

The learning data can be shared between users:

```bash
# Back up / share
ime-core/data/word_dict.json       # Base dictionary + LLM-refined ordering
ime-core/data/llm_scores.json      # Accumulated LLM feedback scores
~/.cache/ime-llm/user_weights.json # User selection preferences
```

---

## License

MIT
