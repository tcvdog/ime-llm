# IME LLM — AI-Powered Pinyin Input Method

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **three-layer pinyin input method engine** combining traditional pinyin mapping, local vocabulary learning cache, and LLM-based context-aware disambiguation. Supports standalone GUI and Linux IBus integration.

---

## Architecture

```
User Input (pinyin)
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Layer 0: CharFrequencyDB (freq_db.py)           │
│  Character frequency weights (3512 chars)        │
│  LLM-learned adjustments persisted across sess.  │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Layer 1: Pinyin Map (pinyin_map.py)             │
│  Built-in syllable → character mapping           │
│  Always available, zero latency                  │
│  Guarantees candidates for ANY valid pinyin      │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Layer 1.5: Viterbi + Bigram (viterbi.py,        │
│             bigram_model.py)                     │
│  HMM beam search for sequence disambiguation     │
│  Offline — 2025 bigram entries from word list    │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Layer 2: Local Cache (cache.py)                 │
│  Learns user preferences per context             │
│  Context-sensitive n-gram suffix matching        │
│  Zero latency on cache hit                       │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Layer 3: LLM Backend (llm_backend.py)           │
│  Async re-ranking via OpenAI-compatible API      │
│  Non-blocking — doesn't slow down typing         │
│  Deep context-aware disambiguation               │
│  Supports multiple providers (remote + local)    │
└──────────────────────────────────────────────────┘
               │
               ▼
         Candidates Output
```

### Why layered architecture?

| Layer | Latency | Offline? | Purpose |
|-------|---------|----------|---------|
| CharFrequencyDB | ~0ms | ✅ Yes | Frequency-weighted character ordering |
| Pinyin Map | ~0ms | ✅ Yes | Guaranteed baseline candidates |
| Viterbi + Bigram | ~0ms | ✅ Yes | Sequence-level disambiguation |
| Local Cache | ~0ms | ✅ Yes | Personalization from user corrections |
| LLM | ~500-1500ms | ❌ No | Deep context-aware disambiguation |

Each layer **falls through** to the next — LLM is never a bottleneck for basic typing.

---

## Project Structure

```
ime-llm/
├── ime-core/                   ← Core engine (Python)
│   ├── main.py                 ← Entry point (GUI mode)
│   ├── engine.py               ← Pipeline orchestrator (3 layers + Viterbi)
│   ├── pinyin_map.py           ← Layer 1: syllable→char map + word dictionary
│   ├── cache.py                ← Layer 2: learning cache (context n-gram)
│   ├── llm_backend.py          ← Layer 3: LLM API client + rank/convert
│   ├── freq_db.py              ← CharFrequencyDB (3512 chars, LLM weight adj.)
│   ├── bigram_model.py         ← Bigram language model (add-k + λ interpolation)
│   ├── viterbi.py              ← Viterbi beam search decoder
│   ├── config.py               ← Configuration (env > file > defaults)
│   ├── gui.py                  ← tkinter GUI prototype
│   ├── ibus_main.py            ← IBus engine (Linux desktop)
│   ├── test_engine.py          ← Integration test
│   ├── test_viterbi.py         ← Viterbi unit tests
│   └── data/
│       ├── pinyin_map_ext.json ← Extended syllable→char map (3512 chars, 407 syllables)
│       ├── gen_freq_data.py    ← Frequency data generator script
│       ├── bigram_counts.json  ← Bigram frequency table (2025 entries)
│       ├── unigram_counts.json ← Unigram frequency table (1186 chars)
│       └── gen_bigram_data.py  ← Bigram data generator script
│
├── ime-ibus/                   ← IBus desktop integration
│   ├── ime-llm.xml             ← IBus component registration
│   ├── ime-llm.svg             ← Engine icon
│   ├── install.sh              ← System install script
│   └── start-ibus.sh           ← IBus daemon launcher
│
├── validate-ime/               ← Validation & testing
│   ├── validate.py             ← LLM conversion accuracy test (37 cases)
│   ├── validate_cache.py       ← Cache effectiveness test
│   ├── local_cache.py          ← Standalone cache implementation (validation)
│   ├── test_cases.py           ← 37 test cases (5 categories)
│   ├── test_cases.json         ← JSON-exported test cases
│   ├── requirements.txt        ← Python dependencies
│   └── results/                ← Validation results
│
├── README.md
└── anchor-summary.md           ← Development progress tracker
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Linux Desktop** (for IBus integration)
- **Optional:** An LLM API key (DeepSeek / OpenAI / Ollama)

### 1. Clone

```bash
git clone https://github.com/tcvdog/ime-llm.git
cd ime-llm
```

### 2. Run the GUI Prototype (Standalone)

No dependencies needed — uses only Python standard library:

```bash
cd ime-core
python3 main.py
```

This launches a tkinter window where you can type pinyin and see candidates.

### 3. Run the Integration Test

```bash
cd ime-core
python3 test_engine.py
```

### 4. Set Up LLM (Optional)

To enable Layer 3 (AI-powered disambiguation), set an API key:

```bash
# DeepSeek (recommended, cheapest)
export LLM_API_KEY="sk-..."
export LLM_ENDPOINT="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"

# Or OpenAI
export LLM_API_KEY="sk-..."
export LLM_ENDPOINT="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"

# Or local (Ollama)
export LLM_ENDPOINT="http://localhost:11434/v1"
export LLM_MODEL="qwen2.5:1.5b"
```

### 5. Install as IBus Engine (Linux Desktop)

```bash
cd ime-llm
sudo bash ime-ibus/install.sh
```

Then:
1. Open **IBus Preferences** (`ibus-setup`)
2. Go to **Input Method → Add → Chinese → IME LLM (AI 输入法)**
3. Switch to the engine with **Super+Space** (or the IBus panel menu)

### 6. Validate LLM Accuracy

```bash
cd validate-ime
export LLM_API_KEY="sk-..."
python3 validate.py

# Or with a specific model:
python3 validate.py --endpoint https://api.deepseek.com/v1 --model deepseek-chat

# Cache validation (simulate learning from user corrections):
python3 validate_cache.py
```

---

## Usage

### GUI Prototype

```
┌──────────────────────────────────────────┐
│ 拼音: [xiamian_______________________]   │
│ 输入: 我饿了想吃                        │
│                                          │
│ 候选: [1]虾面 [2]下面 [3]夏眠            │
│       [4]虾免 [5]瞎面 [6]虾棉            │
│                                          │
│ 来源: 拼音映射                           │
└──────────────────────────────────────────┘
```

| Key | Action |
|-----|--------|
| `a-z` | Type pinyin |
| `1-9` | Select candidate (on current page) |
| `Space` | Select top candidate |
| `Enter` | **Commit raw pinyin letters** |
| `+` / `-` | Next / previous candidate page |
| `Backspace` | Delete last pinyin character |
| `Escape` | Cancel composing |
| `,` `.` `?` `!` `:` `;` etc. | Commit composed text + output Chinese punctuation |

### IBus Engine (Linux)

Same keybindings apply system-wide in any application. PageUp/PageDown and cursor-down also navigate candidate pages.

---

## Configuration

Configuration priority (highest first):

1. **Environment variables** — `LLM_API_KEY`, `LLM_ENDPOINT`, `LLM_MODEL`, `IME_CACHE_PATH`
2. **User config file** — `~/.config/ime-llm/config.json`
3. **Defaults** — Embedded in `config.py`

### Default Configuration

```json
{
  "llm": {
    "endpoint": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key": "",
    "timeout": 15
  },
  "cache": {
    "context_window": 6,
    "save_path": "~/.cache/ime-llm/user_cache.json"
  },
  "engine": {
    "max_candidates": 9,
    "llm_fallback": true
  }
}
```

---

## Test Cases

37 test cases across 5 categories:

| Category | Count | Description |
|----------|-------|-------------|
| `basic` | 4 | Unambiguous conversions |
| `disambig` | 16 | **Core:** Context-sensitive disambiguation (虾面/下面, 暑假/书价/书架, etc.) |
| `long_sentence` | 5 | Continuous pinyin without spaces |
| `named_entity` | 4 | Person/place names (李白 vs 三百) |
| `edge` | 8 | Edge cases (multi-phonetic characters, single syllables) |

---

## Technical Highlights

- **N-gram Context Cache**: Learns user preferences with suffix-based fuzzy matching — "想吃的" can match "吃的" from previous training
- **Non-blocking LLM**: LLM refinement runs asynchronously via a polling timer (every 400-500ms), never blocking input
- **Viterbi Beam Search**: HMM-based sequence disambiguation using bigram transition probabilities with add-k smoothing
- **CharFrequencyDB**: 3512-character frequency database with runtime LLM weight adjustments that persist across sessions
- **Multi-Provider LLM**: Supports multiple LLM backends simultaneously (remote API + local Ollama) with runtime switching
- **Greedy Pinyin Segmentation**: Handles both space-separated (`ni hao`) and continuous (`womenyinggaizenmeban`) input
- **Zero External Dependencies**: GUI mode uses only Python stdlib (`tkinter`, `urllib`, `json`)

---

## License

MIT
