"""Configuration for the IME prototype.

Config priority (highest first):
  1. Environment variables (LLM_MODE, LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL,
     OLLAMA_ENDPOINT, OLLAMA_MODEL)
  2. User config file (~/.config/ime-llm/config.json)
  3. Defaults
"""

import os
import json

_XDG_CACHE_HOME = os.environ.get(
    "XDG_CACHE_HOME",
    os.path.expanduser("~/.cache"),
)
_XDG_CONFIG_HOME = os.environ.get(
    "XDG_CONFIG_HOME",
    os.path.expanduser("~/.config"),
)
_USER_CONFIG_PATH = os.path.join(_XDG_CONFIG_HOME, "ime-llm", "config.json")

DEFAULT_CONFIG: dict = {
    "mode": "map_ollama_deepseek",  # map_only | map_ollama | map_deepseek | map_ollama_deepseek
    "llm_enabled": True,           # master switch: False forces map_only
    "llm": {
        "endpoint": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key": "",
        "timeout": 15,
    },
    "ollama": {
        "enabled": True,
        "endpoint": "http://localhost:11434/v1",
        "model": "qwen2.5:1.5b",
        "timeout": 5,
    },
    "engine": {
        "max_candidates": 36,
        "page_size": 9,
    },
    "confidence": {
        "enabled": True,       # master switch for confidence skip
        "ratio": 1.5,          # top1/top2 score ratio threshold
        "min_selections": 3,   # user must have selected this word at least N times
    },
    "learning": {
        "save_path": "~/.cache/ime-llm/user_weights.json",
    },
}

# Valid modes
VALID_MODES = ("map_only", "map_ollama", "map_deepseek", "map_ollama_deepseek")


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | None = None) -> dict:
    """Load config with priority: env > user file > defaults."""
    cfg = dict(DEFAULT_CONFIG)

    # 1. User config file
    file_path = path or _USER_CONFIG_PATH
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                cfg = _merge(cfg, json.load(f))
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Environment variables override everything
    if os.environ.get("LLM_ENABLED"):
        cfg["llm_enabled"] = os.environ["LLM_ENABLED"].lower() in ("1", "true", "yes", "on")
    if os.environ.get("LLM_MODE"):
        mode = os.environ["LLM_MODE"]
        if mode in VALID_MODES:
            cfg["mode"] = mode
    if os.environ.get("LLM_API_KEY"):
        cfg["llm"]["api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("LLM_ENDPOINT"):
        cfg["llm"]["endpoint"] = os.environ["LLM_ENDPOINT"]
    if os.environ.get("LLM_MODEL"):
        cfg["llm"]["model"] = os.environ["LLM_MODEL"]
    if os.environ.get("OLLAMA_ENDPOINT"):
        cfg["ollama"]["endpoint"] = os.environ["OLLAMA_ENDPOINT"]
    if os.environ.get("OLLAMA_MODEL"):
        cfg["ollama"]["model"] = os.environ["OLLAMA_MODEL"]

    return cfg


def save_config(config: dict, path: str | None = None) -> None:
    """Save config to user config file."""
    file_path = path or _USER_CONFIG_PATH
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
