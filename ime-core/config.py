"""Configuration for the IME prototype.

Config priority (highest first):
  1. Environment variables (LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL)
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
    "llm": {
        "endpoint": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key": "",
        "timeout": 15,
    },
    "engine": {
        "max_candidates": 36,
        "page_size": 9,
    },
}


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
    if os.environ.get("LLM_API_KEY"):
        cfg["llm"]["api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("LLM_ENDPOINT"):
        cfg["llm"]["endpoint"] = os.environ["LLM_ENDPOINT"]
    if os.environ.get("LLM_MODEL"):
        cfg["llm"]["model"] = os.environ["LLM_MODEL"]

    return cfg
