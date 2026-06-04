"""Token usage statistics tracker — records cumulative prompt/completion tokens."""

import json
import os

_STATS_DIR = os.path.join(os.path.dirname(__file__), "data")
_STATS_PATH = os.path.join(_STATS_DIR, "token_stats.json")


def _ensure_dir():
    os.makedirs(_STATS_DIR, exist_ok=True)


def _defaults() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def load() -> dict:
    """Load token stats from disk. Returns {prompt_tokens, completion_tokens, calls}."""
    if not os.path.exists(_STATS_PATH):
        return dict(_defaults())
    try:
        with open(_STATS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(_defaults())


def save(stats: dict):
    """Save token stats to disk."""
    _ensure_dir()
    with open(_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def record(prompt: int = 0, completion: int = 0):
    """Add prompt/completion tokens to the cumulative total and persist."""
    if prompt <= 0 and completion <= 0:
        return
    stats = load()
    stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + max(prompt, 0)
    stats["completion_tokens"] = stats.get("completion_tokens", 0) + max(completion, 0)
    stats["calls"] = stats.get("calls", 0) + 1
    save(stats)


def get_stats() -> dict:
    """Return current stats without modifying them."""
    return load()


def reset():
    """Reset all counters to zero."""
    save(_defaults())
