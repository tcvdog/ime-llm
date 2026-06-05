"""
IME operation logger — records pinyin input, output, and all key events.

Log location: ~/.cache/ime-llm/log/ime.log
Auto-rotates at 10MB (keeps last 3 rotated files).
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.expanduser("~/.cache/ime-llm/log")
_LOG_PATH = os.path.join(_LOG_DIR, "ime.log")

_handler: logging.Handler | None = None
_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get or create the IME operation logger."""
    global _logger, _handler
    if _logger is not None:
        return _logger

    os.makedirs(_LOG_DIR, exist_ok=True)

    _logger = logging.getLogger("ime-ops")
    _logger.setLevel(logging.INFO)

    # Rotating file handler: 10 MB per file, keep 3 backups
    _handler = RotatingFileHandler(
        _LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d|%(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(formatter)
    _logger.addHandler(_handler)

    # Log session start
    _logger.info("SESSION|start")
    return _logger


def log_key(key_name: str, pinyin: str, action: str = "", detail: str = ""):
    """Log a key event. action: compose/commit/backspace/nav/cancel/etc."""
    parts = [f"KEY|{key_name}", f"py={pinyin}"]
    if action:
        parts.append(f"act={action}")
    if detail:
        parts.append(detail)
    get_logger().info("|".join(parts))


def log_commit(text: str, pinyin: str, source: str = ""):
    """Log a committed text output."""
    parts = [f"OUT|{text}", f"py={pinyin}"]
    if source:
        parts.append(f"src={source}")
    get_logger().info("|".join(parts))


def log_candidates(pinyin: str, candidates: list, count: int = 6):
    """Log the current candidate list (abbreviated)."""
    top = " ".join(c[0] if isinstance(c, tuple) else c for c in candidates[:count])
    get_logger().info(f"CDD|py={pinyin}|cnt={len(candidates)}|top={top}")


def log_state(state: str, detail: str = ""):
    """Log a state change (enable/disable/mode/etc)."""
    parts = [f"STATE|{state}"]
    if detail:
        parts.append(detail)
    get_logger().info("|".join(parts))


def log_engine(event: str, detail: str = ""):
    """Log engine events (LLM results, mode switches)."""
    parts = [f"ENG|{event}"]
    if detail:
        parts.append(detail)
    get_logger().info("|".join(parts))
