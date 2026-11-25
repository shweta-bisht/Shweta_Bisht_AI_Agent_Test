"""Structured logging configuration with secret redaction."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional


class SecretRedactingFilter(logging.Filter):
    """Filter that masks common API key patterns if accidentally included in log messages."""

    _PATTERNS = [
        re.compile(r"(AIza[0-9A-Za-z\-_]{10,})"),
        re.compile(r"(api[_-]?key\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        redacted = msg
        redacted = self._PATTERNS[0].sub("***REDACTED_KEY***", redacted)
        redacted = self._PATTERNS[1].sub(r"\1***REDACTED***", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(log_dir: Path, log_level: str = "INFO", run_id: Optional[str] = None) -> logging.Logger:
    """Configure root logger for console and rotating/run file output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger("commerce_ai")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    redactor = SecretRedactingFilter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redactor)
    logger.addHandler(console_handler)

    log_filename = f"commerce_ai_{run_id}.log" if run_id else "commerce_ai.log"
    file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redactor)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger under `commerce_ai`."""
    return logging.getLogger(f"commerce_ai.{name}")
