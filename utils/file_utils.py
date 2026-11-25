"""Filesystem and JSON helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """Create directory and parents if missing and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value: str) -> str:
    """Sanitize an identifier (such as a SKU) into a safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "item"


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON deterministically with UTF-8 encoding."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def read_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
