"""Common validation and normalization helpers."""

from __future__ import annotations

import math
import re
from typing import Any, Optional
from urllib.parse import urlparse


def normalize_whitespace(value: Any) -> Optional[str]:
    """Convert value to stripped single-spaced string, returning None for empty/NaN."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return re.sub(r"\s+", " ", text)


def is_valid_http_url(url: Optional[str]) -> bool:
    """Return True if url is a syntactically valid HTTP or HTTPS URL."""
    if not url:
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
