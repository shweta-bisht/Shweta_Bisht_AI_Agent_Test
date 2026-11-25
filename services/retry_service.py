"""Exponential backoff retry utility for external API and HTTP calls.

Error classification:
- 400/404: Non-retryable unless demonstrably transient (e.g., model not found, bad request).
- 401/403: Non-retryable authentication or permission errors.
- 429 with quota_limit=0 or daily exhaustion: Non-retryable; recorded as quota_exhausted.
- 429 with server RetryInfo delay: Respect server delay, retry once if delay <= MAX_QUOTA_RETRY_DELAY_S.
- 5xx / network: Retryable with exponential backoff.
"""

from __future__ import annotations

import re
import random
import time
from typing import Callable, Optional, Tuple, Type, TypeVar

from utils.cost_tracker import CostTracker
from utils.logger import get_logger

logger = get_logger("retry_service")

T = TypeVar("T")

# If the server RetryInfo delay exceeds this threshold, treat as daily exhaustion (non-retryable).
MAX_QUOTA_RETRY_DELAY_S = 300  # 5 minutes


class NonRetryableError(Exception):
    """Error indicating a permanent failure (e.g., invalid credentials, bad config,
    zero-quota exhaustion). Will never be retried."""


class QuotaExhaustedError(NonRetryableError):
    """Raised when a Gemini API 429 reports zero quota or a daily limit exhaustion.
    Extends NonRetryableError so the retry loop stops immediately."""


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted for a transient error."""


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

def _extract_retry_delay_seconds(msg: str) -> Optional[float]:
    """Parse a server-provided RetryInfo delay from the error message string.

    Handles formats like:
    - 'retrydelay': '68515s'
    - "retry in 19h1m55s"
    Returns the total delay in seconds, or None if not found.
    """
    # JSON-style: 'retryDelay': '12345s'
    m = re.search(r"retrydelay['\"]?\s*:\s*['\"]?(\d+)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Prose: "retry in 5m30s" or "retry in 19h1m55s"
    m2 = re.search(
        r"retry\s+in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?",
        msg,
        re.IGNORECASE,
    )
    if m2:
        hours = float(m2.group(1) or 0)
        mins = float(m2.group(2) or 0)
        secs = float(m2.group(3) or 0)
        total = hours * 3600 + mins * 60 + secs
        if total > 0:
            return total
    return None


def _is_zero_quota_exhaustion(msg: str) -> bool:
    """Return True if the error indicates a quota limit of zero (free-tier with no allocation)."""
    return "limit: 0" in msg or "limit:0" in msg.replace(" ", "")


def _is_model_not_found(msg: str) -> bool:
    """Return True for 404 model-not-found errors."""
    return ("404" in msg and "not found" in msg.lower()) or "model not found" in msg.lower()


def _is_bad_request(msg: str) -> bool:
    """Return True for 400 bad-request errors that are configuration mistakes."""
    return "400" in msg and ("bad request" in msg.lower() or "invalid" in msg.lower())


def classify_api_error(exc: Exception) -> str:
    """Classify an API exception into one of:
      'auth'            - 401/403 authentication or permission failure
      'not_found'       - 404 model/resource not found
      'bad_request'     - 400 configuration or schema error
      'quota_zero'      - 429 with quota limit of zero (never retryable)
      'quota_daily'     - 429 with daily exhaustion (long RetryInfo delay, not worth retrying)
      'rate_limit'      - 429 temporary rate limit with short server delay (may retry once)
      'server_error'    - 5xx transient server error (retry with backoff)
      'network'         - Connection / timeout error (retry with backoff)
      'non_retryable'   - Explicitly marked NonRetryableError
      'transient'       - Other retryable error
    """
    if isinstance(exc, NonRetryableError):
        return "non_retryable"

    msg = str(exc)
    msg_lower = msg.lower()

    if any(m in msg_lower for m in ("401", "403", "unauthorized", "permission denied", "api key not valid", "api_key_invalid", "invalid api key")):
        return "auth"
    if _is_model_not_found(msg_lower):
        return "not_found"
    if _is_bad_request(msg_lower):
        return "bad_request"
    if "429" in msg or "resource_exhausted" in msg_lower:
        if _is_zero_quota_exhaustion(msg):
            return "quota_zero"
        delay = _extract_retry_delay_seconds(msg)
        if delay is not None and delay > MAX_QUOTA_RETRY_DELAY_S:
            return "quota_daily"
        return "rate_limit"
    if any(c in msg for c in ("500", "502", "503", "504")):
        return "server_error"
    if any(m in msg_lower for m in ("timeout", "connection", "network", "connectionerror", "timedout")):
        return "network"
    return "transient"


NON_RETRYABLE_CLASSES = {"auth", "not_found", "bad_request", "quota_zero", "quota_daily", "non_retryable"}


def is_non_retryable_exception(exc: Exception) -> bool:
    """Return True if the exception should never be retried."""
    return classify_api_error(exc) in NON_RETRYABLE_CLASSES


def failure_category(exc: Exception) -> str:
    """Map API and validation errors to stable categories for execution reports."""
    message = str(exc).lower()
    if isinstance(exc, QuotaExhaustedError) or "quota exhausted" in message:
        return "quota_exhausted"
    if "invalid image url" in message:
        return "invalid_url"
    if any(term in message for term in ("invalid image", "empty image", "corrupt bytes")):
        return "validation_error"

    error_class = classify_api_error(exc)
    if error_class == "auth":
        return "auth_error"
    if error_class == "not_found" or (
        "model" in message and any(term in message for term in ("unavailable", "does not support"))
    ):
        return "model_unavailable"
    if error_class == "bad_request":
        return "validation_error"
    if error_class in {"rate_limit", "server_error", "network", "transient"}:
        return "transient"
    return "api_error"


def execute_with_retry(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = 3,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    cost_tracker: Optional[CostTracker] = None,
) -> Tuple[T, int]:
    """Execute ``func`` with error-class-aware retry logic.

    - auth / 404 / bad_request / zero-quota / daily-quota: fail immediately, no retry.
    - rate_limit (429 with short RetryInfo): sleep the server-specified delay, retry once.
    - server_error / network / transient: exponential backoff up to ``max_retries``.
    Returns ``(result, retries_used)``.
    """
    attempt = 0
    while True:
        try:
            result = func()
            return result, attempt
        except retryable_exceptions as exc:
            error_class = classify_api_error(exc)

            if error_class in NON_RETRYABLE_CLASSES:
                if cost_tracker:
                    cost_tracker.record_failure()
                # Wrap quota errors clearly so callers can distinguish them.
                if error_class in ("quota_zero", "quota_daily"):
                    raise QuotaExhaustedError(
                        f"{operation_name}: quota exhausted ({error_class}) — {exc}"
                    ) from exc
                raise

            if attempt >= max_retries:
                if cost_tracker:
                    cost_tracker.record_failure()
                raise RetryExhaustedError(
                    f"{operation_name} failed after {attempt + 1} attempt(s): {exc}"
                ) from exc

            attempt += 1
            if cost_tracker:
                cost_tracker.record_retry()

            if error_class == "rate_limit":
                server_delay = _extract_retry_delay_seconds(str(exc))
                if server_delay is not None:
                    sleep_secs = min(server_delay, max_delay)
                    logger.warning(
                        "%s rate-limited (attempt %d/%d); server requests retry in %.0fs, "
                        "sleeping %.2fs...",
                        operation_name, attempt, max_retries + 1, server_delay, sleep_secs,
                    )
                    time.sleep(sleep_secs)
                    continue

            sleep_seconds = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0.0, 0.35 * sleep_seconds)
            total_sleep = sleep_seconds + jitter
            logger.warning(
                "%s failed (attempt %d/%d) [%s]: %s. Retrying in %.2fs...",
                operation_name, attempt, max_retries + 1, error_class, exc, total_sleep,
            )
            time.sleep(total_sleep)
