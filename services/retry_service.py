"""Exponential backoff retry utility for external API and HTTP calls."""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, Tuple, Type, TypeVar

from utils.cost_tracker import CostTracker
from utils.logger import get_logger

logger = get_logger("retry_service")

T = TypeVar("T")


class NonRetryableError(Exception):
    """Error indicating a permanent failure (e.g., invalid credentials or bad config)."""


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""


def is_non_retryable_exception(exc: Exception) -> bool:
    """Identify auth/permission/configuration errors that should not be retried."""
    if isinstance(exc, NonRetryableError):
        return True
    msg = str(exc).lower()
    permanent_markers = (
        "api key not valid",
        "api_key_invalid",
        "invalid api key",
        "permission denied",
        "unauthorized",
        "401",
        "403",
    )
    return any(marker in msg for marker in permanent_markers)


def execute_with_retry(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = 3,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    cost_tracker: Optional[CostTracker] = None,
) -> Tuple[T, int]:
    """Execute `func` with exponential backoff + jitter. Returns `(result, retries_used)`."""
    attempt = 0
    while True:
        try:
            result = func()
            return result, attempt
        except retryable_exceptions as exc:
            if is_non_retryable_exception(exc):
                if cost_tracker:
                    cost_tracker.record_failure()
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

            sleep_seconds = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0.0, 0.35 * sleep_seconds)
            total_sleep = sleep_seconds + jitter
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.2fs...",
                operation_name,
                attempt,
                max_retries + 1,
                exc,
                total_sleep,
            )
            time.sleep(total_sleep)
