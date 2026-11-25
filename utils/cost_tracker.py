"""Thread-safe API token, retry, failure, and estimated cost tracker."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict


# Indicative public Gemini API pricing per 1M tokens (USD) for estimation only.
# Actual billing depends on the user's configured model and Google AI tier.
DEFAULT_MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash-image": {"input": 0.15, "output": 30.00},
}


@dataclass
class CostTracker:
    """Accumulates token counts, image generations, retries, and API failures across threads."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    images_generated: int = 0
    search_requests: int = 0
    retries: int = 0
    api_failures: int = 0
    estimated_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _pricing_for(self, model_name: str) -> Dict[str, float]:
        lowered = model_name.lower()
        for key, prices in DEFAULT_MODEL_PRICING_PER_1M.items():
            if key in lowered:
                return prices
        return {"input": 0.15, "output": 0.60}

    def record_text_usage(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> None:
        prices = self._pricing_for(model_name)
        cost = (prompt_tokens / 1_000_000.0) * prices["input"] + (
            completion_tokens / 1_000_000.0
        ) * prices["output"]
        with self._lock:
            self.prompt_tokens += max(0, prompt_tokens)
            self.completion_tokens += max(0, completion_tokens)
            self.total_tokens += max(0, prompt_tokens) + max(0, completion_tokens)
            self.estimated_cost_usd += cost

    def record_image_generation(self, model_name: str, count: int = 1) -> None:
        # Rough indicative ~$0.039 per generated image for flash-image tier when token counts unavailable
        per_image_cost = 0.039 if "flash" in model_name.lower() else 0.06
        with self._lock:
            self.images_generated += count
            self.estimated_cost_usd += count * per_image_cost

    def record_search_request(self, count: int = 1) -> None:
        with self._lock:
            self.search_requests += count

    def record_retry(self, count: int = 1) -> None:
        with self._lock:
            self.retries += count

    def record_failure(self, count: int = 1) -> None:
        with self._lock:
            self.api_failures += count

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "images_generated": self.images_generated,
                "search_requests": self.search_requests,
                "retries": self.retries,
                "api_failures": self.api_failures,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            }
