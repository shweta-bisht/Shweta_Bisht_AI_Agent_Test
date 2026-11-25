"""Google Gemini text generation service with Pydantic structured output validation."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel

from schemas.product_schema import NormalizedVendorProduct, ProductContentLLMResponse
from services.retry_service import execute_with_retry
from utils.cost_tracker import CostTracker
from utils.logger import get_logger

logger = get_logger("gemini_service")


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    return stripped


class GeminiContentService:
    """Wrapper around Google GenAI SDK for structured product content generation."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        request_timeout: int = 120,
        cost_tracker: Optional[CostTracker] = None,
        client: Optional[Any] = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("GEMINI_API_KEY is required to initialize GeminiContentService.")
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.cost_tracker = cost_tracker
        if client is not None:
            self.client = client
        else:
            timeout_ms = max(1000, int(request_timeout * 1000))
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )

    @staticmethod
    def build_product_prompt(product: NormalizedVendorProduct) -> str:
        """Build a strict, grounded prompt using only vendor-supplied product attributes."""
        bullets_text = (
            "\n".join(f"- {b}" for b in product.bullets)
            if product.bullets
            else "- (No vendor bullets provided)"
        )
        payload = {
            "sku": product.sku,
            "brand": product.brand or "Unavailable",
            "category": product.category or "Unavailable",
            "colour": product.color or "Unavailable",
            "material": product.material_used or "Unavailable",
            "assembly": product.assembly_required or "Unavailable",
            "dimensions": product.dimensions_summary or "Unavailable",
            "weight_lbs": product.primary_weight if product.primary_weight is not None else "Unavailable",
        }
        return (
            "You are a senior e-commerce catalog copywriter. Generate factual product listing content "
            "strictly from the vendor-provided data below.\n\n"
            "STRICT RULES:\n"
            "1. Derive ALL facts solely from the provided vendor attributes and bullets.\n"
            "2. NEVER invent specifications, dimensions, weights, certifications, warranties, or materials.\n"
            "3. If an attribute is missing or 'Unavailable', set its value to null in `attributes`.\n"
            "4. Produce a clear, compelling e-commerce `title` (no placeholder text).\n"
            "5. Produce a well-written, complete-sentence `description` (2-4 sentences, strictly factual).\n"
            "6. Produce `features`: a list of EXACTLY 5 distinct, non-repetitive, complete-sentence key features.\n"
            "7. Return valid JSON matching the requested schema.\n\n"
            f"VENDOR ATTRIBUTES:\n{json.dumps(payload, indent=2)}\n\n"
            f"VENDOR BULLETS:\n{bullets_text}\n"
        )

    def generate_product_content(
        self, product: NormalizedVendorProduct
    ) -> Tuple[ProductContentLLMResponse, Dict[str, Any]]:
        """Call Gemini, validate structured JSON against Pydantic v2, and return metrics."""
        prompt = self.build_product_prompt(product)

        def _invoke() -> Tuple[ProductContentLLMResponse, Dict[str, Any]]:
            t0 = time.perf_counter()
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductContentLLMResponse,
                temperature=0.2,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            latency = round(time.perf_counter() - t0, 3)

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, ProductContentLLMResponse):
                validated = parsed
            elif isinstance(parsed, dict):
                validated = ProductContentLLMResponse.model_validate(parsed)
            else:
                raw_text = getattr(response, "text", None) or ""
                if not raw_text.strip():
                    raise ValueError("Gemini returned empty response text.")
                json_text = _extract_json_block(raw_text)
                validated = ProductContentLLMResponse.model_validate_json(json_text)

            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            total_tokens = int(
                getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or 0
            )
            if self.cost_tracker:
                self.cost_tracker.record_text_usage(
                    self.model_name, prompt_tokens, completion_tokens
                )

            return validated, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": latency,
            }

        (validated_resp, metrics), retries_used = execute_with_retry(
            func=_invoke,
            operation_name=f"Gemini content generation ({product.sku})",
            max_retries=self.max_retries,
            cost_tracker=self.cost_tracker,
        )
        metrics["retries"] = retries_used
        return validated_resp, metrics

    def generate_structured_json(
        self,
        prompt: str,
        schema_cls: type[BaseModel],
        operation_name: str = "Gemini structured JSON",
    ) -> BaseModel:
        """Generic structured JSON helper (used by competitor evaluation)."""

        def _invoke() -> BaseModel:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=0.1,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            usage = getattr(response, "usage_metadata", None)
            if usage and self.cost_tracker:
                self.cost_tracker.record_text_usage(
                    self.model_name,
                    int(getattr(usage, "prompt_token_count", 0) or 0),
                    int(getattr(usage, "candidates_token_count", 0) or 0),
                )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema_cls):
                return parsed
            if isinstance(parsed, dict):
                return schema_cls.model_validate(parsed)
            raw_text = getattr(response, "text", None) or ""
            return schema_cls.model_validate_json(_extract_json_block(raw_text))

        result, _ = execute_with_retry(
            func=_invoke,
            operation_name=operation_name,
            max_retries=self.max_retries,
            cost_tracker=self.cost_tracker,
        )
        return result
