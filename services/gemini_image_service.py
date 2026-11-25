"""Google Gemini multimodal image-to-image lifestyle generation service."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from google import genai
from google.genai import types
from PIL import Image

from schemas.product_schema import NormalizedVendorProduct
from services.retry_service import NonRetryableError, execute_with_retry
from utils.cost_tracker import CostTracker
from utils.file_utils import ensure_dir
from utils.logger import get_logger
from utils.validators import is_valid_http_url

logger = get_logger("gemini_image_service")


class ImageGenerationError(Exception):
    """Raised when image download, validation, or Gemini image generation fails."""


class GeminiImageService:
    """Downloads reference product images and generates realistic lifestyle photos with Gemini."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-image",
        max_retries: int = 3,
        request_timeout: int = 120,
        cost_tracker: Optional[CostTracker] = None,
        client: Optional[Any] = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("GEMINI_API_KEY is required to initialize GeminiImageService.")
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

    def download_and_validate_original(
        self, image_url: str, destination_path: Path
    ) -> Tuple[bytes, str]:
        """Download original product image, validate format via Pillow, and save as JPEG."""
        if not is_valid_http_url(image_url):
            raise NonRetryableError(f"Invalid image URL: '{image_url}'")

        ensure_dir(destination_path.parent)

        def _fetch() -> bytes:
            resp = requests.get(image_url, timeout=min(30, self.request_timeout))
            if resp.status_code in (401, 403, 404):
                raise NonRetryableError(
                    f"HTTP {resp.status_code} fetching reference image '{image_url}'."
                )
            resp.raise_for_status()
            if not resp.content:
                raise ImageGenerationError(f"Empty image response from '{image_url}'.")
            return resp.content

        raw_bytes, _ = execute_with_retry(
            func=_fetch,
            operation_name=f"Download reference image ({destination_path.stem})",
            max_retries=self.max_retries,
            cost_tracker=self.cost_tracker,
        )

        validated_jpeg = self.validate_and_normalize_image_bytes(raw_bytes)
        destination_path.write_bytes(validated_jpeg)
        return validated_jpeg, "image/jpeg"

    @staticmethod
    def validate_and_normalize_image_bytes(raw_bytes: bytes) -> bytes:
        """Verify image bytes are decodable with Pillow and return normalized RGB JPEG bytes."""
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise ImageGenerationError("Image has zero width or height.")
                rgb_img = img.convert("RGB")
                out_buf = io.BytesIO()
                rgb_img.save(out_buf, format="JPEG", quality=92)
                return out_buf.getvalue()
        except ImageGenerationError:
            raise
        except Exception as exc:
            raise ImageGenerationError(f"Invalid image format or corrupt bytes: {exc}") from exc

    @staticmethod
    def build_lifestyle_prompt(product: NormalizedVendorProduct) -> str:
        """Construct an interior lifestyle photography prompt preserving the reference furniture item."""
        desc_parts = [
            product.color,
            product.material_used,
            product.bullets[0] if product.bullets else product.category or "furniture piece",
        ]
        product_summary = " | ".join(p for p in desc_parts if p)
        return (
            "Generate a realistic, high-resolution e-commerce lifestyle interior photograph placing the "
            "EXACT furniture product shown in the reference image inside a tastefully styled, natural-light "
            "modern living room or bedroom setting.\n"
            f"Product context: {product_summary}.\n\n"
            "STRICT PRESERVATION REQUIREMENTS:\n"
            "- Preserve the exact product structure, silhouette, proportions, colour, upholstery/wood grain, "
            "hardware, legs, armrests, stitching, and visible design details from the reference image.\n"
            "- Do NOT redesign, replace, recolor, or omit components of the furniture item.\n"
            "- Keep lighting photorealistic with natural shadows, realistic floor contact, and complementary "
            "neutral interior decor suitable for a professional furniture e-commerce listing."
        )

    @staticmethod
    def _extract_image_bytes_from_response(response: Any) -> Optional[bytes]:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None:
                    data = getattr(inline, "data", None)
                    if data:
                        return bytes(data)
        return None

    def generate_lifestyle_image(
        self,
        product: NormalizedVendorProduct,
        reference_bytes: bytes,
        reference_mime: str,
        lifestyle_path: Path,
    ) -> Dict[str, Any]:
        """Pass the reference image + prompt to Gemini multimodal image generation and save output."""
        ensure_dir(lifestyle_path.parent)
        prompt = self.build_lifestyle_prompt(product)

        def _invoke() -> Tuple[bytes, float]:
            t0 = time.perf_counter()
            image_part = types.Part.from_bytes(data=reference_bytes, mime_type=reference_mime)
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[image_part, prompt],
                    config=config,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "response_modalities" in msg or "does not support" in msg or "not supported" in msg:
                    raise NonRetryableError(
                        f"Configured image model '{self.model_name}' does not support multimodal "
                        f"reference-image-to-image generation: {exc}"
                    ) from exc
                raise

            latency = round(time.perf_counter() - t0, 3)
            img_bytes = self._extract_image_bytes_from_response(response)
            if not img_bytes:
                raise ImageGenerationError(
                    f"Gemini model '{self.model_name}' returned no image data in response parts."
                )

            validated_jpeg = self.validate_and_normalize_image_bytes(img_bytes)
            if self.cost_tracker:
                self.cost_tracker.record_image_generation(self.model_name, count=1)
            return validated_jpeg, latency

        (generated_bytes, latency), retries_used = execute_with_retry(
            func=_invoke,
            operation_name=f"Gemini lifestyle image ({product.sku})",
            max_retries=self.max_retries,
            cost_tracker=self.cost_tracker,
        )

        lifestyle_path.write_bytes(generated_bytes)
        return {
            "lifestyle_image_path": str(lifestyle_path),
            "latency_seconds": latency,
            "retries": retries_used,
        }
