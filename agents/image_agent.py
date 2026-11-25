"""Lifestyle Image Generation Agent using reference-image-to-image Gemini API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from config import AppConfig
from schemas.execution_schema import ImageGenerationRecord
from schemas.product_schema import NormalizedVendorProduct
from services.gemini_image_service import GeminiImageService
from services.retry_service import failure_category
from utils.cost_tracker import CostTracker
from utils.file_utils import ensure_dir, safe_filename
from utils.logger import get_logger

logger = get_logger("image_agent")


class LifestyleImageAgent:
    """Downloads vendor reference images and generates lifestyle product photography."""

    def __init__(
        self,
        config: AppConfig,
        image_service: Optional[GeminiImageService] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.config = config
        self.image_service = image_service
        self.cost_tracker = cost_tracker

    def _get_service(self) -> GeminiImageService:
        if self.image_service is None:
            self.image_service = GeminiImageService(
                api_key=self.config.gemini_api_key or "",
                model_name=self.config.gemini_image_model,
                max_retries=self.config.max_retries,
                request_timeout=self.config.request_timeout,
                cost_tracker=self.cost_tracker,
            )
        return self.image_service

    def _process_single_sku(
        self,
        product: NormalizedVendorProduct,
        original_dir: Path,
        lifestyle_dir: Path,
    ) -> ImageGenerationRecord:
        stem = safe_filename(product.sku)
        original_path = original_dir / f"{stem}.jpg"
        lifestyle_path = lifestyle_dir / f"{stem}.jpg"

        if not product.image_url or not product.image_url_valid:
            return ImageGenerationRecord(
                sku=product.sku,
                image_url=product.image_url,
                status="failed",
                error="Missing or invalid Image URL 1.",
                error_category="invalid_url",
            )

        service = self._get_service()

        # Check cache for recovery if both original & lifestyle exist and are valid
        if self.config.cache_enabled and original_path.exists() and lifestyle_path.exists():
            try:
                service.validate_and_normalize_image_bytes(original_path.read_bytes())
                service.validate_and_normalize_image_bytes(lifestyle_path.read_bytes())
                logger.info("Reusing valid cached lifestyle image for SKU %s", product.sku)
                return ImageGenerationRecord(
                    sku=product.sku,
                    image_url=product.image_url,
                    original_image_path=str(original_path),
                    lifestyle_image_path=str(lifestyle_path),
                    original_valid=True,
                    lifestyle_valid=True,
                    status="completed",
                )
            except Exception:
                pass

        try:
            ref_bytes, ref_mime = service.download_and_validate_original(
                product.image_url, original_path
            )
        except Exception as exc:
            logger.error("Original image download/validation failed for %s: %s", product.sku, exc)
            return ImageGenerationRecord(
                sku=product.sku,
                image_url=product.image_url,
                status="failed",
                error=f"Reference image error: {exc}",
                error_category=failure_category(exc),
            )

        try:
            gen_meta = service.generate_lifestyle_image(
                product=product,
                reference_bytes=ref_bytes,
                reference_mime=ref_mime,
                lifestyle_path=lifestyle_path,
            )
            return ImageGenerationRecord(
                sku=product.sku,
                image_url=product.image_url,
                original_image_path=str(original_path),
                lifestyle_image_path=gen_meta["lifestyle_image_path"],
                original_valid=True,
                lifestyle_valid=True,
                status="completed",
                latency_seconds=gen_meta.get("latency_seconds", 0.0),
                retries=gen_meta.get("retries", 0),
            )
        except Exception as exc:
            logger.error("Lifestyle image generation failed for %s: %s", product.sku, exc)
            return ImageGenerationRecord(
                sku=product.sku,
                image_url=product.image_url,
                original_image_path=str(original_path),
                original_valid=True,
                lifestyle_valid=False,
                status="failed",
                error=str(exc),
                error_category=failure_category(exc),
            )

    def run(
        self,
        products: Sequence[NormalizedVendorProduct],
        image_dir: Optional[Path] = None,
    ) -> List[ImageGenerationRecord]:
        if self.config.skip_images:
            logger.info("Lifestyle image generation skipped (--skip-images).")
            return [
                ImageGenerationRecord(
                    sku=p.sku,
                    image_url=p.image_url,
                    status="skipped",
                    error="Skipped (--skip-images)",
                )
                for p in products
            ]

        base_dir = image_dir or self.config.image_dir
        original_dir = ensure_dir(base_dir / "original")
        lifestyle_dir = ensure_dir(base_dir / "lifestyle")

        limit = self.config.image_limit
        if limit is not None:
            valid_candidates = [p for p in products if p.image_url and p.image_url_valid]
            target_products = valid_candidates[:limit]
            target_skus = {p.sku for p in target_products}
            logger.info(
                "Applying --image-limit=%d: selected %d valid product(s) out of %d total.",
                limit,
                len(target_products),
                len(products),
            )
        else:
            target_products = list(products)
            target_skus = {p.sku for p in target_products}

        ordered_records: Dict[str, ImageGenerationRecord] = {}
        if target_products:
            max_workers = max(1, min(self.config.max_concurrent_requests, len(target_products)))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {
                    pool.submit(
                        self._process_single_sku, product, original_dir, lifestyle_dir
                    ): product.sku
                    for product in target_products
                }
                for future in as_completed(future_map):
                    sku = future_map[future]
                    ordered_records[sku] = future.result()

        results: List[ImageGenerationRecord] = []
        for p in products:
            if p.sku in ordered_records:
                results.append(ordered_records[p.sku])
            elif p.sku not in target_skus:
                results.append(
                    ImageGenerationRecord(
                        sku=p.sku,
                        image_url=p.image_url,
                        status="skipped",
                        error=f"Skipped (--image-limit={limit})",
                    )
                )

        processed = sum(1 for r in results if r.status in {"completed", "failed"})
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")
        logger.info(
            "Lifestyle image generation finished: %d processed, %d completed, %d failed, %d skipped.",
            processed,
            completed,
            failed,
            skipped,
        )
        return results
