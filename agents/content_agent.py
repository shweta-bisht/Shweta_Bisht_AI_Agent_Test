"""Product Content Generation Agent using Google Gemini and Pydantic validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from config import AppConfig
from schemas.product_schema import (
    NormalizedVendorProduct,
    ProductAttributes,
    ProductContentItem,
)
from services.gemini_service import GeminiContentService
from utils.cost_tracker import CostTracker
from utils.file_utils import ensure_dir, read_json, safe_filename, write_json
from utils.logger import get_logger

logger = get_logger("content_agent")


class ContentGenerationAgent:
    """Generates and caches validated e-commerce product content for all vendor SKUs."""

    def __init__(
        self,
        config: AppConfig,
        gemini_service: Optional[GeminiContentService] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.config = config
        self.cost_tracker = cost_tracker
        self.gemini_service = gemini_service

    def _get_service(self) -> GeminiContentService:
        if self.gemini_service is None:
            self.gemini_service = GeminiContentService(
                api_key=self.config.gemini_api_key or "",
                model_name=self.config.gemini_model,
                max_retries=self.config.max_retries,
                request_timeout=self.config.request_timeout,
                cost_tracker=self.cost_tracker,
            )
        return self.gemini_service

    def _cache_path(self, sku: str) -> Path:
        return self.config.output_dir / ".cache" / "content" / f"{safe_filename(sku)}.json"

    @staticmethod
    def _enforce_grounding(
        product: NormalizedVendorProduct, attrs: ProductAttributes
    ) -> ProductAttributes:
        """Ensure missing vendor attributes are not fabricated in final output."""
        return ProductAttributes(
            colour=product.color if product.color else None,
            material=product.material_used if product.material_used else None,
            assembly=product.assembly_required if product.assembly_required else None,
            dimensions=product.dimensions_summary if product.dimensions_summary else None,
            weight=f"{product.primary_weight:g} lbs" if product.primary_weight is not None else None,
            brand=product.brand if product.brand else attrs.brand,
            category=product.category if product.category else attrs.category,
        )

    def _process_single_sku(self, product: NormalizedVendorProduct) -> ProductContentItem:
        cache_file = self._cache_path(product.sku)
        if self.config.cache_enabled and cache_file.exists():
            try:
                cached = ProductContentItem.model_validate(read_json(cache_file))
                if cached.status == "completed":
                    cached.from_cache = True
                    logger.info("Loaded cached content for SKU %s", product.sku)
                    return cached
            except Exception as exc:
                logger.warning("Ignoring corrupt cache for SKU %s: %s", product.sku, exc)

        service = self._get_service()
        try:
            llm_resp, metrics = service.generate_product_content(product)
            grounded_attrs = self._enforce_grounding(product, llm_resp.attributes)
            item = ProductContentItem(
                sku=product.sku,
                masked_sku=product.masked_sku,
                brand=product.brand,
                category=product.category,
                title=llm_resp.title,
                description=llm_resp.description,
                features=llm_resp.features,
                attributes=grounded_attrs,
                source_bullets_count=len(product.bullets),
                status="completed",
                from_cache=False,
                prompt_tokens=metrics.get("prompt_tokens", 0),
                completion_tokens=metrics.get("completion_tokens", 0),
                total_tokens=metrics.get("total_tokens", 0),
                latency_seconds=metrics.get("latency_seconds", 0.0),
            )
            if self.config.cache_enabled:
                write_json(cache_file, item.model_dump())
            return item
        except Exception as exc:
            logger.error("Content generation failed for SKU %s: %s", product.sku, exc)
            return ProductContentItem(
                sku=product.sku,
                masked_sku=product.masked_sku,
                brand=product.brand,
                category=product.category,
                title=f"{product.brand or 'Product'} {product.sku} (Generation Failed)",
                description="Unavailable due to content generation failure.",
                features=[
                    "Feature unavailable due to API error (slot 1).",
                    "Feature unavailable due to API error (slot 2).",
                    "Feature unavailable due to API error (slot 3).",
                    "Feature unavailable due to API error (slot 4).",
                    "Feature unavailable due to API error (slot 5).",
                ],
                attributes=ProductAttributes(
                    colour=product.color,
                    material=product.material_used,
                    assembly=product.assembly_required,
                    dimensions=product.dimensions_summary,
                    weight=(
                        f"{product.primary_weight:g} lbs"
                        if product.primary_weight is not None
                        else None
                    ),
                    brand=product.brand,
                    category=product.category,
                ),
                source_bullets_count=len(product.bullets),
                status="failed",
                error=str(exc),
            )

    def run(
        self,
        products: Sequence[NormalizedVendorProduct],
        output_dir: Optional[Path] = None,
    ) -> List[ProductContentItem]:
        target_dir = output_dir or self.config.output_dir
        ensure_dir(target_dir)

        ordered_results: Dict[str, ProductContentItem] = {}
        max_workers = max(1, min(self.config.max_concurrent_requests, len(products) or 1))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(self._process_single_sku, product): product.sku
                for product in products
            }
            for future in as_completed(future_map):
                sku = future_map[future]
                ordered_results[sku] = future.result()

        results = [ordered_results[p.sku] for p in products]
        json_path = target_dir / "product_content.json"
        write_json(json_path, [item.model_dump() for item in results])

        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status != "completed")
        logger.info(
            "Content generation finished: %d completed, %d failed (saved %s).",
            completed,
            failed,
            json_path,
        )
        return results
