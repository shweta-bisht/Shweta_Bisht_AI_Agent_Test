"""Competitor Analysis Agent with rigorous reference-price matching and verifiable web research."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from config import AppConfig
from processors.vendor_processor import (
    build_product_display_name,
    infer_product_type,
    select_representative_products,
)
from schemas.competitor_schema import (
    CompetitorComparisonRecord,
    CompetitorResearchResult,
    ReferencePriceMatch,
)
from schemas.product_schema import NormalizedVendorProduct
from services.gemini_service import GeminiContentService
from services.search_service import SearchHit, SearchService
from utils.cost_tracker import CostTracker
from utils.logger import get_logger

logger = get_logger("competitor_agent")

MIN_REFERENCE_MATCH_CONFIDENCE = 0.75
MIN_COMPETITOR_MATCH_CONFIDENCE = 0.40
TARGET_MARKETPLACES = ("Amazon India", "IKEA", "Pepperfry", "Wooden Street", "Flipkart")


class _CompetitorEvalLLM(BaseModel):
    """Structured evaluation of search hits for one vendor product."""

    selected_hit_index: Optional[int] = Field(
        default=None, description="0-based index of best comparable hit, or null if none match"
    )
    competitor_product_name: Optional[str] = None
    competitor_price: Optional[float] = None
    currency: Optional[str] = None
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    competitive_insight: str


def _norm_code(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def build_sales_price_catalog(sales_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate unique sales items with positive unit prices for reference-price matching."""
    if sales_df.empty:
        return []
    positive = sales_df[(sales_df["price"].notna()) & (sales_df["price"] > 0)].copy()
    if positive.empty:
        return []

    grouped = (
        positive.groupby(["Masked SKU", "itemName"], dropna=False)
        .agg(median_price=("price", "median"), order_count=("orderId", "count"))
        .reset_index()
    )
    catalog: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        catalog.append(
            {
                "masked_sku": str(row["Masked SKU"]) if pd.notna(row["Masked SKU"]) else "",
                "item_name": str(row["itemName"]) if pd.notna(row["itemName"]) else "",
                "median_price": round(float(row["median_price"]), 2),
                "order_count": int(row["order_count"]),
            }
        )
    return catalog


def match_reference_price(
    product: NormalizedVendorProduct,
    sales_catalog: Sequence[Dict[str, Any]],
    threshold: float = MIN_REFERENCE_MATCH_CONFIDENCE,
) -> ReferencePriceMatch:
    """Match a vendor product to sales sheet items using exact, normalized, and fuzzy matching.

    Weak matches below `threshold` are explicitly rejected and reference price is left as None.
    """
    if not sales_catalog:
        return ReferencePriceMatch(
            our_sku=product.sku,
            match_confidence=0.0,
            match_method="none",
            accepted=False,
            reason="Sales catalog has no positive-price items.",
        )

    vendor_codes = [c for c in (product.sku, product.masked_sku) if c]
    vendor_norm_codes = {_norm_code(c) for c in vendor_codes if _norm_code(c)}
    vendor_name = build_product_display_name(product)

    best_score = 0.0
    best_method = "fuzzy_token_sort"
    best_entry: Optional[Dict[str, Any]] = None

    for entry in sales_catalog:
        sales_masked = entry["masked_sku"]
        sales_item = entry["item_name"]

        # 1. Exact SKU match
        if any(vc.strip().lower() in {sales_masked.strip().lower(), sales_item.strip().lower()} for vc in vendor_codes):
            return ReferencePriceMatch(
                our_sku=product.sku,
                matched_sales_sku=sales_masked,
                matched_item_name=sales_item,
                our_reference_price=entry["median_price"],
                match_confidence=1.0,
                match_method="exact_sku",
                accepted=True,
                reason=f"Exact SKU match on '{sales_masked or sales_item}'.",
            )

        # 2. Normalized alphanumeric SKU match
        sales_norms = {_norm_code(sales_masked), _norm_code(sales_item)} - {""}
        if vendor_norm_codes.intersection(sales_norms):
            return ReferencePriceMatch(
                our_sku=product.sku,
                matched_sales_sku=sales_masked,
                matched_item_name=sales_item,
                our_reference_price=entry["median_price"],
                match_confidence=0.95,
                match_method="normalized_sku",
                accepted=True,
                reason=f"Normalized alphanumeric SKU match on '{sales_masked or sales_item}'.",
            )

        # 3. Fuzzy / semantic token matching between vendor name/SKU and sales itemName/Masked SKU
        # Only credit meaningful word overlap or high string similarity; short SKU codes shouldn't falsely match.
        name_score = fuzz.token_set_ratio(vendor_name.lower(), sales_item.lower()) / 100.0
        sku_score = fuzz.ratio(product.sku.lower(), sales_masked.lower()) / 100.0
        # Penalize comparisons where sales itemName looks like an opaque alphanumeric SKU code
        is_opaque_code = bool(re.fullmatch(r"[A-Za-z0-9\-_]{4,16}", sales_item.strip())) and " " not in sales_item.strip()
        candidate_score = sku_score if is_opaque_code else max(name_score, sku_score)

        if candidate_score > best_score:
            best_score = round(candidate_score, 3)
            best_entry = entry
            best_method = "fuzzy_sku" if is_opaque_code else "fuzzy_semantic_name"

    if best_entry and best_score >= threshold:
        return ReferencePriceMatch(
            our_sku=product.sku,
            matched_sales_sku=best_entry["masked_sku"],
            matched_item_name=best_entry["item_name"],
            our_reference_price=best_entry["median_price"],
            match_confidence=best_score,
            match_method=best_method,
            accepted=True,
            reason=f"Accepted {best_method} match (confidence {best_score:.2f} >= {threshold:.2f}).",
        )

    return ReferencePriceMatch(
        our_sku=product.sku,
        matched_sales_sku=best_entry["masked_sku"] if best_entry else None,
        matched_item_name=best_entry["item_name"] if best_entry else None,
        our_reference_price=None,
        match_confidence=best_score,
        match_method=best_method,
        accepted=False,
        reason=(
            f"Rejected weak match (best confidence {best_score:.2f} < {threshold:.2f}; "
            "vendor and sales SKUs use disjoint coding systems without descriptive overlap)."
        ),
    )


def _score_search_hit(product: NormalizedVendorProduct, hit: SearchHit) -> float:
    """Compute comparability confidence [0, 1] between a vendor product and a search hit."""
    ptype = infer_product_type(product).lower()
    haystack = f"{hit.title} {hit.snippet}".lower()
    tokens = [t for t in re.findall(r"[a-z]{3,}", ptype) if t not in {"with", "and", "the", "for", "one"}]
    if not tokens:
        return 0.0
    token_hits = sum(1 for t in tokens if t in haystack)
    type_score = token_hits / len(tokens)

    attr_tokens: List[str] = []
    if product.material_used:
        attr_tokens.extend(re.findall(r"[a-z]{4,}", product.material_used.lower())[:4])
    if product.color:
        attr_tokens.extend(re.findall(r"[a-z]{4,}", product.color.lower())[:2])
    attr_score = (
        sum(1 for t in attr_tokens if t in haystack) / len(attr_tokens)
        if attr_tokens
        else 0.0
    )

    price_bonus = 0.15 if hit.extracted_price is not None else 0.0
    marketplace_bonus = 0.10 if hit.marketplace != "Web" else 0.0
    raw = 0.55 * type_score + 0.20 * attr_score + price_bonus + marketplace_bonus
    return round(min(1.0, raw), 3)


class CompetitorAnalysisAgent:
    """Selects 5 diverse products, matches reference prices, and researches competitors."""

    def __init__(
        self,
        config: AppConfig,
        search_service: Optional[SearchService] = None,
        gemini_service: Optional[GeminiContentService] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.config = config
        self.cost_tracker = cost_tracker
        self.search_service = search_service or SearchService(
            provider=config.search_provider,
            api_key=config.search_api_key,
            cse_id=config.search_cse_id,
            max_retries=config.max_retries,
            request_timeout=min(30, config.request_timeout),
            cost_tracker=cost_tracker,
        )
        self.gemini_service = gemini_service

    @staticmethod
    def build_search_query(product: NormalizedVendorProduct) -> str:
        ptype = infer_product_type(product)
        material = (product.material_used or "").split(",")[0].strip()
        color = (product.color or "").split(",")[0].strip()
        parts = [ptype, material, color, "price", "(site:amazon.in OR site:ikea.com OR site:pepperfry.com OR site:woodenstreet.com OR site:flipkart.com)"]
        return " ".join(p for p in parts if p)

    def _select_best_competitor(
        self,
        product: NormalizedVendorProduct,
        hits: List[SearchHit],
        ref_match: ReferencePriceMatch,
    ) -> CompetitorComparisonRecord:
        today = date.today().isoformat()
        our_name = build_product_display_name(product)
        ref_price = ref_match.our_reference_price
        ref_status = (
            f"Matched ({ref_match.match_method})"
            if ref_match.accepted and ref_price is not None
            else f"Unavailable ({ref_match.reason})"
        )

        if not hits:
            return CompetitorComparisonRecord(
                our_sku=product.sku,
                our_product_name=our_name,
                our_reference_price=ref_price,
                reference_price_status=ref_status,
                reference_match_confidence=ref_match.match_confidence,
                competitive_insight="No verifiable competitor search hits returned.",
                research_date=today,
                match_confidence=0.0,
                status="unavailable",
            )

        # Deterministic ranking first
        scored = sorted(
            ((hit, _score_search_hit(product, hit)) for hit in hits),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best_hit, best_conf = scored[0]
        comp_price = best_hit.extracted_price
        currency = best_hit.currency
        comp_name = best_hit.title
        insight = (
            f"Comparable listing on {best_hit.marketplace} matched by product type and attributes "
            f"(confidence {best_conf:.2f})."
        )

        # Optionally refine selection and insight with Gemini using strictly retrieved hits
        if self.gemini_service is not None and len(scored) > 0:
            try:
                hit_summaries = [
                    {
                        "index": idx,
                        "title": h.title,
                        "marketplace": h.marketplace,
                        "url": h.url,
                        "extracted_price": h.extracted_price,
                        "currency": h.currency,
                        "snippet": h.snippet[:280],
                    }
                    for idx, (h, _) in enumerate(scored[:5])
                ]
                prompt = (
                    "Evaluate the search hits below against our furniture product and select the single "
                    "most comparable competitor listing. NEVER invent a price or URL not in the hits.\n\n"
                    f"OUR PRODUCT: {our_name} | Category: {product.category} | Material: {product.material_used} | "
                    f"Dimensions: {product.dimensions_summary}\n"
                    f"SEARCH HITS:\n{hit_summaries}\n"
                )
                eval_res = self.gemini_service.generate_structured_json(
                    prompt=prompt,
                    schema_cls=_CompetitorEvalLLM,
                    operation_name=f"Competitor hit evaluation ({product.sku})",
                )
                if (
                    eval_res.selected_hit_index is not None
                    and 0 <= eval_res.selected_hit_index < len(scored[:5])
                ):
                    chosen_hit, heuristic_conf = scored[eval_res.selected_hit_index]
                    best_hit = chosen_hit
                    comp_name = eval_res.competitor_product_name or chosen_hit.title
                    comp_price = chosen_hit.extracted_price if chosen_hit.extracted_price is not None else eval_res.competitor_price
                    currency = chosen_hit.currency or eval_res.currency
                    best_conf = round(max(heuristic_conf, eval_res.match_confidence) if eval_res.match_confidence else heuristic_conf, 3)
                    insight = eval_res.competitive_insight
            except Exception as exc:
                logger.warning("LLM refinement skipped for %s, using deterministic ranking: %s", product.sku, exc)

        if best_conf < MIN_COMPETITOR_MATCH_CONFIDENCE:
            return CompetitorComparisonRecord(
                our_sku=product.sku,
                our_product_name=our_name,
                our_reference_price=ref_price,
                reference_price_status=ref_status,
                reference_match_confidence=ref_match.match_confidence,
                competitor_product=comp_name,
                marketplace=best_hit.marketplace,
                competitor_price=None,
                currency=None,
                price_difference=None,
                competitive_insight=(
                    f"Best candidate confidence ({best_conf:.2f}) below comparability threshold "
                    f"({MIN_COMPETITOR_MATCH_CONFIDENCE:.2f}); rejected to avoid mismatched category comparison."
                ),
                source_url=best_hit.url,
                research_date=today,
                match_confidence=best_conf,
                status="rejected_low_confidence",
            )

        price_diff = (
            round(comp_price - ref_price, 2)
            if comp_price is not None and ref_price is not None
            else None
        )
        if comp_price is None:
            insight = f"{insight} Competitor price could not be verified from snippet."
        elif ref_price is None:
            insight = f"{insight} Price difference unavailable because our reference price has no verified SKU match."

        return CompetitorComparisonRecord(
            our_sku=product.sku,
            our_product_name=our_name,
            our_reference_price=ref_price,
            reference_price_status=ref_status,
            reference_match_confidence=ref_match.match_confidence,
            competitor_product=comp_name,
            marketplace=best_hit.marketplace,
            competitor_price=comp_price,
            currency=currency,
            price_difference=price_diff,
            competitive_insight=insight,
            source_url=best_hit.url,
            research_date=today,
            match_confidence=best_conf,
            status="completed" if comp_price is not None else "price_unavailable",
        )

    def run(
        self,
        vendor_products: Sequence[NormalizedVendorProduct],
        sales_df: pd.DataFrame,
    ) -> CompetitorResearchResult:
        selected = select_representative_products(
            list(vendor_products), count=self.config.competitor_product_count
        )
        sales_catalog = build_sales_price_catalog(sales_df)
        today = date.today().isoformat()

        matching_policy = (
            f"Exact SKU (1.0) -> Normalized Alphanumeric SKU (0.95) -> Fuzzy/Semantic Token Match; "
            f"minimum acceptance threshold = {MIN_REFERENCE_MATCH_CONFIDENCE:.2f}. "
            "Matches below threshold are rejected and reference price is marked Unavailable."
        )

        ref_matches: List[ReferencePriceMatch] = []
        comparisons: List[CompetitorComparisonRecord] = []
        warnings: List[str] = []

        if not self.search_service.is_configured:
            warnings.append(
                f"Search API ({self.config.search_provider}) is not configured (missing SEARCH_API_KEY). "
                "Competitor prices and URLs were not fabricated."
            )
            for product in selected:
                ref_match = match_reference_price(product, sales_catalog)
                ref_matches.append(ref_match)
                comparisons.append(
                    CompetitorComparisonRecord(
                        our_sku=product.sku,
                        our_product_name=build_product_display_name(product),
                        our_reference_price=ref_match.our_reference_price,
                        reference_price_status=(
                            f"Matched ({ref_match.match_method})"
                            if ref_match.accepted
                            else f"Unavailable ({ref_match.reason})"
                        ),
                        reference_match_confidence=ref_match.match_confidence,
                        competitive_insight=(
                            "Competitor research unavailable: SEARCH_API_KEY is not configured."
                        ),
                        research_date=today,
                        match_confidence=0.0,
                        status="unavailable_unconfigured",
                    )
                )
            return CompetitorResearchResult(
                selected_skus=[p.sku for p in selected],
                matching_policy=matching_policy,
                search_provider=self.config.search_provider,
                search_configured=False,
                comparisons=comparisons,
                reference_matches=ref_matches,
                warnings=warnings,
            )

        for product in selected:
            ref_match = match_reference_price(product, sales_catalog)
            ref_matches.append(ref_match)
            query = self.build_search_query(product)
            try:
                hits = self.search_service.search(query, num_results=8)
                comparisons.append(self._select_best_competitor(product, hits, ref_match))
            except Exception as exc:
                logger.error("Competitor search failed for SKU %s: %s", product.sku, exc)
                comparisons.append(
                    CompetitorComparisonRecord(
                        our_sku=product.sku,
                        our_product_name=build_product_display_name(product),
                        our_reference_price=ref_match.our_reference_price,
                        reference_price_status=(
                            f"Matched ({ref_match.match_method})"
                            if ref_match.accepted
                            else f"Unavailable ({ref_match.reason})"
                        ),
                        reference_match_confidence=ref_match.match_confidence,
                        competitive_insight=f"Competitor search failed: {exc}",
                        research_date=today,
                        match_confidence=0.0,
                        status="failed",
                        error=str(exc),
                    )
                )

        return CompetitorResearchResult(
            selected_skus=[p.sku for p in selected],
            matching_policy=matching_policy,
            search_provider=self.config.search_provider,
            search_configured=True,
            comparisons=comparisons,
            reference_matches=ref_matches,
            warnings=warnings,
        )
