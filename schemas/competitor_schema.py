"""Pydantic v2 schemas for competitor analysis and reference price matching."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ReferencePriceMatch(BaseModel):
    """Result of attempting to match a vendor product to sales sheet pricing."""

    our_sku: str
    matched_sales_sku: Optional[str] = None
    matched_item_name: Optional[str] = None
    our_reference_price: Optional[float] = None
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    match_method: str = "none"
    accepted: bool = False
    reason: str


class CompetitorComparisonRecord(BaseModel):
    """Single competitor comparison row for a selected vendor SKU."""

    our_sku: str
    our_product_name: str
    our_reference_price: Optional[float] = None
    reference_price_status: str = "Unavailable"
    reference_match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    competitor_product: Optional[str] = None
    marketplace: Optional[str] = None
    competitor_price: Optional[float] = None
    currency: Optional[str] = None
    price_difference: Optional[float] = None
    competitive_insight: str
    source_url: Optional[str] = None
    research_date: str
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "completed"
    error: Optional[str] = None


class CompetitorResearchResult(BaseModel):
    """Container for all competitor comparisons and matching metadata."""

    selected_skus: List[str] = Field(default_factory=list)
    matching_policy: str
    search_provider: str
    search_configured: bool
    comparisons: List[CompetitorComparisonRecord] = Field(default_factory=list)
    reference_matches: List[ReferencePriceMatch] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
