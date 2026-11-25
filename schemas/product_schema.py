"""Pydantic v2 schemas for vendor products and Gemini-generated product content."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ProductAttributes(BaseModel):
    """Normalized product attributes strictly grounded in vendor data."""

    colour: Optional[str] = Field(default=None, description="Product colour(s) from vendor sheet")
    material: Optional[str] = Field(default=None, description="Material used from vendor sheet")
    assembly: Optional[str] = Field(default=None, description="Assembly requirement from vendor sheet")
    dimensions: Optional[str] = Field(
        default=None, description="Primary product dimensions string if available"
    )
    weight: Optional[str] = Field(default=None, description="Primary product weight if available")
    brand: Optional[str] = Field(default=None, description="Brand name from vendor sheet")
    category: Optional[str] = Field(default=None, description="Product category from vendor sheet")


class ProductContentLLMResponse(BaseModel):
    """Schema requested directly from Gemini structured output."""

    title: str = Field(..., min_length=5, description="Concise e-commerce product title")
    description: str = Field(
        ..., min_length=20, description="Factual product description grounded in vendor data"
    )
    features: List[str] = Field(
        ..., min_length=5, max_length=5, description="Exactly five distinct factual key features"
    )
    attributes: ProductAttributes

    @field_validator("title", "description")
    @classmethod
    def _must_not_be_placeholder(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank.")
        lowered = cleaned.lower()
        for banned in ("lorem ipsum", "tbd", "placeholder", "[insert", "n/a - placeholder"):
            if banned in lowered:
                raise ValueError(f"Placeholder text detected: '{banned}'.")
        return cleaned

    @field_validator("features")
    @classmethod
    def _validate_five_unique_features(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if len(cleaned) != 5:
            raise ValueError(f"Expected exactly 5 non-empty features, got {len(cleaned)}.")
        seen = set()
        for feat in cleaned:
            key = feat.lower()
            if key in seen:
                raise ValueError(f"Duplicate feature detected: '{feat}'.")
            seen.add(key)
        return cleaned


class ProductContentItem(BaseModel):
    """Complete product content record including provenance and API metrics."""

    sku: str
    masked_sku: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    title: str
    description: str
    features: List[str] = Field(min_length=5, max_length=5)
    attributes: ProductAttributes
    source_bullets_count: int = 0
    status: str = "completed"
    error: Optional[str] = None
    from_cache: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0


class NormalizedVendorProduct(BaseModel):
    """Normalized representation of one vendor SKU row preserving original attributes."""

    sku: str
    masked_sku: Optional[str] = None
    vendor_name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    color: Optional[str] = None
    material_used: Optional[str] = None
    assembly_required: Optional[str] = None
    image_url: Optional[str] = None
    image_url_valid: bool = False
    bullets: List[str] = Field(default_factory=list)
    dimensions_summary: Optional[str] = None
    primary_weight: Optional[float] = None
    product_dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    shipping_dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    raw_attributes: Dict[str, Any] = Field(default_factory=dict)
    cleaning_notes: List[str] = Field(default_factory=list)
