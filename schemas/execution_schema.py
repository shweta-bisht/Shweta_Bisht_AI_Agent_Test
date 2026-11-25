"""Pydantic v2 schemas for data quality, image generation, and execution reporting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataQualityReport(BaseModel):
    """Comprehensive audit trail of Excel ingestion and cleaning operations."""

    workbook_path: str
    vendor_original_records: int = 0
    vendor_processed_records: int = 0
    vendor_duplicate_skus: List[str] = Field(default_factory=list)
    vendor_missing_values: Dict[str, int] = Field(default_factory=dict)
    vendor_invalid_image_urls: List[Dict[str, str]] = Field(default_factory=list)
    vendor_blank_bullets_removed: int = 0
    vendor_duplicate_bullets_removed: int = 0

    sales_original_records: int = 0
    sales_processed_records: int = 0
    sales_missing_values: Dict[str, int] = Field(default_factory=dict)
    sales_invalid_dates: int = 0
    sales_non_positive_quantities: int = 0
    sales_non_positive_prices: int = 0
    sales_amount_mismatches: int = 0
    sales_exact_duplicate_rows: int = 0
    sales_duplicate_order_lines: int = 0
    sales_unmapped_carriers: int = 0
    sales_date_range: Dict[str, Optional[str]] = Field(default_factory=dict)

    status_distribution: Dict[str, int] = Field(default_factory=dict)
    carrier_distribution: Dict[str, int] = Field(default_factory=dict)
    cleaning_operations: List[str] = Field(default_factory=list)
    excluded_or_flagged_records: List[Dict[str, Any]] = Field(default_factory=list)
    sku_overlap_count: int = 0
    sku_overlap_note: str = ""


class ImageGenerationRecord(BaseModel):
    """Per-SKU tracking record for original image download and lifestyle generation."""

    sku: str
    image_url: Optional[str] = None
    original_image_path: Optional[str] = None
    lifestyle_image_path: Optional[str] = None
    original_valid: bool = False
    lifestyle_valid: bool = False
    status: str = "pending"
    error: Optional[str] = None
    latency_seconds: float = 0.0
    retries: int = 0


class ExecutionReport(BaseModel):
    """Final run summary written to execution_report.json and final_summary.xlsx."""

    run_id: str
    start_time: str
    end_time: str
    total_duration_seconds: float
    final_status: str
    dry_run: bool = False
    input_workbook: str
    products_total: int = 0
    products_processed: int = 0
    product_content_generated: int = 0
    product_content_failed: int = 0
    images_generated: int = 0
    images_failed: int = 0
    images_skipped: int = 0
    competitor_comparisons_completed: int = 0
    competitor_comparisons_failed: int = 0
    competitor_skipped: bool = False
    sales_records_processed: int = 0
    api_failures: int = 0
    retry_counts: int = 0
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    estimated_api_cost_usd: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    generated_files: List[str] = Field(default_factory=list)
