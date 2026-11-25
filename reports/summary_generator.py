"""Execution summary report builder and JSON writer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from schemas.competitor_schema import CompetitorResearchResult
from schemas.execution_schema import ExecutionReport, ImageGenerationRecord
from schemas.product_schema import ProductContentItem
from utils.cost_tracker import CostTracker
from utils.file_utils import write_json


def build_execution_report(
    run_id: str,
    start_dt: datetime,
    end_dt: datetime,
    input_workbook: str,
    dry_run: bool,
    products_total: int,
    content_items: Sequence[ProductContentItem],
    image_records: Sequence[ImageGenerationRecord],
    competitor_result: Optional[CompetitorResearchResult],
    sales_records_processed: int,
    cost_tracker: CostTracker,
    errors: Sequence[str],
    warnings: Sequence[str],
    generated_files: Sequence[str],
    images_skipped: bool = False,
    competitor_skipped: bool = False,
) -> ExecutionReport:
    """Build an accurate ExecutionReport reflecting exact completed and failed operations."""
    content_ok = sum(1 for item in content_items if item.status == "completed")
    content_err = sum(1 for item in content_items if item.status != "completed")

    img_ok = sum(1 for rec in image_records if rec.status == "completed" and rec.lifestyle_valid)
    img_err = sum(1 for rec in image_records if rec.status == "failed")
    img_skip_count = products_total if (images_skipped or dry_run) else 0

    comp_ok = 0
    comp_err = 0
    if competitor_result:
        comp_ok = sum(
            1
            for c in competitor_result.comparisons
            if c.status in {"completed", "price_unavailable"}
        )
        comp_err = sum(1 for c in competitor_result.comparisons if c.status == "failed")

    snap = cost_tracker.snapshot()
    total_duration = round((end_dt - start_dt).total_seconds(), 3)

    if errors or content_err > 0 or img_err > 0 or comp_err > 0:
        if sales_records_processed > 0 or content_ok > 0 or img_ok > 0 or comp_ok > 0:
            final_status = "completed_with_partial_failures"
        else:
            final_status = "failed"
    elif dry_run:
        final_status = "dry_run_completed"
    else:
        final_status = "completed"

    return ExecutionReport(
        run_id=run_id,
        start_time=start_dt.astimezone(timezone.utc).isoformat(),
        end_time=end_dt.astimezone(timezone.utc).isoformat(),
        total_duration_seconds=total_duration,
        final_status=final_status,
        dry_run=dry_run,
        input_workbook=input_workbook,
        products_total=products_total,
        products_processed=products_total,
        product_content_generated=content_ok,
        product_content_failed=content_err,
        images_generated=img_ok,
        images_failed=img_err,
        images_skipped=img_skip_count,
        competitor_comparisons_completed=comp_ok,
        competitor_comparisons_failed=comp_err,
        competitor_skipped=competitor_skipped or dry_run,
        sales_records_processed=sales_records_processed,
        api_failures=int(snap["api_failures"]),
        retry_counts=int(snap["retries"]),
        token_usage={
            "prompt_tokens": int(snap["prompt_tokens"]),
            "completion_tokens": int(snap["completion_tokens"]),
            "total_tokens": int(snap["total_tokens"]),
        },
        estimated_api_cost_usd=float(snap["estimated_cost_usd"]),
        errors=list(errors),
        warnings=list(warnings),
        generated_files=list(generated_files),
    )


def write_execution_report(report: ExecutionReport, output_path: Path) -> Path:
    """Persist ExecutionReport to JSON."""
    return write_json(output_path, report.model_dump())
