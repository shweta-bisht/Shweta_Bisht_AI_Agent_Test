"""Shared LangGraph workflow state definition (100% msgpack/JSON checkpoint-serializable)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class CommerceAIState(TypedDict, total=False):
    """Shared serializable state passed across LangGraph nodes."""

    run_id: str
    start_time_iso: str
    input_workbook_path: str

    # Serializable summaries & Pydantic model_dump() payloads
    ingestion_summary: Dict[str, Any]
    vendor_products: List[Dict[str, Any]]
    sales_record_count: int
    data_quality_report: Dict[str, Any]

    # Branch outputs (serializable dicts/lists)
    product_content: List[Dict[str, Any]]
    image_records: List[Dict[str, Any]]
    competitor_result: Dict[str, Any] | None
    sales_analysis: Dict[str, Any]

    # Aggregated diagnostics & outputs
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    generated_files: Annotated[List[str], operator.add]
    fatal_error: bool

    execution_report: Dict[str, Any]
    final_status: str
