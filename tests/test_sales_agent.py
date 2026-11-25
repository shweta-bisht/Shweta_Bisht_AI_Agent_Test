"""Tests for deterministic sales analysis and chart/JSON outputs."""

from __future__ import annotations

from pathlib import Path

from agents.sales_agent import SalesAnalysisAgent
from config import AppConfig
from processors.data_cleaner import clean_workbook_data
from processors.excel_reader import read_workbook


def test_sales_analysis_status_treatment_and_kpis(sample_workbook: Path, tmp_path: Path) -> None:
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    cfg = AppConfig(input_file=sample_workbook, output_dir=tmp_path, dry_run=True)
    agent = SalesAnalysisAgent(cfg)

    res = agent.run(cleaned.sales_df, vendor_products=cleaned.vendor_products, output_dir=tmp_path)
    kpis = res["kpis"]

    # Confirmed (Processed) rows: 300 + 200 + 450 = 950; units: 2 + 1 + 3 = 6; distinct orders = 3
    assert kpis["total_confirmed_sales"] == 950.0
    assert kpis["total_units_sold"] == 6
    assert kpis["distinct_confirmed_orders"] == 3
    assert kpis["average_order_value"] == round(950.0 / 3.0, 2)

    # OJ Shipping must be reported separately, not included in confirmed sales
    oj = res["oj_shipping_summary"]
    assert oj["order_lines"] == 1
    assert oj["potential_revenue"] == 120.0

    # Top SKU by revenue is SL-9001 ($750, 5 units)
    assert res["top_10_by_revenue"][0]["sku"] == "SL-9001"
    assert res["top_10_by_revenue"][0]["revenue"] == 750.0

    # Non-moving sales SKUs (only in Cancelled/OJ/Pending): SL-9003, SL-9004, SL-9005
    assert res["non_moving_skus"]["sales_catalog_non_moving_count"] == 3
    assert len(res["business_insights"]) >= 3

    # Files persisted
    assert (tmp_path / "sales_summary.json").exists()
    assert (tmp_path / "sales_dashboard.png").exists()
