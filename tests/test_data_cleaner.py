"""Tests for automated vendor and sales data cleaning and quality auditing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from processors.data_cleaner import clean_workbook_data
from processors.excel_reader import read_workbook


def test_clean_workbook_handles_bullets_whitespace_and_optional_attrs(sample_workbook: Path) -> None:
    ingestion = read_workbook(sample_workbook)
    cleaned = clean_workbook_data(ingestion)

    assert len(cleaned.vendor_products) == 3
    p1 = cleaned.vendor_products[0]
    assert p1.color == "Black"
    # Bullet 2 & 3 were duplicates; Bullet 4 was None
    assert p1.bullets == [
        "One power reclining chair with cup holders",
        "Solid wood frame for durability",
    ]
    assert cleaned.data_quality_report.vendor_duplicate_bullets_removed >= 1
    assert cleaned.data_quality_report.vendor_blank_bullets_removed >= 1

    # Missing optional attribute stays None (never fabricated)
    p3 = cleaned.vendor_products[2]
    assert p3.color is None


def test_clean_workbook_flags_invalid_image_url_and_sales_anomalies(
    sample_workbook: Path, tmp_path: Path
) -> None:
    v_df = pd.read_excel(sample_workbook, sheet_name="vendor data")
    s_df = pd.read_excel(sample_workbook, sheet_name="sales data")
    v_df.loc[0, "Image URL 1"] = "not-a-valid-url"
    # Introduce an amount mismatch on first row
    s_df["amount"] = s_df["amount"].astype(float)
    s_df.loc[0, "amount"] = 999.99

    bad_path = tmp_path / "anomalies.xlsx"
    with pd.ExcelWriter(bad_path, engine="openpyxl") as writer:
        v_df.to_excel(writer, sheet_name="vendor data", index=False)
        s_df.to_excel(writer, sheet_name="sales data", index=False)

    cleaned = clean_workbook_data(read_workbook(bad_path))
    dq = cleaned.data_quality_report

    assert cleaned.vendor_products[0].image_url_valid is False
    assert len(dq.vendor_invalid_image_urls) == 1
    assert dq.sales_amount_mismatches == 1
    assert dq.sales_unmapped_carriers == 1
    assert dq.sales_non_positive_quantities == 1
    # Ensure zero silent record drops
    assert dq.sales_processed_records == dq.sales_original_records == 6
    assert dq.sku_overlap_count == 0
