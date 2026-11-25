"""Tests for Excel workbook ingestion, sheet/column validation, and configuration validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from processors.excel_reader import ExcelIngestionError, read_workbook


def test_read_workbook_success(sample_workbook: Path) -> None:
    res = read_workbook(sample_workbook)
    assert res.validation.vendor_row_count == 3
    assert res.validation.sales_row_count == 6
    assert res.validation.duplicate_vendor_skus == []


def test_missing_workbook_raises(tmp_path: Path) -> None:
    with pytest.raises(ExcelIngestionError, match="does not exist"):
        read_workbook(tmp_path / "nonexistent.xlsx")


def test_missing_sheet_raises(tmp_path: Path) -> None:
    path = tmp_path / "missing_sheet.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([{"SKU": "A"}]).to_excel(writer, sheet_name="vendor data", index=False)

    with pytest.raises(ExcelIngestionError, match="missing required sheet"):
        read_workbook(path)


def test_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "missing_col.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Missing 'Brand', 'Color', etc.
        pd.DataFrame([{"SKU": "A", "Category": "Furniture"}]).to_excel(
            writer, sheet_name="vendor data", index=False
        )
        pd.DataFrame(
            [
                {
                    "orderDate": "2025-07-01",
                    "orderId": 1,
                    "itemName": "X",
                    "Masked SKU": "X",
                    "quantity": 1,
                    "price": 10,
                    "amount": 10,
                    "orderStatus": "Processed",
                    "shippingCarrier": "FEDX",
                    "Vendor": "WareHouse",
                    "vendorType": "Sup",
                }
            ]
        ).to_excel(writer, sheet_name="sales data", index=False)

    with pytest.raises(ExcelIngestionError, match="missing required column"):
        read_workbook(path)


def test_duplicate_vendor_sku_detected(sample_workbook: Path, tmp_path: Path) -> None:
    v_df = pd.read_excel(sample_workbook, sheet_name="vendor data")
    s_df = pd.read_excel(sample_workbook, sheet_name="sales data")
    v_df = pd.concat([v_df, v_df.iloc[[0]]], ignore_index=True)

    dup_path = tmp_path / "dup_sku.xlsx"
    with pd.ExcelWriter(dup_path, engine="openpyxl") as writer:
        v_df.to_excel(writer, sheet_name="vendor data", index=False)
        s_df.to_excel(writer, sheet_name="sales data", index=False)

    res = read_workbook(dup_path)
    assert res.validation.duplicate_vendor_skus == ["SKU-101"]


def test_config_validation_missing_gemini_key(sample_workbook: Path) -> None:
    cfg = AppConfig(input_file=sample_workbook, gemini_api_key=None, dry_run=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        cfg.validate()


def test_config_validation_dry_run_allows_missing_keys(sample_workbook: Path) -> None:
    cfg = AppConfig(input_file=sample_workbook, gemini_api_key=None, search_api_key=None, dry_run=True)
    warnings = cfg.validate()
    assert len(warnings) == 2
