"""Automated Excel ingestion and structural validation using Pandas and OpenPyXL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("excel_reader")

REQUIRED_VENDOR_COLUMNS = [
    "SKU",
    "Category",
    "Color",
    "Material Used",
    "Assembly Required",
    "Brand",
    "Bullet 1",
    "Image URL 1",
]

OPTIONAL_VENDOR_COLUMNS = [
    "Masked SKU",
    "Vendor1 Name",
]

REQUIRED_SALES_COLUMNS = [
    "orderDate",
    "orderId",
    "itemName",
    "Masked SKU",
    "quantity",
    "price",
    "amount",
    "orderStatus",
    "shippingCarrier",
    "Vendor",
    "vendorType",
]


class ExcelIngestionError(Exception):
    """Raised when the input Excel workbook fails structural validation."""


@dataclass
class IngestionValidationSummary:
    """Validation findings collected during Excel ingestion."""

    workbook_path: str
    sheet_names: List[str]
    vendor_sheet_name: str
    sales_sheet_name: str
    vendor_row_count: int
    vendor_col_count: int
    sales_row_count: int
    sales_col_count: int
    duplicate_vendor_skus: List[str] = field(default_factory=list)
    missing_required_vendor_fields: Dict[str, int] = field(default_factory=dict)
    missing_required_sales_fields: Dict[str, int] = field(default_factory=dict)
    type_warnings: List[str] = field(default_factory=list)


@dataclass
class ExcelIngestionResult:
    """Structured output returned by `read_workbook`."""

    vendor_raw_df: pd.DataFrame
    sales_raw_df: pd.DataFrame
    validation: IngestionValidationSummary


def _resolve_sheet_name(available_sheets: List[str], target: str) -> Optional[str]:
    normalized_target = target.strip().lower()
    for sheet in available_sheets:
        if sheet.strip().lower() == normalized_target:
            return sheet
    return None


def _build_column_lookup(columns: List[object]) -> Dict[str, str]:
    """Map normalized lowercase column header -> actual column header."""
    lookup: Dict[str, str] = {}
    for col in columns:
        col_str = str(col)
        lookup[col_str.strip().lower()] = col_str
    return lookup


def _standardize_required_headers(df: pd.DataFrame, expected_cols: List[str], sheet_label: str) -> pd.DataFrame:
    """Validate presence of required columns (case-insensitive/trimmed) and align header names."""
    lookup = _build_column_lookup(list(df.columns))
    missing: List[str] = []
    rename_map: Dict[str, str] = {}

    for req in expected_cols:
        key = req.strip().lower()
        if key not in lookup:
            missing.append(req)
        else:
            actual = lookup[key]
            if actual != req:
                rename_map[actual] = req

    if missing:
        raise ExcelIngestionError(
            f"Sheet '{sheet_label}' is missing required column(s): {', '.join(missing)}."
        )

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def read_workbook(workbook_path: str | Path) -> ExcelIngestionResult:
    """Read and validate the vendor and sales sheets from an Excel workbook."""
    path = Path(workbook_path)
    if not path.exists():
        raise ExcelIngestionError(f"Workbook file does not exist: '{path}'.")
    if not path.is_file():
        raise ExcelIngestionError(f"Workbook path is not a file: '{path}'.")

    logger.info("Reading Excel workbook: %s", path)
    try:
        excel_file = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:
        raise ExcelIngestionError(f"Failed to open workbook '{path}': {exc}") from exc

    sheet_names = list(excel_file.sheet_names)
    vendor_sheet = _resolve_sheet_name(sheet_names, "vendor data")
    sales_sheet = _resolve_sheet_name(sheet_names, "sales data")

    missing_sheets = []
    if not vendor_sheet:
        missing_sheets.append("vendor data")
    if not sales_sheet:
        missing_sheets.append("sales data")
    if missing_sheets:
        raise ExcelIngestionError(
            f"Workbook '{path}' is missing required sheet(s): {', '.join(missing_sheets)}. "
            f"Found sheets: {sheet_names}."
        )

    vendor_df = pd.read_excel(excel_file, sheet_name=vendor_sheet)
    sales_df = pd.read_excel(excel_file, sheet_name=sales_sheet)

    if vendor_df.empty:
        raise ExcelIngestionError(f"Sheet '{vendor_sheet}' contains no rows.")
    if sales_df.empty:
        raise ExcelIngestionError(f"Sheet '{sales_sheet}' contains no rows.")

    vendor_df = _standardize_required_headers(
        vendor_df, REQUIRED_VENDOR_COLUMNS, vendor_sheet
    )
    sales_df = _standardize_required_headers(
        sales_df, REQUIRED_SALES_COLUMNS, sales_sheet
    )

    # Also align optional known columns if whitespace differs
    vendor_lookup = _build_column_lookup(list(vendor_df.columns))
    for opt in OPTIONAL_VENDOR_COLUMNS:
        key = opt.strip().lower()
        if key in vendor_lookup and vendor_lookup[key] != opt:
            vendor_df = vendor_df.rename(columns={vendor_lookup[key]: opt})

    # Detect missing required fields
    missing_vendor_fields: Dict[str, int] = {}
    for col in REQUIRED_VENDOR_COLUMNS:
        null_cnt = int(vendor_df[col].isna().sum())
        if null_cnt > 0:
            missing_vendor_fields[col] = null_cnt

    missing_sales_fields: Dict[str, int] = {}
    for col in REQUIRED_SALES_COLUMNS:
        null_cnt = int(sales_df[col].isna().sum())
        if null_cnt > 0:
            missing_sales_fields[col] = null_cnt

    # Detect duplicate vendor SKUs
    sku_series = vendor_df["SKU"].dropna().astype(str).str.strip()
    dup_skus = sorted(sku_series[sku_series.duplicated(keep=False)].unique().tolist())

    # Validate data types
    type_warnings: List[str] = []
    parsed_dates = pd.to_datetime(sales_df["orderDate"], errors="coerce")
    bad_dates = int(parsed_dates.isna().sum() - sales_df["orderDate"].isna().sum())
    if bad_dates > 0:
        type_warnings.append(f"sales.orderDate has {bad_dates} unparseable date value(s).")

    for num_col in ("quantity", "price", "amount"):
        coerced = pd.to_numeric(sales_df[num_col], errors="coerce")
        bad_nums = int(coerced.isna().sum() - sales_df[num_col].isna().sum())
        if bad_nums > 0:
            type_warnings.append(f"sales.{num_col} has {bad_nums} non-numeric value(s).")

    summary = IngestionValidationSummary(
        workbook_path=str(path),
        sheet_names=sheet_names,
        vendor_sheet_name=vendor_sheet,
        sales_sheet_name=sales_sheet,
        vendor_row_count=len(vendor_df),
        vendor_col_count=len(vendor_df.columns),
        sales_row_count=len(sales_df),
        sales_col_count=len(sales_df.columns),
        duplicate_vendor_skus=dup_skus,
        missing_required_vendor_fields=missing_vendor_fields,
        missing_required_sales_fields=missing_sales_fields,
        type_warnings=type_warnings,
    )

    logger.info(
        "Ingested '%s': %d vendor rows (%d cols), %d sales rows (%d cols).",
        path.name,
        summary.vendor_row_count,
        summary.vendor_col_count,
        summary.sales_row_count,
        summary.sales_col_count,
    )
    if dup_skus:
        logger.warning("Duplicate vendor SKUs detected: %s", dup_skus)
    for warn in type_warnings:
        logger.warning("Type validation warning: %s", warn)

    return ExcelIngestionResult(
        vendor_raw_df=vendor_df,
        sales_raw_df=sales_df,
        validation=summary,
    )
