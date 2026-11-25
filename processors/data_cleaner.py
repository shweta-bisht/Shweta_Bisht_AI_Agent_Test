"""Automated in-pipeline cleaning and data quality auditing for vendor and sales sheets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from processors.excel_reader import ExcelIngestionResult
from schemas.execution_schema import DataQualityReport
from schemas.product_schema import NormalizedVendorProduct
from utils.logger import get_logger
from utils.validators import is_valid_http_url, normalize_whitespace

logger = get_logger("data_cleaner")

CANONICAL_STATUS_MAP = {
    "processed": "Processed",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "pending": "Pending",
    "oj shipping": "OJ Shipping",
}


@dataclass
class CleanedDataset:
    """Normalized vendor products, cleaned sales DataFrame, and full DataQualityReport."""

    vendor_products: List[NormalizedVendorProduct]
    sales_df: pd.DataFrame
    data_quality_report: DataQualityReport


def _to_python_scalar(val: Any) -> Any:
    """Convert numpy/pandas scalars into native Python primitives for serialization."""
    if val is None or pd.isna(val):
        return None
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return val


def _to_float_or_none(val: Any) -> Optional[float]:
    if val is None or pd.isna(val):
        return None
    try:
        parsed = float(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _extract_dimension_slots(
    row: pd.Series, prefix: str, max_slots: int
) -> List[Dict[str, Any]]:
    """Extract non-zero dimension/weight slots without inventing missing values."""
    slots: List[Dict[str, Any]] = []
    for idx in range(1, max_slots + 1):
        weight = _to_float_or_none(row.get(f"{prefix}Weight{idx}"))
        length = _to_float_or_none(row.get(f"{prefix}Length{idx}"))
        width = _to_float_or_none(row.get(f"{prefix}Width{idx}"))
        height = _to_float_or_none(row.get(f"{prefix}Height{idx}"))
        dim_str = normalize_whitespace(row.get(f"{prefix}Dim{idx}")) if prefix == "" else None

        numeric_vals = [v for v in (weight, length, width, height) if v is not None]
        has_non_zero = any(v > 0 for v in numeric_vals) or bool(dim_str)
        if has_non_zero:
            slot: Dict[str, Any] = {"slot": idx}
            if weight is not None and weight > 0:
                slot["weight"] = weight
            if length is not None and length > 0:
                slot["length"] = length
            if width is not None and width > 0:
                slot["width"] = width
            if height is not None and height > 0:
                slot["height"] = height
            if dim_str:
                slot["dim"] = dim_str
            slots.append(slot)
    return slots


def _format_dimensions_summary(product_slots: List[Dict[str, Any]]) -> Optional[str]:
    if not product_slots:
        return None
    primary = product_slots[0]
    if primary.get("dim"):
        return f"{primary['dim']} (L x W x H, inches)"
    l_val = primary.get("length")
    w_val = primary.get("width")
    h_val = primary.get("height")
    if l_val and w_val and h_val:
        return f"{l_val:g} x {w_val:g} x {h_val:g} in"
    return None


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: re.sub(r"\s+", " ", str(col).strip()) for col in df.columns}
    return df.rename(columns=renamed)


def clean_vendor_data(
    vendor_raw_df: pd.DataFrame,
) -> Tuple[List[NormalizedVendorProduct], Dict[str, Any]]:
    """Clean and normalize vendor product records while preserving original attributes."""
    df = _normalize_column_names(vendor_raw_df.copy())
    bullet_cols = [
        col for col in df.columns if re.fullmatch(r"Bullet\s+\d+", col, flags=re.IGNORECASE)
    ]
    bullet_cols.sort(key=lambda c: int(re.search(r"\d+", c).group(0)))

    products: List[NormalizedVendorProduct] = []
    seen_skus: Dict[str, int] = {}
    duplicate_skus: List[str] = []
    invalid_image_urls: List[Dict[str, str]] = []
    blank_bullets_removed = 0
    duplicate_bullets_removed = 0
    excluded_or_flagged: List[Dict[str, Any]] = []

    missing_values: Dict[str, int] = {}
    core_cols = [
        "SKU",
        "Masked SKU",
        "Vendor1 Name",
        "Category",
        "Color",
        "Material Used",
        "Assembly Required",
        "Brand",
        "Image URL 1",
    ]
    for col in core_cols:
        if col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                missing_values[col] = null_count

    for row_idx, row in df.iterrows():
        raw_attributes = {
            str(k): _to_python_scalar(v) for k, v in vendor_raw_df.iloc[row_idx].items()
        }
        notes: List[str] = []

        sku = normalize_whitespace(row.get("SKU"))
        if not sku:
            fallback_sku = f"MISSING_SKU_ROW_{row_idx + 1}"
            notes.append("Missing required SKU; flagged without discarding.")
            excluded_or_flagged.append(
                {
                    "sheet": "vendor data",
                    "row_index": int(row_idx),
                    "identifier": fallback_sku,
                    "reason": "Missing SKU",
                    "action": "Flagged and retained with synthetic identifier",
                }
            )
            sku = fallback_sku

        if sku in seen_skus:
            duplicate_skus.append(sku)
            notes.append(f"Duplicate SKU (first seen at row {seen_skus[sku] + 1}).")
            excluded_or_flagged.append(
                {
                    "sheet": "vendor data",
                    "row_index": int(row_idx),
                    "identifier": sku,
                    "reason": "Duplicate vendor SKU",
                    "action": "Flagged for audit; retained in normalized list",
                }
            )
        else:
            seen_skus[sku] = int(row_idx)

        # Clean bullets: strip whitespace, ignore blanks, deduplicate preserving order
        bullets: List[str] = []
        seen_bullets = set()
        for bcol in bullet_cols:
            cleaned_bullet = normalize_whitespace(row.get(bcol))
            if not cleaned_bullet:
                blank_bullets_removed += 1
                continue
            dedup_key = cleaned_bullet.lower()
            if dedup_key in seen_bullets:
                duplicate_bullets_removed += 1
                notes.append(f"Removed duplicate bullet from {bcol}.")
                continue
            seen_bullets.add(dedup_key)
            bullets.append(cleaned_bullet)

        image_url = normalize_whitespace(row.get("Image URL 1"))
        image_valid = is_valid_http_url(image_url)
        if not image_valid:
            invalid_image_urls.append({"sku": sku, "image_url": image_url or ""})
            notes.append("Invalid or missing Image URL 1.")

        product_dims = _extract_dimension_slots(row, prefix="", max_slots=7)
        shipping_dims = _extract_dimension_slots(row, prefix="Ship ", max_slots=8)
        primary_weight = product_dims[0].get("weight") if product_dims else None
        dim_summary = _format_dimensions_summary(product_dims)

        products.append(
            NormalizedVendorProduct(
                sku=sku,
                masked_sku=normalize_whitespace(row.get("Masked SKU")),
                vendor_name=normalize_whitespace(row.get("Vendor1 Name")),
                brand=normalize_whitespace(row.get("Brand")),
                category=normalize_whitespace(row.get("Category")),
                color=normalize_whitespace(row.get("Color")),
                material_used=normalize_whitespace(row.get("Material Used")),
                assembly_required=normalize_whitespace(row.get("Assembly Required")),
                image_url=image_url,
                image_url_valid=image_valid,
                bullets=bullets,
                dimensions_summary=dim_summary,
                primary_weight=primary_weight,
                product_dimensions=product_dims,
                shipping_dimensions=shipping_dims,
                raw_attributes=raw_attributes,
                cleaning_notes=notes,
            )
        )

    stats = {
        "original_records": len(vendor_raw_df),
        "processed_records": len(products),
        "duplicate_skus": sorted(set(duplicate_skus)),
        "missing_values": missing_values,
        "invalid_image_urls": invalid_image_urls,
        "blank_bullets_removed": blank_bullets_removed,
        "duplicate_bullets_removed": duplicate_bullets_removed,
        "flagged": excluded_or_flagged,
    }
    return products, stats


def clean_sales_data(sales_raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate sales records while preserving original records and audit flags."""
    df = _normalize_column_names(sales_raw_df.copy())
    df["_original_row_index"] = df.index.astype(int)

    # Preserve original status and carrier before normalization
    df["orderStatus_raw"] = df["orderStatus"].astype(str)
    df["shippingCarrier_raw"] = df["shippingCarrier"].astype(str)

    # Strip whitespace across text columns
    text_cols = [
        "itemName",
        "Masked SKU",
        "orderStatus",
        "shippingCarrier",
        "Vendor",
        "vendorType",
        "Customer Name",
        "customerReferenceNumber",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_whitespace)

    # Normalize orderStatus consistently
    def _normalize_status(val: Optional[str]) -> str:
        if not val:
            return "Unknown"
        return CANONICAL_STATUS_MAP.get(val.lower(), val)

    df["orderStatus"] = df["orderStatus"].apply(_normalize_status)
    df["shippingCarrier"] = df["shippingCarrier"].fillna("Carrier Not Mapped")

    # Parse dates and numeric columns
    df["orderDate_parsed"] = pd.to_datetime(df["orderDate"], errors="coerce")
    invalid_dates = int(df["orderDate_parsed"].isna().sum())

    for col in ("quantity", "price", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_values: Dict[str, int] = {}
    for col in df.columns:
        if col.startswith("_") or col.endswith("_raw") or col == "orderDate_parsed":
            continue
        null_cnt = int(df[col].isna().sum())
        if null_cnt > 0:
            missing_values[col] = null_cnt

    # Numeric and calculation validations
    df["calc_expected_amount"] = (df["quantity"] * df["price"]).round(2)
    df["calc_amount_diff"] = (df["amount"] - df["calc_expected_amount"]).abs()
    df["flag_invalid_date"] = df["orderDate_parsed"].isna()
    df["flag_non_positive_qty"] = df["quantity"].isna() | (df["quantity"] <= 0)
    df["flag_non_positive_price"] = df["price"].isna() | (df["price"] <= 0)
    df["flag_amount_mismatch"] = df["calc_amount_diff"].isna() | (df["calc_amount_diff"] > 0.01)
    df["flag_unmapped_carrier"] = df["shippingCarrier"].str.lower().isin(
        {"carrier not mapped", "unknown", "unmapped", ""}
    )

    exact_dup_count = int(sales_raw_df.duplicated().sum())
    order_line_dup_mask = df.duplicated(subset=["orderId", "itemName", "Masked SKU"], keep=False)
    order_line_dup_count = int(df.duplicated(subset=["orderId", "itemName", "Masked SKU"]).sum())
    df["flag_duplicate_order_line"] = order_line_dup_mask

    # Capture notable flagged record summaries (without discarding any row)
    flagged: List[Dict[str, Any]] = []
    if invalid_dates > 0:
        flagged.append(
            {
                "sheet": "sales data",
                "reason": "Unparseable orderDate",
                "count": invalid_dates,
                "action": "Retained in dataset and flagged via flag_invalid_date",
            }
        )
    non_pos_qty = int(df["flag_non_positive_qty"].sum())
    if non_pos_qty > 0:
        flagged.append(
            {
                "sheet": "sales data",
                "reason": "Non-positive quantity (<= 0)",
                "count": non_pos_qty,
                "action": "Retained for audit; excluded from confirmed sales if status is Cancelled/Pending",
            }
        )
    non_pos_price = int(df["flag_non_positive_price"].sum())
    if non_pos_price > 0:
        flagged.append(
            {
                "sheet": "sales data",
                "reason": "Non-positive unit price (<= 0)",
                "count": non_pos_price,
                "action": "Retained and flagged via flag_non_positive_price (zero-revenue/promo lines)",
            }
        )
    amt_mismatches = int(df["flag_amount_mismatch"].sum())
    if amt_mismatches > 0:
        flagged.append(
            {
                "sheet": "sales data",
                "reason": "Amount != quantity * price (> 0.01)",
                "count": amt_mismatches,
                "action": "Retained and flagged via flag_amount_mismatch",
            }
        )
    unmapped_carriers = int(df["flag_unmapped_carrier"].sum())
    if unmapped_carriers > 0:
        flagged.append(
            {
                "sheet": "sales data",
                "reason": "Unmapped shipping carrier",
                "count": unmapped_carriers,
                "action": "Retained and reported explicitly in carrier distribution",
            }
        )

    valid_dates = df["orderDate_parsed"].dropna()
    date_range = {
        "min_date": valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        "max_date": valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None,
    }

    status_dist = {
        str(k): int(v) for k, v in df["orderStatus"].value_counts(dropna=False).to_dict().items()
    }
    carrier_dist = {
        str(k): int(v) for k, v in df["shippingCarrier"].value_counts(dropna=False).to_dict().items()
    }

    stats = {
        "original_records": len(sales_raw_df),
        "processed_records": len(df),
        "missing_values": missing_values,
        "invalid_dates": invalid_dates,
        "non_positive_quantities": non_pos_qty,
        "non_positive_prices": non_pos_price,
        "amount_mismatches": amt_mismatches,
        "exact_duplicate_rows": exact_dup_count,
        "duplicate_order_lines": order_line_dup_count,
        "unmapped_carriers": unmapped_carriers,
        "date_range": date_range,
        "status_distribution": status_dist,
        "carrier_distribution": carrier_dist,
        "flagged": flagged,
    }
    return df, stats


def clean_workbook_data(ingestion: ExcelIngestionResult) -> CleanedDataset:
    """Run vendor and sales cleaning and assemble the consolidated DataQualityReport."""
    vendor_products, v_stats = clean_vendor_data(ingestion.vendor_raw_df)
    sales_df, s_stats = clean_sales_data(ingestion.sales_raw_df)

    # Check SKU overlap across both sheets
    vendor_skus = {p.sku for p in vendor_products} | {
        p.masked_sku for p in vendor_products if p.masked_sku
    }
    sales_skus = set(sales_df["Masked SKU"].dropna().astype(str)) | set(
        sales_df["itemName"].dropna().astype(str)
    )
    overlap = vendor_skus.intersection(sales_skus)

    overlap_note = (
        f"Exact SKU overlap count: {len(overlap)}. "
        + (
            "Vendor SKUs and sales Masked SKUs are completely disjoint SKU systems; "
            "reference pricing requires explicit fuzzy/semantic matching with confidence thresholds."
            if len(overlap) == 0
            else f"Found {len(overlap)} overlapping SKU(s)."
        )
    )

    cleaning_operations = [
        "Normalized column names by trimming leading/trailing and repeated whitespace.",
        "Stripped whitespace across all vendor and sales string attributes.",
        "Removed blank vendor bullet slots (Bullet 1..20) and deduplicated repeated bullets per SKU.",
        "Parsed non-zero product and shipping dimension slots while keeping missing optional values as null.",
        "Validated vendor Image URL 1 syntax (HTTP/HTTPS with valid network location).",
        "Preserved original raw attributes for every vendor SKU and original row indices/values for sales.",
        "Parsed sales orderDate timestamps and validated numeric quantity, price, and amount columns.",
        "Verified amount == round(quantity * price, 2) within 0.01 tolerance across all sales records.",
        "Normalized orderStatus values and flagged unmapped shipping carriers without discarding records.",
        "Audited exact duplicate rows, duplicate order lines, and cross-sheet SKU overlap.",
    ]

    report = DataQualityReport(
        workbook_path=ingestion.validation.workbook_path,
        vendor_original_records=v_stats["original_records"],
        vendor_processed_records=v_stats["processed_records"],
        vendor_duplicate_skus=v_stats["duplicate_skus"],
        vendor_missing_values=v_stats["missing_values"],
        vendor_invalid_image_urls=v_stats["invalid_image_urls"],
        vendor_blank_bullets_removed=v_stats["blank_bullets_removed"],
        vendor_duplicate_bullets_removed=v_stats["duplicate_bullets_removed"],
        sales_original_records=s_stats["original_records"],
        sales_processed_records=s_stats["processed_records"],
        sales_missing_values=s_stats["missing_values"],
        sales_invalid_dates=s_stats["invalid_dates"],
        sales_non_positive_quantities=s_stats["non_positive_quantities"],
        sales_non_positive_prices=s_stats["non_positive_prices"],
        sales_amount_mismatches=s_stats["amount_mismatches"],
        sales_exact_duplicate_rows=s_stats["exact_duplicate_rows"],
        sales_duplicate_order_lines=s_stats["duplicate_order_lines"],
        sales_unmapped_carriers=s_stats["unmapped_carriers"],
        sales_date_range=s_stats["date_range"],
        status_distribution=s_stats["status_distribution"],
        carrier_distribution=s_stats["carrier_distribution"],
        cleaning_operations=cleaning_operations,
        excluded_or_flagged_records=v_stats["flagged"] + s_stats["flagged"],
        sku_overlap_count=len(overlap),
        sku_overlap_note=overlap_note,
    )

    logger.info(
        "Cleaned %d vendor SKUs (removed %d blank bullets, %d dup bullets) and %d sales rows.",
        len(vendor_products),
        v_stats["blank_bullets_removed"],
        v_stats["duplicate_bullets_removed"],
        len(sales_df),
    )
    return CleanedDataset(
        vendor_products=vendor_products,
        sales_df=sales_df,
        data_quality_report=report,
    )
