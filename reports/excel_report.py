"""OpenPyXL report builders for standalone and consolidated Excel deliverables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from schemas.competitor_schema import CompetitorResearchResult
from schemas.execution_schema import DataQualityReport, ExecutionReport, ImageGenerationRecord
from schemas.product_schema import ProductContentItem
from utils.file_utils import ensure_dir
from utils.logger import get_logger

logger = get_logger("excel_report")

HEADER_FILL = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
SECTION_FILL = PatternFill(start_color="DBEAFE", end_color="DBEAFE", fill_type="solid")
SECTION_FONT = Font(color="1E3A8A", bold=True, size=11)
WRAP_ALIGN = Alignment(vertical="top", wrap_text=True)


def _style_sheet_table(ws, headers: List[str], rows: List[List[Any]]) -> None:
    ws.append(headers)
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    for row in rows:
        ws.append(row)

    for r_idx in range(2, len(rows) + 2):
        for c_idx in range(1, len(headers) + 1):
            ws.cell(row=r_idx, column=c_idx).alignment = WRAP_ALIGN

    ws.freeze_panes = "A2"
    _autofit_columns(ws)


def _autofit_columns(ws, min_width: int = 12, max_width: int = 55) -> None:
    for col_cells in ws.columns:
        max_len = 0
        col_idx = col_cells[0].column
        for cell in col_cells:
            val = "" if cell.value is None else str(cell.value)
            first_line = val.splitlines()[0] if val else ""
            if len(first_line) > max_len:
                max_len = len(first_line)
        ws.column_dimensions[get_column_letter(col_idx)].width = max(
            min_width, min(max_width, max_len + 4)
        )


def _content_rows(
    items: Sequence[ProductContentItem],
    image_records: Optional[Sequence[ImageGenerationRecord]] = None,
) -> tuple[List[str], List[List[Any]]]:
    img_map = {rec.sku: rec for rec in (image_records or [])}
    headers = [
        "SKU",
        "Masked SKU",
        "Brand",
        "Category",
        "Title",
        "Description",
        "Feature 1",
        "Feature 2",
        "Feature 3",
        "Feature 4",
        "Feature 5",
        "Colour",
        "Material",
        "Assembly",
        "Dimensions",
        "Weight",
        "Status",
        "From Cache",
        "Lifestyle Image",
        "Error",
    ]
    rows: List[List[Any]] = []
    for item in items:
        feats = list(item.features) + [""] * max(0, 5 - len(item.features))
        img_rec = img_map.get(item.sku)
        rows.append(
            [
                item.sku,
                item.masked_sku or "",
                item.brand or "",
                item.category or "",
                item.title,
                item.description,
                feats[0],
                feats[1],
                feats[2],
                feats[3],
                feats[4],
                item.attributes.colour or "Unavailable",
                item.attributes.material or "Unavailable",
                item.attributes.assembly or "Unavailable",
                item.attributes.dimensions or "Unavailable",
                item.attributes.weight or "Unavailable",
                item.status,
                "Yes" if item.from_cache else "No",
                (img_rec.lifestyle_image_path if img_rec and img_rec.lifestyle_valid else "Unavailable"),
                item.error or "",
            ]
        )
    return headers, rows


def export_product_content_workbook(
    items: Sequence[ProductContentItem],
    output_path: Path,
    image_records: Optional[Sequence[ImageGenerationRecord]] = None,
) -> Path:
    """Write `product_content.xlsx`."""
    ensure_dir(output_path.parent)
    wb = Workbook()
    ws = wb.active
    ws.title = "Product Content"
    headers, rows = _content_rows(items, image_records=image_records)
    _style_sheet_table(ws, headers, rows)
    wb.save(output_path)
    logger.info("Saved %s", output_path)
    return output_path


def _competitor_rows(res: Optional[CompetitorResearchResult]) -> tuple[List[str], List[List[Any]]]:
    headers = [
        "Our SKU",
        "Our Product Name",
        "Our Reference Price",
        "Reference Price Status",
        "Reference Match Confidence",
        "Competitor Product",
        "Marketplace",
        "Competitor Price",
        "Currency",
        "Price Difference",
        "Competitive Insight",
        "Source URL",
        "Research Date",
        "Match Confidence",
        "Status",
        "Error",
    ]
    rows: List[List[Any]] = []
    if res:
        for c in res.comparisons:
            rows.append(
                [
                    c.our_sku,
                    c.our_product_name,
                    c.our_reference_price if c.our_reference_price is not None else "Unavailable",
                    c.reference_price_status,
                    c.reference_match_confidence,
                    c.competitor_product or "Unavailable",
                    c.marketplace or "Unavailable",
                    c.competitor_price if c.competitor_price is not None else "Unavailable",
                    c.currency or "",
                    c.price_difference if c.price_difference is not None else "Unavailable",
                    c.competitive_insight,
                    c.source_url or "",
                    c.research_date,
                    c.match_confidence,
                    c.status,
                    c.error or "",
                ]
            )
    return headers, rows


def export_competitor_analysis_workbook(
    result: CompetitorResearchResult,
    output_path: Path,
) -> Path:
    """Write `competitor_analysis.xlsx` with comparisons and reference-price matching audit."""
    ensure_dir(output_path.parent)
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Competitor Analysis"
    headers, rows = _competitor_rows(result)
    _style_sheet_table(ws1, headers, rows)

    ws2 = wb.create_sheet("Reference Price Matching")
    ref_headers = [
        "Our SKU",
        "Matched Sales SKU",
        "Matched Item Name",
        "Reference Price",
        "Match Confidence",
        "Method",
        "Accepted",
        "Reason",
    ]
    ref_rows = [
        [
            m.our_sku,
            m.matched_sales_sku or "",
            m.matched_item_name or "",
            m.our_reference_price if m.our_reference_price is not None else "Unavailable",
            m.match_confidence,
            m.match_method,
            "Yes" if m.accepted else "No",
            m.reason,
        ]
        for m in result.reference_matches
    ]
    _style_sheet_table(ws2, ref_headers, ref_rows)

    wb.save(output_path)
    logger.info("Saved %s", output_path)
    return output_path


def _populate_sales_summary_sheet(ws, sales_analysis: Dict[str, Any], dashboard_path: Optional[Path] = None) -> None:
    kpis = sales_analysis.get("kpis", {})
    oj = sales_analysis.get("oj_shipping_summary", {})
    insights = sales_analysis.get("business_insights", [])
    statuses = sales_analysis.get("status_breakdown", [])
    monthly = sales_analysis.get("monthly_performance", [])

    rows: List[List[Any]] = [
        ["Metric", "Value", "Notes"],
        ["Total Confirmed Sales ($)", kpis.get("total_confirmed_sales", 0.0), f"Statuses: {', '.join(kpis.get('confirmed_statuses_rule', []))}"],
        ["Total Confirmed Units Sold", kpis.get("total_units_sold", 0), "Confirmed order lines only"],
        ["Distinct Confirmed Orders", kpis.get("distinct_confirmed_orders", 0), "Unique orderId in confirmed statuses"],
        ["Average Order Value ($)", kpis.get("average_order_value", 0.0), "Total Confirmed Sales / Distinct Confirmed Orders"],
        ["Total Order Lines (All Statuses)", kpis.get("total_order_lines", 0), "Before status exclusions"],
        ["OJ Shipping Lines (Separate)", oj.get("order_lines", 0), oj.get("note", "")],
        ["OJ Shipping Potential Revenue ($)", oj.get("potential_revenue", 0.0), "Excluded from confirmed sales until verified"],
    ]
    _style_sheet_table(ws, rows[0], rows[1:])

    # Monthly table below
    ws.append([])
    ws.append(["Monthly Sales Performance"])
    ws.cell(row=ws.max_row, column=1).font = SECTION_FONT
    ws.cell(row=ws.max_row, column=1).fill = SECTION_FILL
    ws.append(["Month", "Confirmed Revenue ($)", "Units Sold", "Distinct Orders", "Order Lines", "AOV ($)"])
    for m in monthly:
        ws.append([m["month"], m["confirmed_revenue"], m["units_sold"], m["distinct_orders"], m["order_lines"], m["average_order_value"]])

    # Status breakdown below
    ws.append([])
    ws.append(["Order Status Treatment"])
    ws.cell(row=ws.max_row, column=1).font = SECTION_FONT
    ws.cell(row=ws.max_row, column=1).fill = SECTION_FILL
    ws.append(["Order Status", "Treatment", "Order Lines", "Distinct Orders", "Units", "Amount ($)"])
    for st in statuses:
        ws.append([st["order_status"], st["treatment"], st["order_lines"], st["distinct_orders"], st["units"], st["amount"]])

    # Insights below
    ws.append([])
    ws.append(["Key Business Insights"])
    ws.cell(row=ws.max_row, column=1).font = SECTION_FONT
    ws.cell(row=ws.max_row, column=1).fill = SECTION_FILL
    for idx, ins in enumerate(insights, start=1):
        ws.append([f"Insight {idx}", ins])

    _autofit_columns(ws)

    if dashboard_path and dashboard_path.exists():
        try:
            img = XLImage(str(dashboard_path))
            img.width = 960
            img.height = 640
            ws.add_image(img, "E2")
        except Exception as exc:
            logger.warning("Could not embed dashboard image in Excel: %s", exc)


def export_sales_analysis_workbook(
    sales_analysis: Dict[str, Any],
    output_path: Path,
    dashboard_path: Optional[Path] = None,
) -> Path:
    """Write `sales_analysis.xlsx` with comprehensive deterministic tables."""
    ensure_dir(output_path.parent)
    wb = Workbook()

    ws_sum = wb.active
    ws_sum.title = "Sales Summary"
    _populate_sales_summary_sheet(ws_sum, sales_analysis, dashboard_path=dashboard_path)

    ws_top_rev = wb.create_sheet("Top 10 SKUs by Revenue")
    _style_sheet_table(
        ws_top_rev,
        ["SKU", "Revenue ($)", "Units Sold", "Distinct Orders", "Avg Unit Price ($)"],
        [
            [r["sku"], r["revenue"], r["units_sold"], r["distinct_orders"], r["avg_unit_price"]]
            for r in sales_analysis.get("top_10_by_revenue", [])
        ],
    )

    ws_top_u = wb.create_sheet("Top 10 SKUs by Units")
    _style_sheet_table(
        ws_top_u,
        ["SKU", "Units Sold", "Revenue ($)", "Distinct Orders", "Avg Unit Price ($)"],
        [
            [r["sku"], r["units_sold"], r["revenue"], r["distinct_orders"], r["avg_unit_price"]]
            for r in sales_analysis.get("top_10_by_units", [])
        ],
    )

    ws_low = wb.create_sheet("Low & Non-Moving SKUs")
    non_moving = sales_analysis.get("non_moving_skus", {})
    low_rows = [
        [r["sku"], r["revenue"], r["units_sold"], r["distinct_orders"], r["avg_unit_price"], "Low-performing confirmed SKU"]
        for r in sales_analysis.get("low_performing_skus", [])
    ]
    for nm_sku in non_moving.get("sales_catalog_non_moving_sample", []):
        low_rows.append([nm_sku, 0.0, 0, 0, 0.0, "Non-moving in Confirmed Sales (Cancelled/Pending/OJ only)"])
    _style_sheet_table(
        ws_low,
        ["SKU", "Revenue ($)", "Units Sold", "Distinct Orders", "Avg Unit Price ($)", "Classification"],
        low_rows,
    )

    ws_vendor = wb.create_sheet("Vendor & Carrier Performance")
    v_rows = [
        [v["vendor"], v["confirmed_revenue"], v["units_sold"], v["distinct_orders"], v["order_lines"], v["revenue_share_pct"]]
        for v in sales_analysis.get("vendor_performance", [])
    ]
    _style_sheet_table(
        ws_vendor,
        ["Vendor Channel", "Confirmed Revenue ($)", "Units Sold", "Distinct Orders", "Order Lines", "Revenue Share (%)"],
        v_rows,
    )
    ws_vendor.append([])
    ws_vendor.append(["Shipping Carrier", "Order Lines", "Line Share (%)", "Distinct Orders", "Units", "Total Amount ($)"])
    for c in sales_analysis.get("carrier_distribution", []):
        ws_vendor.append([c["shipping_carrier"], c["order_lines"], c["line_share_pct"], c["distinct_orders"], c["units"], c["total_amount"]])
    _autofit_columns(ws_vendor)

    wb.save(output_path)
    logger.info("Saved %s", output_path)
    return output_path


def export_final_summary_workbook(
    output_path: Path,
    content_items: Sequence[ProductContentItem],
    image_records: Sequence[ImageGenerationRecord],
    competitor_result: Optional[CompetitorResearchResult],
    sales_analysis: Dict[str, Any],
    data_quality: Optional[DataQualityReport],
    execution_report: ExecutionReport,
    dashboard_path: Optional[Path] = None,
) -> Path:
    """Write `final_summary.xlsx` containing all 7 required assessment sheets."""
    ensure_dir(output_path.parent)
    wb = Workbook()

    # 1. Product Listings
    ws1 = wb.active
    ws1.title = "Product Listings"
    c_headers, c_rows = _content_rows(content_items, image_records=image_records)
    _style_sheet_table(ws1, c_headers, c_rows)

    # 2. Competitor Analysis
    ws2 = wb.create_sheet("Competitor Analysis")
    comp_headers, comp_rows = _competitor_rows(competitor_result)
    _style_sheet_table(ws2, comp_headers, comp_rows)

    # 3. Sales Summary
    ws3 = wb.create_sheet("Sales Summary")
    _populate_sales_summary_sheet(ws3, sales_analysis, dashboard_path=dashboard_path)

    # 4. Top Performing Products
    ws4 = wb.create_sheet("Top Performing Products")
    top_rows = [
        ["By Revenue", r["sku"], r["revenue"], r["units_sold"], r["distinct_orders"], r["avg_unit_price"]]
        for r in sales_analysis.get("top_10_by_revenue", [])
    ] + [
        ["By Units", r["sku"], r["revenue"], r["units_sold"], r["distinct_orders"], r["avg_unit_price"]]
        for r in sales_analysis.get("top_10_by_units", [])
    ]
    _style_sheet_table(
        ws4,
        ["Ranking Basis", "SKU", "Revenue ($)", "Units Sold", "Distinct Orders", "Avg Unit Price ($)"],
        top_rows,
    )

    # 5. Vendor Performance
    ws5 = wb.create_sheet("Vendor Performance")
    v_rows = [
        ["Channel", v["vendor"], v["confirmed_revenue"], v["units_sold"], v["distinct_orders"], v["revenue_share_pct"]]
        for v in sales_analysis.get("vendor_performance", [])
    ] + [
        ["Supplier (vendorType)", vt["vendor_type"], vt["confirmed_revenue"], vt["units_sold"], vt["distinct_orders"], ""]
        for vt in sales_analysis.get("top_vendor_types", [])
    ]
    _style_sheet_table(
        ws5,
        ["Dimension", "Name", "Confirmed Revenue ($)", "Units Sold", "Distinct Orders", "Revenue Share (%)"],
        v_rows,
    )

    # 6. Data Quality
    ws6 = wb.create_sheet("Data Quality")
    dq_rows: List[List[Any]] = []
    if data_quality:
        dq_rows = [
            ["Workbook Path", data_quality.workbook_path],
            ["Vendor Original Records", data_quality.vendor_original_records],
            ["Vendor Processed Records", data_quality.vendor_processed_records],
            ["Vendor Duplicate SKUs", ", ".join(data_quality.vendor_duplicate_skus) or "None"],
            ["Vendor Blank Bullets Removed", data_quality.vendor_blank_bullets_removed],
            ["Vendor Duplicate Bullets Removed", data_quality.vendor_duplicate_bullets_removed],
            ["Vendor Invalid Image URLs", len(data_quality.vendor_invalid_image_urls)],
            ["Sales Original Records", data_quality.sales_original_records],
            ["Sales Processed Records", data_quality.sales_processed_records],
            ["Sales Invalid Dates", data_quality.sales_invalid_dates],
            ["Sales Non-Positive Quantities", data_quality.sales_non_positive_quantities],
            ["Sales Non-Positive Prices", data_quality.sales_non_positive_prices],
            ["Sales Amount Mismatches", data_quality.sales_amount_mismatches],
            ["Sales Exact Duplicate Rows", data_quality.sales_exact_duplicate_rows],
            ["Sales Duplicate Order Lines", data_quality.sales_duplicate_order_lines],
            ["Sales Unmapped Carriers", data_quality.sales_unmapped_carriers],
            ["Sales Date Range", f"{data_quality.sales_date_range.get('min_date')} to {data_quality.sales_date_range.get('max_date')}"],
            ["Cross-Sheet SKU Overlap", data_quality.sku_overlap_note],
        ]
        for op in data_quality.cleaning_operations:
            dq_rows.append(["Cleaning Operation", op])
    _style_sheet_table(ws6, ["Audit Metric", "Detail"], dq_rows)

    # 7. Execution Summary
    ws7 = wb.create_sheet("Execution Summary")
    exec_rows = [
        ["Run ID", execution_report.run_id],
        ["Start Time", execution_report.start_time],
        ["End Time", execution_report.end_time],
        ["Total Duration (s)", execution_report.total_duration_seconds],
        ["Final Status", execution_report.final_status],
        ["Dry Run", "Yes" if execution_report.dry_run else "No"],
        ["Input Workbook", execution_report.input_workbook],
        ["Products Total", execution_report.products_total],
        ["Products Processed", execution_report.products_processed],
        ["Product Content Generated", execution_report.product_content_generated],
        ["Product Content Failed", execution_report.product_content_failed],
        ["Images Generated", execution_report.images_generated],
        ["Images Failed", execution_report.images_failed],
        ["Images Skipped", execution_report.images_skipped],
        ["Competitor Comparisons Completed", execution_report.competitor_comparisons_completed],
        ["Competitor Comparisons Failed", execution_report.competitor_comparisons_failed],
        ["Sales Records Processed", execution_report.sales_records_processed],
        ["API Failures", execution_report.api_failures],
        ["Retry Counts", execution_report.retry_counts],
        ["Prompt Tokens", execution_report.token_usage.get("prompt_tokens", 0)],
        ["Completion Tokens", execution_report.token_usage.get("completion_tokens", 0)],
        ["Total Tokens", execution_report.token_usage.get("total_tokens", 0)],
        ["Estimated API Cost (USD)", execution_report.estimated_api_cost_usd],
        ["Errors", " | ".join(execution_report.errors) or "None"],
        ["Warnings", " | ".join(execution_report.warnings) or "None"],
    ]
    _style_sheet_table(ws7, ["Execution Field", "Value"], exec_rows)

    wb.save(output_path)
    logger.info("Saved consolidated summary workbook at %s", output_path)
    return output_path
