"""Deterministic Pandas-based sales analytics processor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from schemas.product_schema import NormalizedVendorProduct
from utils.logger import get_logger

logger = get_logger("sales_processor")


def _safe_round(val: Any, digits: int = 2) -> float:
    if val is None or pd.isna(val):
        return 0.0
    return round(float(val), digits)


def _generate_business_insights(
    kpis: Dict[str, Any],
    monthly_perf: List[Dict[str, Any]],
    vendor_perf: List[Dict[str, Any]],
    carrier_dist: List[Dict[str, Any]],
    status_breakdown: List[Dict[str, Any]],
    top_revenue_skus: List[Dict[str, Any]],
    oj_summary: Dict[str, Any],
) -> List[str]:
    """Generate 3 to 5 strictly data-backed business insights from computed metrics."""
    insights: List[str] = []

    confirmed_rev = kpis["total_confirmed_sales"]
    confirmed_orders = kpis["distinct_confirmed_orders"]
    aov = kpis["average_order_value"]
    total_lines = kpis["total_order_lines"]

    # 1. Status & OJ Shipping revenue exposure
    oj_lines = oj_summary.get("order_lines", 0)
    oj_rev = oj_summary.get("potential_revenue", 0.0)
    cancelled_lines = sum(
        row["order_lines"] for row in status_breakdown if row["order_status"].lower() == "cancelled"
    )
    if confirmed_rev > 0:
        oj_pct = (oj_rev / confirmed_rev) * 100.0
        cancel_pct = (cancelled_lines / total_lines) * 100.0 if total_lines else 0.0
        insights.append(
            f"Confirmed sales reached ${confirmed_rev:,.2f} across {confirmed_orders:,} distinct orders "
            f"(AOV ${aov:,.2f}); however, ${oj_rev:,.2f} across {oj_lines:,} lines ({oj_pct:.1f}% of confirmed revenue) "
            f"sits in 'OJ Shipping' status and requires operational definition before revenue recognition, "
            f"while {cancelled_lines:,} lines ({cancel_pct:.1f}%) were cancelled."
        )

    # 2. Monthly sales trajectory
    if len(monthly_perf) >= 2:
        peak_month = max(monthly_perf, key=lambda x: x["confirmed_revenue"])
        low_month = min(monthly_perf, key=lambda x: x["confirmed_revenue"])
        insights.append(
            f"Monthly confirmed revenue peaked in {peak_month['month']} at ${peak_month['confirmed_revenue']:,.2f} "
            f"({peak_month['units_sold']:,} units across {peak_month['distinct_orders']:,} orders), compared to "
            f"{low_month['month']} at ${low_month['confirmed_revenue']:,.2f} (note partial-month coverage at date boundaries)."
        )

    # 3. SKU concentration
    if top_revenue_skus and confirmed_rev > 0:
        top10_rev = sum(item["revenue"] for item in top_revenue_skus)
        top10_share = (top10_rev / confirmed_rev) * 100.0
        leader = top_revenue_skus[0]
        insights.append(
            f"Revenue has a long-tail structure: the top 10 SKUs generated ${top10_rev:,.2f} ({top10_share:.1f}% of "
            f"confirmed revenue), led by {leader['sku']} at ${leader['revenue']:,.2f} across {leader['units_sold']:,} units."
        )

    # 4. Vendor channel concentration
    if vendor_perf and confirmed_rev > 0:
        top_vendor = vendor_perf[0]
        share = (top_vendor["confirmed_revenue"] / confirmed_rev) * 100.0
        second_line = (
            f", followed by {vendor_perf[1]['vendor']} (${vendor_perf[1]['confirmed_revenue']:,.2f})"
            if len(vendor_perf) > 1
            else ""
        )
        insights.append(
            f"Fulfillment channel '{top_vendor['vendor']}' accounts for ${top_vendor['confirmed_revenue']:,.2f} "
            f"({share:.1f}% of confirmed revenue){second_line}, indicating strong reliance on primary warehouse/import stock."
        )

    # 5. Unmapped shipping carriers
    unmapped = [c for c in carrier_dist if "not mapped" in c["shipping_carrier"].lower()]
    if unmapped and total_lines > 0:
        unmapped_lines = sum(c["order_lines"] for c in unmapped)
        unmapped_share = (unmapped_lines / total_lines) * 100.0
        insights.append(
            f"{unmapped_lines:,} order lines ({unmapped_share:.1f}% of all lines) have 'Carrier Not Mapped', "
            f"creating a logistics visibility gap that limits freight cost attribution and delivery SLA tracking."
        )

    return insights[:5]


def analyze_sales(
    sales_df: pd.DataFrame,
    vendor_products: Optional[Sequence[NormalizedVendorProduct]] = None,
    confirmed_statuses: Sequence[str] = ("Processed",),
    excluded_statuses: Sequence[str] = ("Cancelled", "Pending"),
    separate_statuses: Sequence[str] = ("OJ Shipping",),
) -> Dict[str, Any]:
    """Perform deterministic sales analytics using Pandas with explicit order-status rules."""
    df = sales_df.copy()
    if "orderDate_parsed" not in df.columns:
        df["orderDate_parsed"] = pd.to_datetime(df["orderDate"], errors="coerce")

    confirmed_set = {s.strip().lower() for s in confirmed_statuses}
    excluded_set = {s.strip().lower() for s in excluded_statuses}
    separate_set = {s.strip().lower() for s in separate_statuses}

    status_lower = df["orderStatus"].astype(str).str.strip().str.lower()
    confirmed_df = df[status_lower.isin(confirmed_set)].copy()
    excluded_df = df[status_lower.isin(excluded_set)].copy()
    separate_df = df[status_lower.isin(separate_set)].copy()
    other_df = df[~status_lower.isin(confirmed_set | excluded_set | separate_set)].copy()

    total_confirmed_sales = _safe_round(confirmed_df["amount"].sum())
    total_units_sold = int(confirmed_df["quantity"].fillna(0).sum())
    distinct_confirmed_orders = int(confirmed_df["orderId"].nunique())
    distinct_all_orders = int(df["orderId"].nunique())
    aov = _safe_round(
        total_confirmed_sales / distinct_confirmed_orders if distinct_confirmed_orders > 0 else 0.0
    )

    # Order status distribution across all rows
    status_grouped = (
        df.groupby("orderStatus", dropna=False)
        .agg(
            order_lines=("orderId", "count"),
            distinct_orders=("orderId", "nunique"),
            units=("quantity", "sum"),
            amount=("amount", "sum"),
        )
        .reset_index()
        .sort_values("order_lines", ascending=False)
    )
    status_breakdown: List[Dict[str, Any]] = []
    for _, r in status_grouped.iterrows():
        st = str(r["orderStatus"])
        st_l = st.lower()
        treatment = (
            "Included (Confirmed Sales)"
            if st_l in confirmed_set
            else "Reported Separately"
            if st_l in separate_set
            else "Excluded"
        )
        status_breakdown.append(
            {
                "order_status": st,
                "treatment": treatment,
                "order_lines": int(r["order_lines"]),
                "distinct_orders": int(r["distinct_orders"]),
                "units": int(r["units"]) if pd.notna(r["units"]) else 0,
                "amount": _safe_round(r["amount"]),
            }
        )

    # OJ Shipping / Separate statuses summary
    oj_summary = {
        "statuses": list(separate_statuses),
        "order_lines": int(len(separate_df)),
        "distinct_orders": int(separate_df["orderId"].nunique()) if not separate_df.empty else 0,
        "units": int(separate_df["quantity"].fillna(0).sum()) if not separate_df.empty else 0,
        "potential_revenue": _safe_round(separate_df["amount"].sum()) if not separate_df.empty else 0.0,
        "note": (
            "OJ Shipping is reported separately and excluded from confirmed sales until its "
            "business definition (e.g., in-transit vs. unbilled/pending) is verified."
        ),
    }

    # SKU aggregation on confirmed sales
    sku_col = "Masked SKU" if "Masked SKU" in confirmed_df.columns else "itemName"
    sku_grouped = (
        confirmed_df.groupby(sku_col, dropna=False)
        .agg(
            revenue=("amount", "sum"),
            units_sold=("quantity", "sum"),
            distinct_orders=("orderId", "nunique"),
            avg_unit_price=("price", "mean"),
        )
        .reset_index()
    )
    sku_grouped = sku_grouped.rename(columns={sku_col: "sku"})

    top10_rev_df = sku_grouped.sort_values(["revenue", "units_sold"], ascending=[False, False]).head(10)
    top10_units_df = sku_grouped.sort_values(["units_sold", "revenue"], ascending=[False, False]).head(10)

    # Low-performing SKUs: positive-price confirmed SKUs with lowest revenue/units
    positive_rev_skus = sku_grouped[sku_grouped["revenue"] > 0]
    low_perf_df = positive_rev_skus.sort_values(
        ["revenue", "units_sold"], ascending=[True, True]
    ).head(10)

    def _sku_rows(sub_df: pd.DataFrame) -> List[Dict[str, Any]]:
        rows = []
        for _, r in sub_df.iterrows():
            rows.append(
                {
                    "sku": str(r["sku"]),
                    "revenue": _safe_round(r["revenue"]),
                    "units_sold": int(r["units_sold"]),
                    "distinct_orders": int(r["distinct_orders"]),
                    "avg_unit_price": _safe_round(r["avg_unit_price"]),
                }
            )
        return rows

    top_10_by_revenue = _sku_rows(top10_rev_df)
    top_10_by_units = _sku_rows(top10_units_df)
    low_performing_skus = _sku_rows(low_perf_df)

    # Non-moving SKUs where a defensible product universe exists:
    # 1) Sales-sheet SKUs that appear in non-confirmed lines (Cancelled/Pending/OJ Shipping) with 0 confirmed sales
    confirmed_sku_set = set(sku_grouped[sku_grouped["units_sold"] > 0]["sku"].astype(str))
    all_sales_sku_set = set(df[sku_col].dropna().astype(str))
    sales_non_moving = sorted(all_sales_sku_set - confirmed_sku_set)

    # 2) Vendor-sheet SKUs with 0 confirmed sales (explicitly noting SKU system mismatch if 0 overlap)
    vendor_universe_skus = [p.sku for p in (vendor_products or [])]
    vendor_non_moving = [sku for sku in vendor_universe_skus if sku not in confirmed_sku_set]
    vendor_overlap_count = len(set(vendor_universe_skus).intersection(all_sales_sku_set))

    non_moving_summary = {
        "sales_catalog_non_moving_count": len(sales_non_moving),
        "sales_catalog_non_moving_sample": sales_non_moving[:25],
        "sales_catalog_note": (
            "Sales SKUs appearing only in Cancelled/Pending/OJ Shipping lines with 0 confirmed Processed units."
        ),
        "vendor_catalog_total": len(vendor_universe_skus),
        "vendor_catalog_overlap_with_sales": vendor_overlap_count,
        "vendor_catalog_non_moving_count": len(vendor_non_moving),
        "vendor_catalog_note": (
            "Vendor SKUs have 0 exact matches in the Sales Masked SKU/itemName universe; "
            "they cannot be assumed active or non-moving in sales without a cross-system SKU mapping table."
            if vendor_overlap_count == 0 and vendor_universe_skus
            else f"{len(vendor_non_moving)} vendor SKUs had zero confirmed sales."
        ),
    }

    # Vendor channel performance (confirmed sales)
    vendor_grouped = (
        confirmed_df.groupby("Vendor", dropna=False)
        .agg(
            confirmed_revenue=("amount", "sum"),
            units_sold=("quantity", "sum"),
            distinct_orders=("orderId", "nunique"),
            order_lines=("orderId", "count"),
        )
        .reset_index()
        .sort_values("confirmed_revenue", ascending=False)
    )
    vendor_performance = [
        {
            "vendor": str(r["Vendor"]),
            "confirmed_revenue": _safe_round(r["confirmed_revenue"]),
            "units_sold": int(r["units_sold"]),
            "distinct_orders": int(r["distinct_orders"]),
            "order_lines": int(r["order_lines"]),
            "revenue_share_pct": _safe_round(
                (r["confirmed_revenue"] / total_confirmed_sales * 100.0)
                if total_confirmed_sales > 0
                else 0.0
            ),
        }
        for _, r in vendor_grouped.iterrows()
    ]

    # Top vendorType (supplier) breakdown on confirmed sales
    vtype_grouped = (
        confirmed_df.groupby("vendorType", dropna=False)
        .agg(
            confirmed_revenue=("amount", "sum"),
            units_sold=("quantity", "sum"),
            distinct_orders=("orderId", "nunique"),
        )
        .reset_index()
        .sort_values("confirmed_revenue", ascending=False)
        .head(15)
    )
    top_vendor_types = [
        {
            "vendor_type": str(r["vendorType"]),
            "confirmed_revenue": _safe_round(r["confirmed_revenue"]),
            "units_sold": int(r["units_sold"]),
            "distinct_orders": int(r["distinct_orders"]),
        }
        for _, r in vtype_grouped.iterrows()
    ]

    # Shipping carrier distribution across all order lines and confirmed sales
    carrier_all = (
        df.groupby("shippingCarrier", dropna=False)
        .agg(
            order_lines=("orderId", "count"),
            distinct_orders=("orderId", "nunique"),
            units=("quantity", "sum"),
            total_amount=("amount", "sum"),
        )
        .reset_index()
        .sort_values("order_lines", ascending=False)
    )
    total_all_lines = len(df)
    carrier_distribution = [
        {
            "shipping_carrier": str(r["shippingCarrier"]),
            "order_lines": int(r["order_lines"]),
            "line_share_pct": _safe_round(
                (r["order_lines"] / total_all_lines * 100.0) if total_all_lines else 0.0
            ),
            "distinct_orders": int(r["distinct_orders"]),
            "units": int(r["units"]) if pd.notna(r["units"]) else 0,
            "total_amount": _safe_round(r["total_amount"]),
        }
        for _, r in carrier_all.iterrows()
    ]

    # Monthly sales performance and monthly units sold (confirmed sales)
    valid_confirmed = confirmed_df[confirmed_df["orderDate_parsed"].notna()].copy()
    valid_confirmed["month"] = valid_confirmed["orderDate_parsed"].dt.to_period("M").astype(str)
    monthly_grouped = (
        valid_confirmed.groupby("month")
        .agg(
            confirmed_revenue=("amount", "sum"),
            units_sold=("quantity", "sum"),
            distinct_orders=("orderId", "nunique"),
            order_lines=("orderId", "count"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly_performance = [
        {
            "month": str(r["month"]),
            "confirmed_revenue": _safe_round(r["confirmed_revenue"]),
            "units_sold": int(r["units_sold"]),
            "distinct_orders": int(r["distinct_orders"]),
            "order_lines": int(r["order_lines"]),
            "average_order_value": _safe_round(
                r["confirmed_revenue"] / r["distinct_orders"] if r["distinct_orders"] > 0 else 0.0
            ),
        }
        for _, r in monthly_grouped.iterrows()
    ]

    kpis = {
        "total_confirmed_sales": total_confirmed_sales,
        "total_units_sold": total_units_sold,
        "distinct_confirmed_orders": distinct_confirmed_orders,
        "distinct_all_orders": distinct_all_orders,
        "average_order_value": aov,
        "total_order_lines": int(len(df)),
        "confirmed_order_lines": int(len(confirmed_df)),
        "excluded_order_lines": int(len(excluded_df)),
        "separate_status_order_lines": int(len(separate_df)),
        "unclassified_status_order_lines": int(len(other_df)),
        "unique_confirmed_skus": int(len(confirmed_sku_set)),
        "unique_total_sales_skus": int(len(all_sales_sku_set)),
        "confirmed_statuses_rule": list(confirmed_statuses),
        "excluded_statuses_rule": list(excluded_statuses),
        "separate_statuses_rule": list(separate_statuses),
    }

    insights = _generate_business_insights(
        kpis=kpis,
        monthly_perf=monthly_performance,
        vendor_perf=vendor_performance,
        carrier_dist=carrier_distribution,
        status_breakdown=status_breakdown,
        top_revenue_skus=top_10_by_revenue,
        oj_summary=oj_summary,
    )

    logger.info(
        "Sales analytics complete: confirmed=$%.2f, units=%d, orders=%d, AOV=$%.2f.",
        total_confirmed_sales,
        total_units_sold,
        distinct_confirmed_orders,
        aov,
    )

    return {
        "kpis": kpis,
        "status_breakdown": status_breakdown,
        "oj_shipping_summary": oj_summary,
        "monthly_performance": monthly_performance,
        "top_10_by_revenue": top_10_by_revenue,
        "top_10_by_units": top_10_by_units,
        "low_performing_skus": low_performing_skus,
        "non_moving_skus": non_moving_summary,
        "vendor_performance": vendor_performance,
        "top_vendor_types": top_vendor_types,
        "carrier_distribution": carrier_distribution,
        "business_insights": insights,
    }
