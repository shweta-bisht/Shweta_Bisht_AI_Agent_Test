"""Matplotlib dashboard and chart generator for CommerceAI sales analytics."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "commerce_ai_mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from utils.file_utils import ensure_dir
from utils.logger import get_logger

logger = get_logger("chart_generator")


def generate_sales_dashboard(
    sales_analysis: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Render a 5-panel sales analytics dashboard PNG and return its path."""
    ensure_dir(output_path.parent)

    monthly: List[Dict[str, Any]] = sales_analysis.get("monthly_performance", [])
    top_rev: List[Dict[str, Any]] = sales_analysis.get("top_10_by_revenue", [])
    top_units: List[Dict[str, Any]] = sales_analysis.get("top_10_by_units", [])
    vendors: List[Dict[str, Any]] = sales_analysis.get("vendor_performance", [])
    statuses: List[Dict[str, Any]] = sales_analysis.get("status_breakdown", [])
    kpis: Dict[str, Any] = sales_analysis.get("kpis", {})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 1.0], hspace=0.42, wspace=0.28)

    fig.suptitle(
        f"CommerceAI Sales Performance Dashboard  |  "
        f"Confirmed Revenue: ${kpis.get('total_confirmed_sales', 0):,.2f}  |  "
        f"Units: {kpis.get('total_units_sold', 0):,}  |  "
        f"Confirmed Orders: {kpis.get('distinct_confirmed_orders', 0):,}  |  "
        f"AOV: ${kpis.get('average_order_value', 0):,.2f}",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    # 1. Monthly Revenue + Units (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    if monthly:
        months = [m["month"] for m in monthly]
        revs = [m["confirmed_revenue"] for m in monthly]
        units = [m["units_sold"] for m in monthly]
        bars = ax1.bar(months, revs, color="#2563EB", alpha=0.88, width=0.55)
        ax1.set_title("Monthly Confirmed Revenue & Units", fontweight="bold", fontsize=11)
        ax1.set_ylabel("Confirmed Revenue ($)")
        ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        for bar, rev, u in zip(bars, revs, units):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"${rev:,.0f}\n({u:,} units)",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax1.margins(y=0.22)
    else:
        ax1.text(0.5, 0.5, "No monthly data", ha="center", va="center")
        ax1.set_title("Monthly Confirmed Revenue")

    # 2. Order Status Distribution (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    if statuses:
        st_names = [s["order_status"] for s in statuses]
        st_lines = [s["order_lines"] for s in statuses]
        colors = ["#16A34A" if "included" in s["treatment"].lower() else "#F59E0B" if "separate" in s["treatment"].lower() else "#DC2626" for s in statuses]
        bars2 = ax2.barh(st_names[::-1], st_lines[::-1], color=colors[::-1], alpha=0.88)
        ax2.set_title("Order Status Distribution (Lines & Treatment)", fontweight="bold", fontsize=11)
        ax2.set_xlabel("Order Lines")
        ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        for bar, cnt in zip(bars2, st_lines[::-1]):
            ax2.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2.0,
                f" {cnt:,}",
                va="center",
                fontsize=8.5,
            )
        ax2.margins(x=0.18)
    else:
        ax2.text(0.5, 0.5, "No status data", ha="center", va="center")
        ax2.set_title("Order Status Distribution")

    # 3. Top 10 SKUs by Revenue (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    if top_rev:
        skus = [item["sku"] for item in top_rev][::-1]
        vals = [item["revenue"] for item in top_rev][::-1]
        bars3 = ax3.barh(skus, vals, color="#0D9488", alpha=0.9)
        ax3.set_title("Top 10 SKUs by Confirmed Revenue", fontweight="bold", fontsize=11)
        ax3.set_xlabel("Revenue ($)")
        ax3.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        for bar, v in zip(bars3, vals):
            ax3.text(bar.get_width(), bar.get_y() + bar.get_height() / 2.0, f" ${v:,.0f}", va="center", fontsize=8)
        ax3.margins(x=0.2)
    else:
        ax3.text(0.5, 0.5, "No confirmed SKU revenue", ha="center", va="center")
        ax3.set_title("Top 10 SKUs by Revenue")

    # 4. Top 10 SKUs by Units (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    if top_units:
        skus_u = [item["sku"] for item in top_units][::-1]
        vals_u = [item["units_sold"] for item in top_units][::-1]
        bars4 = ax4.barh(skus_u, vals_u, color="#7C3AED", alpha=0.88)
        ax4.set_title("Top 10 SKUs by Units Sold", fontweight="bold", fontsize=11)
        ax4.set_xlabel("Units Sold")
        ax4.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        for bar, v in zip(bars4, vals_u):
            ax4.text(bar.get_width(), bar.get_y() + bar.get_height() / 2.0, f" {v:,}", va="center", fontsize=8)
        ax4.margins(x=0.18)
    else:
        ax4.text(0.5, 0.5, "No confirmed SKU units", ha="center", va="center")
        ax4.set_title("Top 10 SKUs by Units")

    # 5. Vendor Performance (Bottom Spanning Full Width)
    ax5 = fig.add_subplot(gs[2, :])
    if vendors:
        v_names = [v["vendor"] for v in vendors]
        v_rev = [v["confirmed_revenue"] for v in vendors]
        v_shares = [v["revenue_share_pct"] for v in vendors]
        bars5 = ax5.bar(v_names, v_rev, color="#1D4ED8", alpha=0.85, width=0.5)
        ax5.set_title("Vendor Channel Performance (Confirmed Revenue)", fontweight="bold", fontsize=11)
        ax5.set_ylabel("Confirmed Revenue ($)")
        ax5.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        for bar, rev, pct in zip(bars5, v_rev, v_shares):
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"${rev:,.0f} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax5.margins(y=0.22)
    else:
        ax5.text(0.5, 0.5, "No vendor performance data", ha="center", va="center")
        ax5.set_title("Vendor Channel Performance")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated sales dashboard chart at %s", output_path)
    return output_path
