"""Sales Analysis Agent: deterministic analytics, chart generation, and structured output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config import AppConfig
from processors.sales_processor import analyze_sales
from reports.chart_generator import generate_sales_dashboard
from schemas.product_schema import NormalizedVendorProduct
from utils.file_utils import write_json
from utils.logger import get_logger

logger = get_logger("sales_agent")


class SalesAnalysisAgent:
    """Executes deterministic sales analytics and persists JSON + chart outputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(
        self,
        sales_df: pd.DataFrame,
        vendor_products: Optional[Sequence[NormalizedVendorProduct]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        target_dir = output_dir or self.config.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        analysis = analyze_sales(
            sales_df=sales_df,
            vendor_products=vendor_products,
            confirmed_statuses=self.config.sales_confirmed_statuses,
            excluded_statuses=self.config.sales_excluded_statuses,
            separate_statuses=self.config.sales_separate_statuses,
        )

        summary_json_path = target_dir / "sales_summary.json"
        write_json(summary_json_path, analysis)

        dashboard_path = target_dir / "sales_dashboard.png"
        generate_sales_dashboard(analysis, dashboard_path)

        analysis["artifacts"] = {
            "sales_summary_json": str(summary_json_path),
            "sales_dashboard_png": str(dashboard_path),
        }
        return analysis
