"""LangGraph node implementations wrapping CommerceAI processors, agents, and reporters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.competitor_agent import CompetitorAnalysisAgent
from agents.content_agent import ContentGenerationAgent
from agents.image_agent import LifestyleImageAgent
from agents.sales_agent import SalesAnalysisAgent
from config import AppConfig
from processors.data_cleaner import clean_workbook_data
from processors.excel_reader import ExcelIngestionResult, read_workbook
from reports.excel_report import (
    export_competitor_analysis_workbook,
    export_final_summary_workbook,
    export_product_content_workbook,
    export_sales_analysis_workbook,
)
from reports.summary_generator import build_execution_report, write_execution_report
from schemas.competitor_schema import CompetitorResearchResult
from schemas.execution_schema import DataQualityReport, ImageGenerationRecord
from schemas.product_schema import NormalizedVendorProduct, ProductContentItem
from utils.cost_tracker import CostTracker
from utils.file_utils import ensure_dir, write_json
from utils.logger import get_logger
from workflow.state import CommerceAIState

logger = get_logger("workflow.nodes")


class WorkflowNodes:
    """Stateful container binding runtime config, injected agents, and heavy DataFrames."""

    def __init__(
        self,
        config: AppConfig,
        cost_tracker: Optional[CostTracker] = None,
        content_agent: Optional[ContentGenerationAgent] = None,
        image_agent: Optional[LifestyleImageAgent] = None,
        competitor_agent: Optional[CompetitorAnalysisAgent] = None,
        sales_agent: Optional[SalesAnalysisAgent] = None,
    ) -> None:
        self.config = config
        self.cost_tracker = cost_tracker or CostTracker()
        self.content_agent = content_agent
        self.image_agent = image_agent
        self.competitor_agent = competitor_agent
        self.sales_agent = sales_agent

        # Keep non-msgpack heavy runtime objects (DataFrames) off checkpoint state
        self._ingestion_by_run: Dict[str, ExcelIngestionResult] = {}
        self._sales_df_by_run: Dict[str, pd.DataFrame] = {}

    def excel_ingestion(self, state: CommerceAIState) -> Dict[str, Any]:
        logger.info("[Node: excel_ingestion] Reading workbook '%s'", state["input_workbook_path"])
        try:
            ingestion = read_workbook(state["input_workbook_path"])
            self._ingestion_by_run[state["run_id"]] = ingestion
            v = ingestion.validation
            return {
                "ingestion_summary": {
                    "workbook_path": v.workbook_path,
                    "sheet_names": v.sheet_names,
                    "vendor_row_count": v.vendor_row_count,
                    "sales_row_count": v.sales_row_count,
                    "duplicate_vendor_skus": v.duplicate_vendor_skus,
                },
                "fatal_error": False,
            }
        except Exception as exc:
            msg = f"Excel ingestion failed: {exc}"
            logger.error(msg)
            return {
                "fatal_error": True,
                "errors": [msg],
                "final_status": "failed",
            }

    def data_cleaning(self, state: CommerceAIState) -> Dict[str, Any]:
        if state.get("fatal_error"):
            return {}
        logger.info("[Node: data_cleaning] Cleaning and validating vendor & sales sheets")
        try:
            ingestion = self._ingestion_by_run[state["run_id"]]
            cleaned = clean_workbook_data(ingestion)
            self._sales_df_by_run[state["run_id"]] = cleaned.sales_df

            dq_path = ensure_dir(self.config.output_dir) / "data_quality_report.json"
            write_json(dq_path, cleaned.data_quality_report.model_dump())
            return {
                "vendor_products": [p.model_dump() for p in cleaned.vendor_products],
                "sales_record_count": len(cleaned.sales_df),
                "data_quality_report": cleaned.data_quality_report.model_dump(),
                "generated_files": [str(dq_path)],
                "fatal_error": False,
            }
        except Exception as exc:
            msg = f"Data cleaning failed: {exc}"
            logger.error(msg)
            return {"fatal_error": True, "errors": [msg], "final_status": "failed"}

    def _deserialize_products(self, state: CommerceAIState) -> List[NormalizedVendorProduct]:
        return [
            NormalizedVendorProduct.model_validate(item)
            for item in (state.get("vendor_products") or [])
        ]

    def content_generation(self, state: CommerceAIState) -> Dict[str, Any]:
        if self.config.dry_run or self.config.skip_content:
            reason = "dry-run mode" if self.config.dry_run else "--skip-content flag"
            logger.info("[Node: content_generation] Skipped (%s).", reason)
            return {
                "product_content": [],
                "warnings": [f"Product content generation skipped ({reason})."],
            }

        logger.info("[Node: content_generation] Generating structured content with Gemini")
        try:
            products = self._deserialize_products(state)
            agent = self.content_agent or ContentGenerationAgent(
                self.config, cost_tracker=self.cost_tracker
            )
            items = agent.run(products, output_dir=self.config.output_dir)
            failed = [it for it in items if it.status != "completed"]
            errs = [f"Content failed for {it.sku}: {it.error}" for it in failed]
            files = [str(self.config.output_dir / "product_content.json")]
            return {
                "product_content": [it.model_dump() for it in items],
                "errors": errs,
                "generated_files": files,
            }
        except Exception as exc:
            msg = f"Content generation node failed: {exc}"
            logger.error(msg)
            return {"product_content": [], "errors": [msg]}

    def image_generation(self, state: CommerceAIState) -> Dict[str, Any]:
        if self.config.dry_run or self.config.skip_images:
            reason = "dry-run mode" if self.config.dry_run else "--skip-images flag"
            logger.info("[Node: image_generation] Skipped (%s).", reason)
            return {
                "image_records": [],
                "warnings": [f"Lifestyle image generation skipped ({reason})."],
            }

        logger.info("[Node: image_generation] Generating lifestyle images via Gemini")
        try:
            products = self._deserialize_products(state)
            agent = self.image_agent or LifestyleImageAgent(
                self.config, cost_tracker=self.cost_tracker
            )
            records = agent.run(products, image_dir=self.config.image_dir)
            failed = [r for r in records if r.status == "failed"]
            errs = [f"Image generation failed for {r.sku}: {r.error}" for r in failed]
            return {"image_records": [r.model_dump() for r in records], "errors": errs}
        except Exception as exc:
            msg = f"Image generation node failed: {exc}"
            logger.error(msg)
            return {"image_records": [], "errors": [msg]}

    def sales_analysis(self, state: CommerceAIState) -> Dict[str, Any]:
        logger.info("[Node: sales_analysis] Running deterministic Pandas sales analytics")
        try:
            sales_df = self._sales_df_by_run[state["run_id"]]
            products = self._deserialize_products(state)
            agent = self.sales_agent or SalesAnalysisAgent(self.config)
            analysis = agent.run(
                sales_df=sales_df,
                vendor_products=products,
                output_dir=self.config.output_dir,
            )
            artifacts = analysis.get("artifacts", {})
            files = [p for p in artifacts.values() if p]
            return {"sales_analysis": analysis, "generated_files": files}
        except Exception as exc:
            msg = f"Sales analysis failed: {exc}"
            logger.error(msg)
            return {"sales_analysis": {}, "errors": [msg]}

    def competitor_research(self, state: CommerceAIState) -> Dict[str, Any]:
        if self.config.dry_run or self.config.skip_competitors:
            reason = "dry-run mode" if self.config.dry_run else "--skip-competitors flag"
            logger.info("[Node: competitor_research] Skipped (%s).", reason)
            return {
                "competitor_result": None,
                "warnings": [f"Competitor research skipped ({reason})."],
            }

        logger.info("[Node: competitor_research] Researching comparable products")
        try:
            products = self._deserialize_products(state)
            sales_df = self._sales_df_by_run[state["run_id"]]
            agent = self.competitor_agent or CompetitorAnalysisAgent(
                self.config, cost_tracker=self.cost_tracker
            )
            result: CompetitorResearchResult = agent.run(
                vendor_products=products,
                sales_df=sales_df,
            )
            failed_comps = [c for c in result.comparisons if c.status == "failed"]
            errs = [f"Competitor research failed for {c.our_sku}: {c.error}" for c in failed_comps]
            return {
                "competitor_result": result.model_dump(),
                "warnings": result.warnings,
                "errors": errs,
            }
        except Exception as exc:
            msg = f"Competitor research node failed: {exc}"
            logger.error(msg)
            return {"competitor_result": None, "errors": [msg]}

    def output_generation(self, state: CommerceAIState) -> Dict[str, Any]:
        logger.info("[Node: output_generation] Writing consolidated Excel & JSON reports")
        out_dir = ensure_dir(self.config.output_dir)
        new_files: List[str] = []

        content_items = [
            ProductContentItem.model_validate(d) for d in (state.get("product_content") or [])
        ]
        image_records = [
            ImageGenerationRecord.model_validate(d) for d in (state.get("image_records") or [])
        ]
        comp_raw = state.get("competitor_result")
        competitor_result = (
            CompetitorResearchResult.model_validate(comp_raw) if comp_raw else None
        )
        sales_analysis = state.get("sales_analysis") or {}

        dashboard_path = out_dir / "sales_dashboard.png"
        if not dashboard_path.exists():
            dashboard_path = None

        try:
            if content_items:
                pc_xlsx = export_product_content_workbook(
                    content_items,
                    out_dir / "product_content.xlsx",
                    image_records=image_records,
                )
                new_files.append(str(pc_xlsx))

            if competitor_result is not None:
                comp_xlsx = export_competitor_analysis_workbook(
                    competitor_result,
                    out_dir / "competitor_analysis.xlsx",
                )
                new_files.append(str(comp_xlsx))

            if sales_analysis:
                sales_xlsx = export_sales_analysis_workbook(
                    sales_analysis,
                    out_dir / "sales_analysis.xlsx",
                    dashboard_path=dashboard_path,
                )
                new_files.append(str(sales_xlsx))

            return {"generated_files": new_files}
        except Exception as exc:
            msg = f"Output generation failed: {exc}"
            logger.error(msg)
            return {"errors": [msg], "generated_files": new_files}

    def execution_summary(self, state: CommerceAIState) -> Dict[str, Any]:
        logger.info("[Node: execution_summary] Finalizing execution report")
        out_dir = ensure_dir(self.config.output_dir)
        start_dt = datetime.fromisoformat(state["start_time_iso"])
        end_dt = datetime.now(timezone.utc)

        content_items = [
            ProductContentItem.model_validate(d) for d in (state.get("product_content") or [])
        ]
        image_records = [
            ImageGenerationRecord.model_validate(d) for d in (state.get("image_records") or [])
        ]
        comp_raw = state.get("competitor_result")
        competitor_result = (
            CompetitorResearchResult.model_validate(comp_raw) if comp_raw else None
        )
        dq_raw = state.get("data_quality_report")
        data_quality = DataQualityReport.model_validate(dq_raw) if dq_raw else None

        sales_analysis = state.get("sales_analysis") or {}
        vendor_products = state.get("vendor_products") or []
        sales_count = int(state.get("sales_record_count") or 0)

        report_json_path = out_dir / "execution_report.json"
        final_xlsx_path = out_dir / "final_summary.xlsx"

        all_files = sorted(
            set(
                list(state.get("generated_files") or [])
                + [str(report_json_path), str(final_xlsx_path)]
            )
        )

        report = build_execution_report(
            run_id=state["run_id"],
            start_dt=start_dt,
            end_dt=end_dt,
            input_workbook=state["input_workbook_path"],
            dry_run=self.config.dry_run,
            products_total=len(vendor_products),
            content_items=content_items,
            image_records=image_records,
            competitor_result=competitor_result,
            sales_records_processed=sales_count,
            cost_tracker=self.cost_tracker,
            errors=state.get("errors") or [],
            warnings=state.get("warnings") or [],
            generated_files=all_files,
            images_skipped=self.config.skip_images,
            competitor_skipped=self.config.skip_competitors,
        )

        write_execution_report(report, report_json_path)

        if not state.get("fatal_error"):
            dashboard_path = out_dir / "sales_dashboard.png"
            export_final_summary_workbook(
                output_path=final_xlsx_path,
                content_items=content_items,
                image_records=image_records,
                competitor_result=competitor_result,
                sales_analysis=sales_analysis,
                data_quality=data_quality,
                execution_report=report,
                dashboard_path=dashboard_path if dashboard_path.exists() else None,
            )

        return {
            "execution_report": report.model_dump(),
            "final_status": report.final_status,
            "generated_files": [str(report_json_path), str(final_xlsx_path)],
        }
