"""End-to-end LangGraph workflow, partial failure recovery, and CLI tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.competitor_agent import CompetitorAnalysisAgent
from agents.content_agent import ContentGenerationAgent
from agents.image_agent import LifestyleImageAgent
from config import AppConfig
from main import main as cli_main
from schemas.product_schema import ProductContentLLMResponse
from services.gemini_image_service import GeminiImageService
from services.search_service import SearchHit
from tests.conftest import create_sample_jpeg_bytes
from utils.cost_tracker import CostTracker
from workflow.graph import run_commerce_workflow


def test_full_workflow_with_partial_image_failure(
    sample_workbook: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    out_dir = tmp_path / "outputs"
    img_dir = tmp_path / "images"

    class _FakeContentService:
        def generate_product_content(self, product):
            return (
                ProductContentLLMResponse.model_validate(
                    {
                        "title": f"TestBrand {product.sku} Furniture Piece",
                        "description": "Well-crafted furniture piece based strictly on vendor attributes.",
                        "features": [
                            "Feature one grounded in vendor data.",
                            "Feature two grounded in vendor data.",
                            "Feature three grounded in vendor data.",
                            "Feature four grounded in vendor data.",
                            "Feature five grounded in vendor data.",
                        ],
                        "attributes": {
                            "colour": product.color,
                            "material": product.material_used,
                            "assembly": product.assembly_required,
                            "dimensions": product.dimensions_summary,
                        },
                    }
                ),
                {"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70, "latency_seconds": 0.05},
            )

    sample_jpg = create_sample_jpeg_bytes()
    monkeypatch.setattr(
        "services.gemini_image_service.requests.get",
        lambda url, timeout=30: SimpleNamespace(
            status_code=200, content=sample_jpg, raise_for_status=lambda: None
        ),
    )

    class _FailingOneImageModels:
        def generate_content(self, **kwargs):
            if "coffee table" in kwargs["contents"][1].lower():
                raise RuntimeError("Simulated image generation failure")
            part = SimpleNamespace(inline_data=SimpleNamespace(data=sample_jpg))
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    class _FakeSearchService:
        is_configured = True

        def search(self, query: str, num_results: int = 8):
            return [
                SearchHit(
                    title="Comparable Furniture Item",
                    url="https://www.amazon.in/dp/B001",
                    snippet="Comparable product",
                    marketplace="Amazon",
                    extracted_price=299.0,
                    currency="USD",
                )
            ]

    cfg = AppConfig(
        input_file=sample_workbook,
        output_dir=out_dir,
        image_dir=img_dir,
        gemini_api_key="test-key",
        search_api_key="test-search",
        max_concurrent_requests=1,
        max_retries=0,
    )
    tracker = CostTracker()

    final_state = run_commerce_workflow(
        config=cfg,
        cost_tracker=tracker,
        content_agent=ContentGenerationAgent(cfg, gemini_service=_FakeContentService(), cost_tracker=tracker),
        image_agent=LifestyleImageAgent(
            cfg,
            image_service=GeminiImageService(
                api_key="test-key", max_retries=0, cost_tracker=tracker, client=SimpleNamespace(models=_FailingOneImageModels())
            ),
            cost_tracker=tracker,
        ),
        competitor_agent=CompetitorAnalysisAgent(cfg, search_service=_FakeSearchService(), cost_tracker=tracker),
    )

    from schemas.execution_schema import ExecutionReport
    report = ExecutionReport.model_validate(final_state["execution_report"])
    assert report.final_status == "completed_with_partial_failures"
    assert report.product_content_generated == 3
    assert report.images_generated == 2
    assert report.images_failed == 1
    assert report.sales_records_processed == 6
    assert (out_dir / "final_summary.xlsx").exists()
    assert (out_dir / "execution_report.json").exists()
    assert (out_dir / "data_quality_report.json").exists()
    assert (out_dir / "product_content.xlsx").exists()
    assert (out_dir / "competitor_analysis.xlsx").exists()
    assert (out_dir / "sales_analysis.xlsx").exists()
    assert (out_dir / "sales_dashboard.png").exists()


def test_workflow_fatal_missing_workbook(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    cfg = AppConfig(input_file=tmp_path / "missing.xlsx", output_dir=out_dir, dry_run=True)
    final_state = run_commerce_workflow(config=cfg)
    assert final_state["final_status"] == "failed"
    assert (out_dir / "execution_report.json").exists()


def test_cli_dry_run_exit_code(sample_workbook: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_out"
    code = cli_main(["--input", str(sample_workbook), "--output-dir", str(out_dir), "--dry-run"])
    assert code == 0
    assert (out_dir / "final_summary.xlsx").exists()
    assert (out_dir / "sales_dashboard.png").exists()
