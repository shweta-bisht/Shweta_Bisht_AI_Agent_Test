"""Tests for Gemini content generation, Pydantic validation, retry handling, and caching."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.content_agent import ContentGenerationAgent
from config import AppConfig
from processors.data_cleaner import clean_workbook_data
from processors.excel_reader import read_workbook
from schemas.product_schema import ProductContentLLMResponse
from services.gemini_service import GeminiContentService
from services.retry_service import QuotaExhaustedError, failure_category
from utils.cost_tracker import CostTracker


class _MockModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        item = self.responses[idx]
        if isinstance(item, Exception):
            raise item
        return item


class _MockGenAIClient:
    def __init__(self, responses):
        self.models = _MockModels(responses)


def test_failure_category_identifies_quota_exhaustion() -> None:
    error = QuotaExhaustedError("Gemini content generation: quota exhausted (quota_daily)")
    assert failure_category(error) == "quota_exhausted"


def _valid_llm_json(title: str = "Benjara Black Power Reclining Chair") -> str:
    return (
        '{"title": "' + title + '", '
        '"description": "This reclining chair features a solid wood frame and integrated cup holders.", '
        '"features": ['
        '"Power reclining mechanism for personalized seating.",'
        '"Solid wood inner frame supports daily use.",'
        '"Integrated cup holders keep beverages secure.",'
        '"Black upholstery complements contemporary interiors.",'
        '"Assembly required with provided components."'
        '], '
        '"attributes": {"colour": "Black", "material": "Wood", "assembly": "Yes", "dimensions": "40x36x41", "weight": "110 lbs", "brand": "TestBrand", "category": "Furniture"}}'
    )


def test_content_schema_rejects_duplicate_features() -> None:
    with pytest.raises(ValueError, match="Duplicate feature"):
        ProductContentLLMResponse.model_validate(
            {
                "title": "Valid Product Title",
                "description": "Valid product description with enough detail.",
                "features": ["Same feat", "Same feat", "Feat 3", "Feat 4", "Feat 5"],
                "attributes": {},
            }
        )


def test_gemini_service_retries_on_timeout_and_malformed_json(
    sample_workbook: Path, monkeypatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    product = cleaned.vendor_products[0]

    bad_resp = SimpleNamespace(text="{not valid json", parsed=None, usage_metadata=None)
    good_resp = SimpleNamespace(
        text=_valid_llm_json(),
        parsed=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=80, total_token_count=180
        ),
    )
    client = _MockGenAIClient([TimeoutError("API timed out"), bad_resp, good_resp])
    tracker = CostTracker()

    svc = GeminiContentService(
        api_key="test-key", max_retries=3, cost_tracker=tracker, client=client
    )
    parsed, metrics = svc.generate_product_content(product)

    assert parsed.title == "Benjara Black Power Reclining Chair"
    assert metrics["retries"] == 2
    assert tracker.retries == 2
    assert tracker.total_tokens == 180


def test_content_agent_sku_partial_failure_and_grounding(
    sample_workbook: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cleaned = clean_workbook_data(read_workbook(sample_workbook))

    class _MixedService:
        def generate_product_content(self, product):
            if product.sku == "SKU-102":
                raise RuntimeError("Simulated Gemini outage for SKU-102")
            resp = ProductContentLLMResponse.model_validate_json(_valid_llm_json(f"Title for {product.sku}"))
            return resp, {"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90, "latency_seconds": 0.1}

    cfg = AppConfig(input_file=sample_workbook, output_dir=tmp_path, max_concurrent_requests=1)
    agent = ContentGenerationAgent(cfg, gemini_service=_MixedService())
    results = agent.run(cleaned.vendor_products, output_dir=tmp_path)

    assert len(results) == 3
    status_map = {r.sku: r for r in results}
    assert status_map["SKU-101"].status == "completed"
    assert status_map["SKU-102"].status == "failed"
    assert status_map["SKU-102"].error_category == "transient"
    assert status_map["SKU-103"].status == "completed"
    # SKU-103 had Color=None in workbook, so grounding enforces colour=None
    assert status_map["SKU-103"].attributes.colour is None
    assert (tmp_path / "product_content.json").exists()


@pytest.mark.live
def test_live_gemini_content_generation(sample_workbook: Path, tmp_path: Path) -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_TESTS=1 to run live Gemini API integration test.")
    cfg = AppConfig.from_env(input_file=str(sample_workbook), output_dir=str(tmp_path))
    cfg.validate(require_input_file=True)
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    agent = ContentGenerationAgent(cfg)
    results = agent.run([cleaned.vendor_products[0]], output_dir=tmp_path)
    assert results[0].status == "completed"
    assert len(results[0].features) == 5
