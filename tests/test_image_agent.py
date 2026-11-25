"""Tests for reference image download, validation, and Gemini lifestyle image generation."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.image_agent import LifestyleImageAgent
from config import AppConfig
from processors.data_cleaner import clean_workbook_data
from processors.excel_reader import read_workbook
from services.gemini_image_service import GeminiImageService, ImageGenerationError
from tests.conftest import create_sample_jpeg_bytes


class _MockImageModels:
    def __init__(self, image_bytes=None, exc=None):
        self.image_bytes = image_bytes
        self.exc = exc

    def generate_content(self, **kwargs):
        if self.exc:
            raise self.exc
        part = SimpleNamespace(inline_data=SimpleNamespace(data=self.image_bytes))
        candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
        return SimpleNamespace(candidates=[candidate])


def test_validate_image_bytes_rejects_corrupt_data() -> None:
    with pytest.raises(ImageGenerationError, match="Invalid image format"):
        GeminiImageService.validate_and_normalize_image_bytes(b"not-an-image")


def test_image_agent_generates_and_handles_invalid_url_and_api_failure(
    sample_workbook: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    products = cleaned.vendor_products
    # Make SKU-102 have an invalid URL
    products[1].image_url = "ftp://invalid-scheme"
    products[1].image_url_valid = False

    sample_jpg = create_sample_jpeg_bytes()

    def _fake_requests_get(url, timeout=30):
        return SimpleNamespace(status_code=200, content=sample_jpg, raise_for_status=lambda: None)

    monkeypatch.setattr("services.gemini_image_service.requests.get", _fake_requests_get)

    class _ConditionalModels:
        def generate_content(self, **kwargs):
            contents = kwargs["contents"]
            prompt = contents[1]
            if "TV stand" in prompt:
                raise RuntimeError("Simulated Gemini image outage")
            part = SimpleNamespace(inline_data=SimpleNamespace(data=sample_jpg))
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    mock_client = SimpleNamespace(models=_ConditionalModels())
    svc = GeminiImageService(api_key="test-key", max_retries=1, client=mock_client)
    cfg = AppConfig(input_file=sample_workbook, image_dir=tmp_path / "images", max_concurrent_requests=1)
    agent = LifestyleImageAgent(cfg, image_service=svc)

    records = agent.run(products, image_dir=tmp_path / "images")
    rec_map = {r.sku: r for r in records}

    assert rec_map["SKU-101"].status == "completed"
    assert rec_map["SKU-101"].lifestyle_valid is True
    assert Path(rec_map["SKU-101"].original_image_path).exists()
    assert Path(rec_map["SKU-101"].lifestyle_image_path).exists()

    assert rec_map["SKU-102"].status == "failed"
    assert "invalid image url" in (rec_map["SKU-102"].error or "").lower()
    assert rec_map["SKU-102"].error_category == "invalid_url"

    assert rec_map["SKU-103"].status == "failed"
    assert rec_map["SKU-103"].original_valid is True
    assert rec_map["SKU-103"].lifestyle_valid is False
    assert rec_map["SKU-103"].error_category == "transient"


def test_image_limit_selects_first_n_valid_products(
    sample_workbook: Path, tmp_path: Path, monkeypatch
) -> None:
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    products = cleaned.vendor_products
    # Make first product invalid; image_limit=1 should skip SKU-101 and process SKU-102 only
    products[0].image_url = "not-a-url"
    products[0].image_url_valid = False

    sample_jpg = create_sample_jpeg_bytes()
    monkeypatch.setattr(
        "services.gemini_image_service.requests.get",
        lambda url, timeout=30: SimpleNamespace(
            status_code=200, content=sample_jpg, raise_for_status=lambda: None
        ),
    )
    called_prompts = []

    class _RecordingModels:
        def generate_content(self, **kwargs):
            called_prompts.append(kwargs["contents"][1])
            part = SimpleNamespace(inline_data=SimpleNamespace(data=sample_jpg))
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    svc = GeminiImageService(api_key="test-key", max_retries=0, client=SimpleNamespace(models=_RecordingModels()))
    cfg = AppConfig(
        input_file=sample_workbook,
        image_dir=tmp_path / "images",
        image_limit=1,
        max_concurrent_requests=1,
    )
    agent = LifestyleImageAgent(cfg, image_service=svc)
    records = agent.run(products, image_dir=tmp_path / "images")
    rec_map = {r.sku: r for r in records}

    assert len(called_prompts) == 1
    assert rec_map["SKU-101"].status == "skipped"
    assert rec_map["SKU-102"].status == "completed"
    assert rec_map["SKU-103"].status == "skipped"


def test_skip_images_precedence_over_image_limit(
    sample_workbook: Path, tmp_path: Path
) -> None:
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    cfg = AppConfig(
        input_file=sample_workbook,
        image_dir=tmp_path / "images",
        skip_images=True,
        image_limit=2,
        dry_run=True,
    )
    warnings = cfg.validate()
    assert any("takes precedence" in w for w in warnings)

    agent = LifestyleImageAgent(cfg)
    records = agent.run(cleaned.vendor_products, image_dir=tmp_path / "images")
    assert len(records) == 3
    assert all(r.status == "skipped" for r in records)


@pytest.mark.live
def test_live_gemini_lifestyle_image(sample_workbook: Path, tmp_path: Path) -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_TESTS=1 to run live Gemini image generation test.")
    cfg = AppConfig.from_env(input_file="test data.xlsx", image_dir=str(tmp_path / "images"))
    cfg.validate(require_input_file=True)
    cleaned = clean_workbook_data(read_workbook("test data.xlsx"))
    agent = LifestyleImageAgent(cfg)
    records = agent.run([cleaned.vendor_products[0]], image_dir=tmp_path / "images")
    assert records[0].status == "completed"
    assert records[0].lifestyle_valid is True
