"""Tests for competitor selection, reference price matching, and search API handling."""

from __future__ import annotations

from pathlib import Path

from agents.competitor_agent import (
    CompetitorAnalysisAgent,
    build_sales_price_catalog,
    match_reference_price,
)
from config import AppConfig
from processors.data_cleaner import clean_workbook_data
from processors.excel_reader import read_workbook
from processors.vendor_processor import select_representative_products
from services.search_service import SearchHit


def test_reference_price_matching_accepts_exact_and_rejects_weak(sample_workbook: Path) -> None:
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    catalog = build_sales_price_catalog(cleaned.sales_df)

    # Disjoint SKU -> weak match rejected
    weak = match_reference_price(cleaned.vendor_products[0], catalog)
    assert weak.accepted is False
    assert weak.our_reference_price is None

    # Exact SKU match -> accepted with median price
    matching_product = cleaned.vendor_products[0].model_copy(update={"sku": "SL-9001"})
    exact = match_reference_price(matching_product, catalog)
    assert exact.accepted is True
    assert exact.our_reference_price == 150.0
    assert exact.match_confidence == 1.0


def test_competitor_agent_unconfigured_search_does_not_fabricate(sample_workbook: Path) -> None:
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    cfg = AppConfig(input_file=sample_workbook, search_api_key=None, competitor_product_count=3)
    agent = CompetitorAnalysisAgent(cfg)

    res = agent.run(cleaned.vendor_products, cleaned.sales_df)
    assert res.search_configured is False
    assert len(res.comparisons) == 3
    assert all(c.status == "unavailable_unconfigured" for c in res.comparisons)
    assert all(c.competitor_price is None for c in res.comparisons)


def test_competitor_agent_handles_search_failure_and_missing_price(sample_workbook: Path) -> None:
    cleaned = clean_workbook_data(read_workbook(sample_workbook))

    class _MockSearchService:
        is_configured = True

        def search(self, query: str, num_results: int = 8):
            if "coffee table" in query.lower():
                raise RuntimeError("Search API 503 unavailable")
            if "tv stand" in query.lower():
                return [
                    SearchHit(
                        title="Modern TV Stand Media Cabinet with Fireplace",
                        url="https://www.pepperfry.com/tv-stand-123",
                        snippet="Stylish engineered wood TV stand without price listed",
                        marketplace="Pepperfry",
                        extracted_price=None,
                        currency=None,
                    )
                ]
            return [
                SearchHit(
                    title="Black Power Reclining Chair with Cup Holders",
                    url="https://www.amazon.in/dp/B00TEST",
                    snippet="Power recliner chair in black faux leather",
                    marketplace="Amazon",
                    extracted_price=499.0,
                    currency="USD",
                )
            ]

    cfg = AppConfig(input_file=sample_workbook, search_api_key="dummy-search-key", competitor_product_count=3)
    agent = CompetitorAnalysisAgent(cfg, search_service=_MockSearchService())
    res = agent.run(cleaned.vendor_products, cleaned.sales_df)

    by_sku = {c.our_sku: c for c in res.comparisons}
    assert by_sku["SKU-101"].status == "completed"
    assert by_sku["SKU-101"].competitor_price == 499.0
    assert by_sku["SKU-101"].source_url == "https://www.amazon.in/dp/B00TEST"
    # Reference price was unavailable, so price_difference must remain None
    assert by_sku["SKU-101"].price_difference is None

    assert by_sku["SKU-102"].status == "failed"
    assert by_sku["SKU-103"].status == "price_unavailable"
    assert by_sku["SKU-103"].competitor_price is None


def test_select_representative_products_handles_new_products(sample_workbook: Path) -> None:
    cleaned = clean_workbook_data(read_workbook(sample_workbook))
    selected = select_representative_products(cleaned.vendor_products, count=2)
    assert len(selected) == 2
    assert len({p.sku for p in selected}) == 2
