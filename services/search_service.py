"""Configurable web search service for competitor product research."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from services.retry_service import NonRetryableError, execute_with_retry
from utils.cost_tracker import CostTracker
from utils.logger import get_logger

logger = get_logger("search_service")

PRICE_REGEX = re.compile(
    r"(?:(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.\d{1,2})?))|(?:(?:\$|USD)\s*([0-9][0-9,]*(?:\.\d{1,2})?))",
    flags=re.IGNORECASE,
)


@dataclass
class SearchHit:
    """Normalized search result returned by any configured search provider."""

    title: str
    url: str
    snippet: str
    marketplace: str
    extracted_price: Optional[float] = None
    currency: Optional[str] = None


def infer_marketplace(url: str) -> str:
    lowered = url.lower()
    if "amazon.in" in lowered or "amazon.com" in lowered:
        return "Amazon"
    if "ikea." in lowered:
        return "IKEA"
    if "pepperfry." in lowered:
        return "Pepperfry"
    if "woodenstreet." in lowered:
        return "Wooden Street"
    if "flipkart." in lowered:
        return "Flipkart"
    if "wayfair." in lowered:
        return "Wayfair"
    return "Web"


def extract_price_from_text(text: str) -> tuple[Optional[float], Optional[str]]:
    """Extract first plausible currency price and currency code from snippet/title."""
    if not text:
        return None, None
    match = PRICE_REGEX.search(text)
    if not match:
        return None, None
    inr_val, usd_val = match.group(1), match.group(2)
    raw_num = inr_val or usd_val
    currency = "INR" if inr_val else "USD"
    try:
        return float(raw_num.replace(",", "")), currency
    except ValueError:
        return None, None


class SearchService:
    """Multi-provider search client (Serper, Tavily, SerpAPI, Google CSE) with retries."""

    def __init__(
        self,
        provider: str,
        api_key: Optional[str],
        cse_id: Optional[str] = None,
        max_retries: int = 3,
        request_timeout: int = 30,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.provider = provider.strip().lower()
        self.api_key = (api_key or "").strip() or None
        self.cse_id = (cse_id or "").strip() or None
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.cost_tracker = cost_tracker

    @property
    def is_configured(self) -> bool:
        if not self.api_key:
            return False
        if self.provider == "google_cse" and not self.cse_id:
            return False
        return True

    def search(self, query: str, num_results: int = 8) -> List[SearchHit]:
        """Execute a web search using the configured provider and return normalized hits."""
        if not self.is_configured:
            raise NonRetryableError(
                f"Search provider '{self.provider}' is not configured with credentials."
            )

        def _invoke() -> List[SearchHit]:
            if self.cost_tracker:
                self.cost_tracker.record_search_request()
            if self.provider == "serper":
                return self._search_serper(query, num_results)
            if self.provider == "tavily":
                return self._search_tavily(query, num_results)
            if self.provider == "serpapi":
                return self._search_serpapi(query, num_results)
            if self.provider == "google_cse":
                return self._search_google_cse(query, num_results)
            raise NonRetryableError(f"Unsupported search provider: '{self.provider}'")

        hits, _ = execute_with_retry(
            func=_invoke,
            operation_name=f"Search ({self.provider})",
            max_retries=self.max_retries,
            cost_tracker=self.cost_tracker,
        )
        return hits

    def _to_hit(self, title: str, url: str, snippet: str, raw_price: Any = None) -> SearchHit:
        price_val: Optional[float] = None
        currency: Optional[str] = None
        if isinstance(raw_price, (int, float)):
            price_val = float(raw_price)
        elif isinstance(raw_price, str):
            price_val, currency = extract_price_from_text(raw_price)
        if price_val is None:
            price_val, currency = extract_price_from_text(f"{title} {snippet}")
        return SearchHit(
            title=title.strip(),
            url=url.strip(),
            snippet=snippet.strip(),
            marketplace=infer_marketplace(url),
            extracted_price=price_val,
            currency=currency,
        )

    def _search_serper(self, query: str, num_results: int) -> List[SearchHit]:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.api_key or "", "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=self.request_timeout,
        )
        if resp.status_code in (401, 403):
            raise NonRetryableError(f"Serper authentication failed (HTTP {resp.status_code}).")
        resp.raise_for_status()
        data = resp.json()
        hits: List[SearchHit] = []
        for item in data.get("organic", []):
            hits.append(
                self._to_hit(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    raw_price=item.get("price"),
                )
            )
        return hits

    def _search_tavily(self, query: str, num_results: int) -> List[SearchHit]:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",
            },
            timeout=self.request_timeout,
        )
        if resp.status_code in (401, 403):
            raise NonRetryableError(f"Tavily authentication failed (HTTP {resp.status_code}).")
        resp.raise_for_status()
        data = resp.json()
        return [
            self._to_hit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in data.get("results", [])
        ]

    def _search_serpapi(self, query: str, num_results: int) -> List[SearchHit]:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "api_key": self.api_key, "num": num_results, "engine": "google"},
            timeout=self.request_timeout,
        )
        if resp.status_code in (401, 403):
            raise NonRetryableError(f"SerpAPI authentication failed (HTTP {resp.status_code}).")
        resp.raise_for_status()
        data = resp.json()
        return [
            self._to_hit(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                raw_price=item.get("rich_snippet", {}).get("top", {}).get("detected_extensions", {}).get("price"),
            )
            for item in data.get("organic_results", [])
        ]

    def _search_google_cse(self, query: str, num_results: int) -> List[SearchHit]:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(10, num_results),
            },
            timeout=self.request_timeout,
        )
        if resp.status_code in (401, 403):
            raise NonRetryableError(f"Google CSE authentication failed (HTTP {resp.status_code}).")
        resp.raise_for_status()
        data = resp.json()
        return [
            self._to_hit(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
            for item in data.get("items", [])
        ]
