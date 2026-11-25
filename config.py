"""Centralized configuration management for CommerceAI with BYOK validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


SUPPORTED_SEARCH_PROVIDERS = ("serper", "tavily", "serpapi", "google_cse")


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Optional[str], default: int, name: str, minimum: int = 1) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"Configuration error: {name} must be an integer, got '{value}'.") from exc
    if parsed < minimum:
        raise ValueError(f"Configuration error: {name} must be >= {minimum}, got {parsed}.")
    return parsed


def _parse_csv_list(value: Optional[str], default: List[str]) -> List[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def mask_secret(secret: Optional[str]) -> str:
    """Return a masked representation of an API key safe for logs."""
    if not secret:
        return "<not-set>"
    cleaned = secret.strip()
    if len(cleaned) <= 8:
        return "***"
    return f"{cleaned[:4]}...{cleaned[-4:]}"


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration loaded from environment variables and CLI flags."""

    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-2.5-flash-image"

    search_provider: str = "serper"
    search_api_key: Optional[str] = None
    search_cse_id: Optional[str] = None

    input_file: Path = Path("test data.xlsx")
    output_dir: Path = Path("outputs")
    image_dir: Path = Path("images")
    log_dir: Path = Path("logs")

    max_retries: int = 3
    max_concurrent_requests: int = 3
    request_timeout: int = 120
    competitor_product_count: int = 5
    cache_enabled: bool = True

    sales_confirmed_statuses: List[str] = field(default_factory=lambda: ["Processed"])
    sales_excluded_statuses: List[str] = field(default_factory=lambda: ["Cancelled", "Pending"])
    sales_separate_statuses: List[str] = field(default_factory=lambda: ["OJ Shipping"])

    log_level: str = "INFO"

    skip_images: bool = False
    skip_competitors: bool = False
    skip_content: bool = False
    dry_run: bool = False

    @classmethod
    def from_env(
        cls,
        env_file: Optional[Path] = None,
        input_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        skip_images: bool = False,
        skip_competitors: bool = False,
        skip_content: bool = False,
        dry_run: bool = False,
    ) -> "AppConfig":
        """Load configuration from `.env` and environment variables with optional CLI overrides."""
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)
        else:
            load_dotenv(override=False)

        raw_input = input_file or os.getenv("INPUT_FILE") or "test data.xlsx"
        raw_output = output_dir or os.getenv("OUTPUT_DIR") or "outputs"
        raw_image = image_dir or os.getenv("IMAGE_DIR") or "images"
        raw_log = os.getenv("LOG_DIR") or "logs"

        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
        search_key = (os.getenv("SEARCH_API_KEY") or "").strip() or None
        search_cse_id = (os.getenv("SEARCH_CSE_ID") or "").strip() or None
        search_provider = (os.getenv("SEARCH_PROVIDER") or "serper").strip().lower()

        return cls(
            gemini_api_key=gemini_key,
            gemini_model=(os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip(),
            gemini_image_model=(os.getenv("GEMINI_IMAGE_MODEL") or "gemini-2.5-flash-image").strip(),
            search_provider=search_provider,
            search_api_key=search_key,
            search_cse_id=search_cse_id,
            input_file=Path(raw_input),
            output_dir=Path(raw_output),
            image_dir=Path(raw_image),
            log_dir=Path(raw_log),
            max_retries=_parse_int(os.getenv("MAX_RETRIES"), 3, "MAX_RETRIES", minimum=0),
            max_concurrent_requests=_parse_int(
                os.getenv("MAX_CONCURRENT_REQUESTS"), 3, "MAX_CONCURRENT_REQUESTS", minimum=1
            ),
            request_timeout=_parse_int(os.getenv("REQUEST_TIMEOUT"), 120, "REQUEST_TIMEOUT", minimum=5),
            competitor_product_count=_parse_int(
                os.getenv("COMPETITOR_PRODUCT_COUNT"), 5, "COMPETITOR_PRODUCT_COUNT", minimum=1
            ),
            cache_enabled=_parse_bool(os.getenv("CACHE_ENABLED"), default=True),
            sales_confirmed_statuses=_parse_csv_list(
                os.getenv("SALES_CONFIRMED_STATUSES"), ["Processed"]
            ),
            sales_excluded_statuses=_parse_csv_list(
                os.getenv("SALES_EXCLUDED_STATUSES"), ["Cancelled", "Pending"]
            ),
            sales_separate_statuses=_parse_csv_list(
                os.getenv("SALES_SEPARATE_STATUSES"), ["OJ Shipping"]
            ),
            log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
            skip_images=skip_images,
            skip_competitors=skip_competitors,
            skip_content=skip_content,
            dry_run=dry_run,
        )

    def validate(self, require_input_file: bool = True) -> List[str]:
        """Validate configuration and return non-fatal warnings. Raises ValueError on fatal errors."""
        errors: List[str] = []
        warnings: List[str] = []

        if require_input_file and not self.input_file.exists():
            errors.append(f"Input workbook not found: '{self.input_file}'.")

        if self.search_provider not in SUPPORTED_SEARCH_PROVIDERS:
            errors.append(
                f"Unsupported SEARCH_PROVIDER='{self.search_provider}'. "
                f"Supported providers: {', '.join(SUPPORTED_SEARCH_PROVIDERS)}."
            )

        if not self.dry_run:
            needs_gemini = (not self.skip_content) or (not self.skip_images)
            if needs_gemini and not self.gemini_api_key:
                errors.append(
                    "Missing mandatory GEMINI_API_KEY in environment or .env file. "
                    "Set GEMINI_API_KEY to run Gemini content/image generation, or pass --dry-run."
                )
            if not self.gemini_model:
                errors.append("GEMINI_MODEL cannot be empty.")
            if not self.skip_images and not self.gemini_image_model:
                errors.append("GEMINI_IMAGE_MODEL cannot be empty when image generation is enabled.")

            if not self.skip_competitors:
                if not self.search_api_key:
                    warnings.append(
                        "SEARCH_API_KEY is not set; competitor research will be marked unavailable "
                        "and no competitor prices will be fabricated."
                    )
                elif self.search_provider == "google_cse" and not self.search_cse_id:
                    warnings.append(
                        "SEARCH_PROVIDER=google_cse requires SEARCH_CSE_ID; competitor research will "
                        "be marked unavailable until SEARCH_CSE_ID is configured."
                    )
        else:
            if not self.gemini_api_key:
                warnings.append("GEMINI_API_KEY not set (allowed in --dry-run mode).")
            if not self.search_api_key:
                warnings.append("SEARCH_API_KEY not set (allowed in --dry-run mode).")

        if errors:
            raise ValueError(" | ".join(errors))

        return warnings

    def safe_summary(self) -> dict:
        """Return a sanitized configuration dictionary with masked secrets for reporting/logging."""
        return {
            "gemini_api_key": mask_secret(self.gemini_api_key),
            "gemini_model": self.gemini_model,
            "gemini_image_model": self.gemini_image_model,
            "search_provider": self.search_provider,
            "search_api_key": mask_secret(self.search_api_key),
            "input_file": str(self.input_file),
            "output_dir": str(self.output_dir),
            "image_dir": str(self.image_dir),
            "max_retries": self.max_retries,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "competitor_product_count": self.competitor_product_count,
            "cache_enabled": self.cache_enabled,
            "sales_confirmed_statuses": self.sales_confirmed_statuses,
            "sales_excluded_statuses": self.sales_excluded_statuses,
            "sales_separate_statuses": self.sales_separate_statuses,
            "skip_images": self.skip_images,
            "skip_competitors": self.skip_competitors,
            "skip_content": self.skip_content,
            "dry_run": self.dry_run,
        }
