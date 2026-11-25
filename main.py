"""CommerceAI CLI entrypoint.

Example usage:
    python main.py --input "test data.xlsx"
    python main.py --input "test data.xlsx" --dry-run
    python main.py --input "test data.xlsx" --skip-images --skip-competitors
"""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import List, Optional

from config import AppConfig
from utils.cost_tracker import CostTracker
from utils.logger import setup_logging
from workflow.graph import run_commerce_workflow


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="commerce-ai",
        description="CommerceAI: Self-hosted LangGraph agent for catalog content, lifestyle images, competitor research, and sales analytics.",
    )
    parser.add_argument(
        "--input",
        dest="input_file",
        default=None,
        help="Path to the input Excel workbook (defaults to INPUT_FILE env var or 'test data.xlsx').",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory for generated Excel/JSON/PNG outputs (defaults to OUTPUT_DIR or 'outputs').",
    )
    parser.add_argument(
        "--image-dir",
        dest="image_dir",
        default=None,
        help="Directory for original and lifestyle images (defaults to IMAGE_DIR or 'images').",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip Gemini lifestyle image generation (takes precedence over --image-limit).",
    )
    parser.add_argument(
        "--image-limit",
        dest="image_limit",
        type=int,
        default=None,
        help="Generate lifestyle images for only the first N valid products (default: all products).",
    )
    parser.add_argument(
        "--skip-competitors",
        action="store_true",
        help="Skip competitor research.",
    )
    parser.add_argument(
        "--skip-content",
        action="store_true",
        help="Skip Gemini product content generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, ingest & clean workbook, and run deterministic sales analytics without external API calls.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reuse of cached per-SKU content/image results.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_id = uuid.uuid4().hex[:10]

    try:
        config = AppConfig.from_env(
            input_file=args.input_file,
            output_dir=args.output_dir,
            image_dir=args.image_dir,
            skip_images=args.skip_images,
            skip_competitors=args.skip_competitors,
            skip_content=args.skip_content,
            dry_run=args.dry_run,
            image_limit=args.image_limit,
        )
        if args.no_cache:
            config = AppConfig(**{**config.__dict__, "cache_enabled": False})
        config_warnings = config.validate(require_input_file=True)
    except ValueError as exc:
        print(f"[ERROR] Configuration / Input validation failed: {exc}", file=sys.stderr)
        return 1

    logger = setup_logging(config.log_dir, log_level=config.log_level, run_id=run_id)
    logger.info("Starting CommerceAI run_id=%s | config=%s", run_id, config.safe_summary())
    for warn in config_warnings:
        logger.warning(warn)

    print(f"=== CommerceAI Pipeline (run_id={run_id}) ===")
    print(f"Input workbook : {config.input_file}")
    print(f"Output dir     : {config.output_dir}")
    print(f"Mode           : {'DRY RUN (offline validation + deterministic analytics)' if config.dry_run else 'FULL EXECUTION'}")

    cost_tracker = CostTracker()
    final_state = run_commerce_workflow(config=config, run_id=run_id, cost_tracker=cost_tracker)

    report_raw = final_state.get("execution_report")
    if report_raw is None:
        print("[ERROR] Pipeline terminated without generating an execution report.", file=sys.stderr)
        return 1

    from schemas.execution_schema import ExecutionReport
    report = (
        report_raw
        if isinstance(report_raw, ExecutionReport)
        else ExecutionReport.model_validate(report_raw)
    )

    print("\n=== Execution Summary ===")
    print(f"Status                : {report.final_status}")
    print(f"Duration              : {report.total_duration_seconds:.2f}s")
    print(f"Vendor SKUs Processed : {report.products_processed}/{report.products_total}")
    print(f"Content Generated     : {report.product_content_generated} (failed: {report.product_content_failed})")
    print(f"Images Processed      : {report.images_processed} (generated: {report.images_generated}, failed: {report.images_failed}, skipped: {report.images_skipped})")
    print(f"Competitor Research   : {report.competitor_comparisons_completed} completed (failed: {report.competitor_comparisons_failed})")
    print(f"Sales Lines Processed : {report.sales_records_processed}")
    print(f"Tokens / Est. Cost    : {report.token_usage.get('total_tokens', 0)} tokens / ${report.estimated_api_cost_usd:.4f}")
    print(f"Retries / API Fails   : {report.retry_counts} / {report.api_failures}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")
    if report.errors:
        print("\nErrors:")
        for e in report.errors:
            print(f"  - {e}")

    if report.final_status == "failed":
        return 1
    if report.final_status == "completed_with_partial_failures":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
