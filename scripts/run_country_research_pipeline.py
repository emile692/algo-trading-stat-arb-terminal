from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.country_research_pipeline import (
    DEFAULT_END,
    DEFAULT_START,
    PipelineOptions,
    run_country_research_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized per-country stat-arb research pipeline: reference selection, "
            "edge diagnostic, local hypothesis generation, controlled ablation, robustness and decision."
        )
    )
    parser.add_argument("--country", required=True, help="Country/universe id, e.g. sweden, france, germany.")
    parser.add_argument("--start", default=DEFAULT_START, help="Backtest start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Backtest end date.")
    parser.add_argument(
        "--reference-name",
        default=None,
        help="Optional reference selector. Supported today: auto, cross_sectional, sweden_c.",
    )
    parser.add_argument("--skip-robustness", action="store_true", help="Skip temporal robustness phase.")
    parser.add_argument("--smoke", action="store_true", help="Run a short 2025-Q1 smoke campaign.")
    parser.add_argument("--rebuild-scans", action="store_true", help="Rebuild scans instead of using compatible caches.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "experiments"),
        help="Root folder where the experiment directory is created.",
    )
    parser.add_argument("--output-suffix", default=None, help="Optional suffix appended to output directory.")
    parser.add_argument(
        "--max-ablation-variants",
        type=int,
        default=5,
        help="Maximum number of variants including the reference.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    options = PipelineOptions(
        country=args.country,
        start=args.start,
        end=args.end,
        reference_name=args.reference_name,
        output_root=Path(args.output_root),
        output_suffix=args.output_suffix,
        skip_robustness=bool(args.skip_robustness),
        smoke=bool(args.smoke),
        rebuild_scans=bool(args.rebuild_scans),
        max_ablation_variants=int(args.max_ablation_variants),
    )
    out_dir = run_country_research_pipeline(options)
    logging.getLogger("country_research_pipeline").info("Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
