"""Run the final Germany core-entry decision campaign.

This script compares only:
- Germany reference
- fixed pair filter corr<=0.75
- fixed scan-time stress_trending bypass mitigation

It does not retune thresholds and does not modify the backtest engine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.germany_core_entry import DEFAULT_END, DEFAULT_START, GermanyCoreEntryOptions, run_germany_core_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Germany final core-entry campaign.")
    parser.add_argument("--start", default=DEFAULT_START, help="Backtest start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Backtest end date.")
    parser.add_argument("--smoke", action="store_true", help="Run a short 2024-H1 smoke campaign.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "experiments"),
        help="Root folder where the experiment directory is created.",
    )
    parser.add_argument("--output-suffix", default=None, help="Optional suffix appended to output directory.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    out_dir = run_germany_core_entry(
        GermanyCoreEntryOptions(
            start=args.start,
            end=args.end,
            output_root=Path(args.output_root),
            output_suffix=args.output_suffix,
            smoke=bool(args.smoke),
        )
    )
    logging.getLogger("germany_core_entry").info("Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
