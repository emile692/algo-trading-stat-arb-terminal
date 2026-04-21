"""Run Germany phase-3 validation campaign.

The campaign is intentionally narrow:
- fixed Germany reference
- fixed pair filter corr<=0.75, no retuning
- chronological OOS validation
- stress_trending diagnostic
- one simple mitigation diagnostic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.germany_phase3 import DEFAULT_END, DEFAULT_START, Phase3Options, run_germany_phase3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Germany phase-3 validation campaign.")
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
    out_dir = run_germany_phase3(
        Phase3Options(
            start=args.start,
            end=args.end,
            smoke=bool(args.smoke),
            output_root=Path(args.output_root),
            output_suffix=args.output_suffix,
            log_level=args.log_level,
        )
    )
    logging.getLogger("germany_phase3").info("Germany phase-3 campaign complete: %s", out_dir)


if __name__ == "__main__":
    main()
