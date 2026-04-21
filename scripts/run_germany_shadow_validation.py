"""Run the Germany mitigated shadow-validation campaign.

The campaign reuses existing Germany core-entry and multibook outputs.
It does not create new variants, retune thresholds, or touch the backtest engine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.germany_shadow_validation import ShadowValidationOptions, run_germany_shadow_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Germany mitigated shadow validation.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "experiments")
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--germany-core-entry-dir", type=Path, default=None)
    parser.add_argument("--multibook-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Restrict readout to the 2025 window.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir = run_germany_shadow_validation(
        ShadowValidationOptions(
            output_root=args.output_root,
            output_suffix=args.output_suffix,
            germany_core_entry_dir=args.germany_core_entry_dir,
            multibook_dir=args.multibook_dir,
            smoke=bool(args.smoke),
        )
    )
    logging.info("Output directory: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
