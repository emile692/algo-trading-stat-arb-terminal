from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.top_pf_qualification import QualificationOptions, run_top_pf_qualification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify France, Norway and Netherlands as top-portfolio country books "
            "using existing country research outputs. This script does not run a "
            "new optimization grid."
        )
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["france", "norway", "netherlands"],
        help="Countries to qualify. Defaults to france norway netherlands.",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "experiments")
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run the same aggregation with a smoke suffix.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    options = QualificationOptions(
        countries=tuple(args.countries),
        output_root=args.output_root,
        output_suffix=args.output_suffix,
        smoke=bool(args.smoke),
    )
    out_dir = run_top_pf_qualification(options)
    logging.info("Output directory: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

