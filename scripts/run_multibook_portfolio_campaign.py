from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.multibook_portfolio import MultibookOptions, run_multibook_portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct simple multi-book portfolios from fixed local country books: "
            "Sweden, Germany, France and Netherlands."
        )
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "experiments")
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    options = MultibookOptions(
        start=args.start,
        end=args.end,
        output_root=args.output_root,
        output_suffix=args.output_suffix,
        smoke=bool(args.smoke),
    )
    out_dir = run_multibook_portfolio(options)
    logging.info("Output directory: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

