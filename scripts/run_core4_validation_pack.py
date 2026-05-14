from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.core4_validation_pack import (
    Core4ValidationOptions,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DAILY_CACHE_DIR,
    DEFAULT_OUTPUT_ROOT,
    run_core4_validation_pack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a validation pack around the existing frozen core4 stat-arb reference. "
            "The script reuses the current frozen inputs and exports structured diagnostics."
        )
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / DEFAULT_CONFIG_PATH)
    parser.add_argument("--daily-cache-dir", type=Path, default=PROJECT_ROOT / DEFAULT_DAILY_CACHE_DIR)
    parser.add_argument("--start", default=None, help="Optional override for the analysis start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Optional override for the analysis end date (YYYY-MM-DD).")
    parser.add_argument("--rebuild-daily-cache", action="store_true", help="Force rebuild of the daily country return cache.")
    parser.add_argument("--smoke", action="store_true", help="Run the validation pack on a shorter recent window while testing the full plumbing.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed used for random weight perturbation diagnostics.")
    parser.add_argument("--random-portfolios", type=int, default=100, help="Number of random perturbed portfolios to generate.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    options = Core4ValidationOptions(
        output_root=Path(args.output_root),
        config_path=Path(args.config_path),
        daily_cache_dir=Path(args.daily_cache_dir),
        start=args.start,
        end=args.end,
        rebuild_daily_cache=bool(args.rebuild_daily_cache),
        smoke=bool(args.smoke),
        random_seed=int(args.random_seed),
        random_portfolios=int(args.random_portfolios),
    )
    output_dir = run_core4_validation_pack(options, project_root=PROJECT_ROOT)
    logging.info("Core4 validation pack output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
