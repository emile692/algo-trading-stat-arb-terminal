from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.core4_daily_reporting import Core4ReportingOptions, DEFAULT_OUTPUT_DIR, run_core4_daily_reporting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Industrialized daily reporting pipeline for the frozen core 4 country portfolio. "
            "Generates stable CSV, PNG, HTML and metadata outputs."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "config" / "core_portfolio_reference.json")
    parser.add_argument(
        "--daily-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "core_portfolio_reference_daily_cache",
    )
    parser.add_argument("--start", default=None, help="Optional override for the reporting start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Optional override for the reporting end date (YYYY-MM-DD).")
    parser.add_argument("--lag-days", type=int, default=1)
    parser.add_argument("--skip-risk-parity", action="store_true", help="Only report reference allocator and equal-weight benchmark.")
    parser.add_argument("--rebuild-daily-cache", action="store_true", help="Force rebuild of daily book returns instead of reusing cache.")
    parser.add_argument("--smoke", action="store_true", help="Run a shorter smoke-report window on the latest subset.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    output_dir = Path(args.output_dir)
    if args.smoke and output_dir == PROJECT_ROOT / DEFAULT_OUTPUT_DIR:
        output_dir = PROJECT_ROOT / "data" / "reports" / "core4_daily_reporting_smoke"

    options = Core4ReportingOptions(
        output_dir=output_dir,
        config_path=Path(args.config_path),
        daily_cache_dir=Path(args.daily_cache_dir),
        start=args.start,
        end=args.end,
        lag_days=int(args.lag_days),
        include_optional_allocators=not bool(args.skip_risk_parity),
        rebuild_daily_cache=bool(args.rebuild_daily_cache),
        smoke=bool(args.smoke),
    )
    out_dir = run_core4_daily_reporting(options, project_root=PROJECT_ROOT)
    logging.info("Core 4 daily reporting output directory: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
