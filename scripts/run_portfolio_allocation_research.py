from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.portfolio_allocation_research import AllocationResearchOptions, run_portfolio_allocation_research


DEFAULT_METHODS = (
    "inverse_vol",
    "equal_weight",
    "risk_parity",
    "mean_variance_shrunk",
    "reward_to_risk",
    "contribution_based",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward research campaign for multi-country book allocation. "
            "Weights are estimated only from trailing daily book returns."
        )
    )
    parser.add_argument("--books", nargs="+", default=["france", "germany", "netherlands", "sweden"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--rebalance-frequency", nargs="+", default=["weekly"], choices=["daily", "weekly", "monthly"])
    parser.add_argument("--lookback-days", nargs="+", type=int, default=[126])
    parser.add_argument("--allocation-methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--weight-floor", type=float, default=0.10)
    parser.add_argument("--weight-cap", type=float, default=0.40)
    parser.add_argument("--compounding-mode", choices=["compounded", "additive", "both"], default="both")
    parser.add_argument("--lag-days", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "experiments")
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "config" / "core_portfolio_reference.json")
    parser.add_argument(
        "--daily-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "core_portfolio_reference_daily_cache",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke campaign.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    books = tuple(str(x).lower() for x in args.books)
    methods = tuple(str(x).strip().lower() for x in args.allocation_methods)
    frequencies = tuple(str(x).strip().lower() for x in args.rebalance_frequency)
    lookbacks = tuple(int(x) for x in args.lookback_days)
    start = args.start
    end = args.end

    if args.smoke:
        start = "2024-01-01"
        end = "2025-12-31"
        methods = tuple(m for m in methods if m in {"inverse_vol", "equal_weight", "risk_parity"})
        frequencies = frequencies[:1] or ("weekly",)
        lookbacks = (min(lookbacks) if lookbacks else 63,)
        if lookbacks[0] > 126:
            lookbacks = (63,)

    options = AllocationResearchOptions(
        start=start,
        end=end,
        books=books,
        allocation_methods=methods,
        rebalance_frequencies=frequencies,
        lookback_days=lookbacks,
        weight_floor=float(args.weight_floor),
        weight_cap=float(args.weight_cap),
        compounding_mode=args.compounding_mode,
        lag_days=int(args.lag_days),
        output_root=args.output_root,
        output_suffix=args.output_suffix,
        smoke=bool(args.smoke),
        config_path=args.config_path,
        daily_cache_dir=args.daily_cache_dir,
    )
    out_dir = run_portfolio_allocation_research(options, project_root=PROJECT_ROOT)
    logging.info("Output directory: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
