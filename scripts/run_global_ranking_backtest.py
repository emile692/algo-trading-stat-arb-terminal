# scripts/run_global_ranking_backtest.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root in PYTHONPATH
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from backtesting.global_loop import run_global_ranking_rebalance_walkforward
from object.class_file import BatchConfig, StrategyParams
from config.params import UNIVERSES, START_DATE, END_DATE


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Global Ranking Batch Backtest (window-based)"
    )

    p.add_argument(
        "--universes",
        type=str,
        default="all",
        help='Comma-separated list (e.g. "france,sweden") or "all"',
    )

    p.add_argument("--rebalance_period", type=str, default="daily",
                   choices=["daily", "weekly", "monthly"])

    p.add_argument("--N", type=int, default=20,
                   help="Top-N ranked candidates per scan_date")

    p.add_argument("--K", type=int, default=5,
                   help="Max concurrent positions")

    # Strategy params
    p.add_argument("--z_entry", type=float, default=2.0)
    p.add_argument("--z_exit", type=float, default=0.4)
    p.add_argument("--z_stop", type=float, default=4.0)
    p.add_argument("--z_window", type=int, default=60)
    p.add_argument("--wf_train", type=int, default=120)
    p.add_argument("--fees", type=float, default=0.0002)
    p.add_argument("--beta_mode", type=str, default="monthly")

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    # --------------------------------------------------------
    # Universes
    # --------------------------------------------------------
    if args.universes == "all":
        universes = UNIVERSES
    else:
        universes = [u.strip() for u in args.universes.split(",") if u.strip()]

    if not universes:
        raise ValueError("No universe selected.")

    # --------------------------------------------------------
    # Batch config (NO monthly universe)
    # --------------------------------------------------------
    cfg = BatchConfig(
        data_path=PROJECT_ROOT / "data" / "raw" / "d1",
        scanner_path=PROJECT_ROOT / "data" / "scanner",
        out_dir=PROJECT_ROOT / "data" / "backtests" / "global_ranking",
        universe_name="GLOBAL",
        timeframe="Daily",
        warmup_extra=50,
        equal_weight=True,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # --------------------------------------------------------
    # Strategy params
    # --------------------------------------------------------
    params = StrategyParams(
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        z_stop=args.z_stop,
        z_window=args.z_window,
        wf_train=args.wf_train,
        fees=args.fees,
        beta_mode=args.beta_mode,
        top_n_candidates=args.N,
        max_positions=args.K,
        rebalance_period=args.rebalance_period,
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    res = run_global_ranking_rebalance_walkforward(
        cfg=cfg,
        params=params,
        universes=universes,
    )

    if not res:
        print("No results.")
        return

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    print("\n================ BACKTEST DONE ================")
    for k, v in res["stats"].items():
        print(f"{k:>15}: {v}")

    print(f"\nTrades: {len(res['trades'])}")
    print(f"Equity rows: {len(res['equity'])}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
