from __future__ import annotations

import argparse
from pathlib import Path

from utils.monthly_backtest import StrategyParams, BatchConfig, run_monthly_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--month", type=str, default=None, help='Trade month, e.g. "2025-07". Default: all months.')
    p.add_argument("--universe", type=str, default="sweden", help="Universe name (or empty for all).")
    p.add_argument("--topdir", type=str, default="data/backtests/monthly", help="Output directory.")

    # Strategy params (global, not pair-specific)
    p.add_argument("--z_entry", type=float, default=2.0)
    p.add_argument("--z_exit", type=float, default=0.4)
    p.add_argument("--z_stop", type=float, default=4.0)
    p.add_argument("--z_window", type=int, default=60)
    p.add_argument("--wf_train", type=int, default=120)
    p.add_argument("--wf_test", type=int, default=30)
    p.add_argument("--fees", type=float, default=0.0002)

    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    cfg = BatchConfig(
        data_path=project_root / "data" / "raw" / "d1",
        monthly_universe_path=project_root / "data" / "universe" / "monthly_universe.parquet",
        out_dir=project_root / args.topdir,
        universe_name=args.universe if args.universe else None,
        timeframe="Daily",
        warmup_extra=50,
        equal_weight=True,
    )

    params = StrategyParams(
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        z_stop=args.z_stop,
        z_window=args.z_window,
        wf_train=args.wf_train,
        wf_test=args.wf_test,
        fees=args.fees,
    )

    res = run_monthly_batch(cfg, params, trade_month=args.month)

    print("DONE")
    print("pairs_metrics:", len(res["pairs_metrics"]))
    print("portfolio_equity rows:", len(res["portfolio_equity"]))
    print("out_dir:", cfg.out_dir)


if __name__ == "__main__":
    main()
