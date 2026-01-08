from __future__ import annotations

import argparse
from pathlib import Path

from config.params import UNIVERSES
from utils.monthly_backtest import BatchConfig, StrategyParams
from utils.global_ranking_backtest import run_global_ranking_walkforward


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--universes", type=str, default="all",
                   help='Comma-separated list (e.g. "france,sweden") or "all"')
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--K", type=int, default=5)

    p.add_argument("--z_entry", type=float, default=2.0)
    p.add_argument("--z_exit", type=float, default=0.4)
    p.add_argument("--z_stop", type=float, default=4.0)
    p.add_argument("--z_window", type=int, default=60)
    p.add_argument("--wf_train", type=int, default=120)
    p.add_argument("--fees", type=float, default=0.0002)
    p.add_argument("--beta_mode", type=str, default="monthly")

    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    cfg = BatchConfig(
        data_path=project_root / "data" / "raw" / "d1",
        monthly_universe_path=project_root / "data" / "universe" / "france.parquet",  # just for parent dir
        out_dir=project_root / "data" / "backtests" / "global_ranking",
        universe_name="GLOBAL",
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
        wf_test=0,
        fees=args.fees,
        beta_mode=args.beta_mode,
    )

    if args.universes == "all":
        universes = UNIVERSES
    else:
        universes = [u.strip() for u in args.universes.split(",") if u.strip()]

    res = run_global_ranking_walkforward(
        cfg=cfg,
        params=params,
        universes=universes,
        top_n_candidates=args.N,
        max_positions=args.K,
    )

    if not res:
        print("No results.")
        return

    print("DONE")
    print("Stats:", res["stats"])
    print("Trades:", len(res["trades"]))
    print("Equity rows:", len(res["equity"]))


if __name__ == "__main__":
    main()
