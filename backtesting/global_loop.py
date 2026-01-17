# backtesting/global_loop.py

# ============================================================
# WALK-FORWARD ORCHESTRATION
# ============================================================

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from backtesting.functions import precompute_pair_state_for_month
from backtesting.helpers import _month_bounds
from backtesting.monthly_loop import backtest_month_global_ranking
from object.class_file import  BatchConfig, StrategyParams
from utils.functions import build_global_month_candidates


def run_global_ranking_walkforward(
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    top_n_candidates: int,
    max_positions: int,
) -> dict:

    monthly_universe_dir = Path(cfg.monthly_universe_path).parent

    months = set()
    for u in universes:
        fp = monthly_universe_dir / f"{u}.parquet"
        if fp.exists():
            months |= set(pd.read_parquet(fp, columns=["trade_month"])["trade_month"])

    months = sorted(months)
    if not months:
        return {}

    eq_chunks = []
    trades_chunks = []
    last_equity = 1.0

    for m in months:
        candidates = build_global_month_candidates(monthly_universe_dir, m, universes, top_n_candidates)
        if candidates.empty:
            continue

        trade_start, trade_end = _month_bounds(candidates)
        pair_state = precompute_pair_state_for_month(cfg, params, candidates, trade_start, trade_end)

        res = backtest_month_global_ranking(pair_state, candidates, params, trade_start, trade_end, max_positions)
        if not res:
            continue

        eq_m = res["equity"].copy()
        eq_m["equity"] *= last_equity
        last_equity = float(eq_m["equity"].iloc[-1])
        eq_chunks.append(eq_m)

        tr_m = res.get("trades", pd.DataFrame())
        if isinstance(tr_m, pd.DataFrame) and not tr_m.empty:
            trades_chunks.append(tr_m)

    if not eq_chunks:
        return {}

    eq = pd.concat(eq_chunks, ignore_index=True)

    trades_df = pd.concat(trades_chunks, ignore_index=True) if trades_chunks else pd.DataFrame()

    # ---- monthly breakdown
    monthly = (
        eq.groupby("trade_month", as_index=False)
          .agg(
              start_equity=("equity", "first"),
              end_equity=("equity", "last"),
              n_days=("equity", "size"),
              max_open_positions=("n_open_positions", "max"),
          )
    )
    monthly["month_return"] = monthly["end_equity"] / monthly["start_equity"] - 1.0

    if not trades_df.empty:
        tcount = trades_df.groupby("trade_month").size().rename("n_trades").reset_index()
        monthly = monthly.merge(tcount, on="trade_month", how="left")
    else:
        monthly["n_trades"] = 0

    # ---- stats
    rets = eq["equity"].pct_change().dropna()
    final_eq = float(eq["equity"].iloc[-1]) if not eq.empty else np.nan
    n = len(eq)
    cagr = (final_eq ** (252 / n) - 1.0) if (n > 0 and np.isfinite(final_eq) and final_eq > 0) else np.nan
    std = rets.std(ddof=1)
    sharpe = (np.sqrt(252) * rets.mean() / std) if (std is not None and np.isfinite(std) and std > 0) else np.nan
    mdd = (eq["equity"] / eq["equity"].cummax() - 1).min() if n > 0 else np.nan

    stats = {
        "Final Equity": float(round(final_eq, 2)),
        "CAGR": float(round(cagr, 2)),
        "Sharpe": float(round(sharpe, 2)),
        "Max Drawdown": float(round(mdd, 2)),
        "Nb Trades": int(len(trades_df)) if isinstance(trades_df, pd.DataFrame) else 0,
    }

    return {
        "equity": eq,
        "stats": stats,
        "monthly": monthly,
        "trades": trades_df,
    }

