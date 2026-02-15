# backtesting/global_loop.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from object.class_file import BatchConfig, StrategyParams
from backtesting.functions import precompute_pair_state_for_window
from backtesting.engine import run_daily_portfolio_engine


def _load_scans(cfg: BatchConfig, universes: List[str], scans: Optional[pd.DataFrame]) -> pd.DataFrame:
    if scans is not None:
        df = scans.copy()
        df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.normalize()
        if "universe" not in df.columns:
            df["universe"] = "INLINE"
        return df

    if cfg.scanner_path is None:
        raise ValueError("scanner_path must be set if scans is None.")

    scanner_dir = Path(cfg.scanner_path)
    frames = []
    for universe in universes:
        fp = scanner_dir / f"{universe}.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        if d.empty:
            continue
        d["scan_date"] = pd.to_datetime(d["scan_date"]).dt.normalize()
        d["universe"] = universe
        frames.append(d)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_global_ranking_daily_portfolio(
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    scans: Optional[pd.DataFrame] = None,
) -> Dict:

    scans_df = _load_scans(cfg, universes, scans)
    if scans_df.empty:
        return {}

    start = pd.to_datetime(cfg.start_date).normalize()
    end = pd.to_datetime(cfg.end_date).normalize()

    # keep a small buffer for exec_lag
    scans_df = scans_df[(scans_df["scan_date"] >= start - BDay(5)) & (scans_df["scan_date"] <= end)].copy()
    if scans_df.empty:
        return {}

    scans_by_date = {d: g for d, g in scans_df.groupby("scan_date", sort=True)}

    def get_ranked_pairs(dt: pd.Timestamp) -> List[Tuple[str, str]]:
        scan_date = (dt - BDay(params.exec_lag_days)).normalize()
        df_day = scans_by_date.get(scan_date)
        if df_day is None or df_day.empty:
            return []

        eligible = df_day[df_day["eligibility"] == "ELIGIBLE"]
        if eligible.empty:
            return []

        ranked = (eligible.sort_values("eligibility_score", ascending=False)
                         .head(params.top_n_candidates))

        return list(zip(ranked["asset_1"].astype(str).str.upper(),
                        ranked["asset_2"].astype(str).str.upper()))

    def get_pair_state(dt: pd.Timestamp, pairs: List[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
        cand_df = pd.DataFrame([{"asset_1": a1, "asset_2": a2} for (a1, a2) in pairs])
        # state as-of dt (warmup inside function)
        return precompute_pair_state_for_window(
            cfg=cfg,
            params=params,
            candidates=cand_df,
            start=dt,
            end=dt,
        )

    res = run_daily_portfolio_engine(
        params=params,
        start=start,
        end=end,
        get_ranked_pairs=get_ranked_pairs,
        get_pair_state=get_pair_state,
    )

    if not res:
        return {}

    equity = res["equity"].copy()
    trades = res["trades"].copy()

    # reporting / stats
    equity["trade_month"] = pd.to_datetime(equity["datetime"]).dt.strftime("%Y-%m")

    monthly = (
        equity.groupby("trade_month", as_index=False)
              .agg(
                  start_equity=("equity", "first"),
                  end_equity=("equity", "last"),
                  n_days=("equity", "size"),
                  max_open_positions=("n_open_positions", "max"),
              )
    )
    monthly["month_return"] = monthly["end_equity"] / monthly["start_equity"] - 1.0

    returns = equity["equity"].pct_change().dropna()
    final_eq = float(equity["equity"].iloc[-1])

    n = len(equity)
    cagr = (final_eq ** (252 / n) - 1.0) if n > 0 else np.nan
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(np.sqrt(252) * returns.mean() / vol) if (vol is not None and vol > 0) else np.nan
    mdd = float((equity["equity"] / equity["equity"].cummax() - 1).min())

    stats = {
        "Final Equity": round(final_eq, 2),
        "CAGR": round(float(cagr), 3) if not np.isnan(cagr) else np.nan,
        "Sharpe": round(float(sharpe), 2) if not np.isnan(sharpe) else np.nan,
        "Max Drawdown": round(mdd, 3),
        "Nb Trades": int(len(trades)) if isinstance(trades, pd.DataFrame) else 0,
    }

    return {"equity": equity, "monthly": monthly, "trades": trades, "stats": stats}
