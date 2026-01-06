from __future__ import annotations

import numpy as np
import pandas as pd

from utils.monthly_backtest import run_monthly_batch, StrategyParams, BatchConfig


def run_global_walkforward(cfg: BatchConfig, params: StrategyParams) -> dict:

    df_mu = pd.read_parquet(cfg.monthly_universe_path)
    months = sorted(df_mu["trade_month"].unique())

    equity_chunks = []
    monthly_rows = []
    all_trades = []
    all_pairs = []

    last_equity = 1.0

    for month in months:
        res = run_monthly_batch(cfg, params, trade_month=month)

        pf = res.get("portfolio_equity")
        if pf is None or pf.empty:
            continue

        pf = pf.sort_values("datetime")
        pf["equity"] = pf["pf_equity"] * last_equity
        last_equity = pf["equity"].iloc[-1]

        equity_chunks.append(pf[["datetime", "equity"]])

        monthly_rows.append({
            "trade_month": month,
            "monthly_return": pf["pf_equity"].iloc[-1] - 1.0,
            "pairs_traded": len(res.get("pairs_metrics", [])),
            "beta_mode": params.beta_mode,
        })

        if not res.get("pairs_trades", pd.DataFrame()).empty:
            all_trades.append(res["pairs_trades"])

        if not res.get("pairs_metrics", pd.DataFrame()).empty:
            all_pairs.append(res["pairs_metrics"])

    if not equity_chunks:
        return {}

    eq = pd.concat(equity_chunks).sort_values("datetime").reset_index(drop=True)
    rets = eq["equity"].pct_change().dropna()

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    stats = {
        "Final Equity": eq["equity"].iloc[-1],
        "Sharpe": np.sqrt(252) * rets.mean() / rets.std(ddof=1) if rets.std(ddof=1) > 0 else 0.0,
        "Max Drawdown": (eq["equity"] / eq["equity"].cummax() - 1.0).min(),
        "CAGR": eq["equity"].iloc[-1] ** (252 / len(eq)) - 1.0 if len(eq) > 2 else 0.0,
        "Nb Trade": int(len(trades_df)),
    }

    return {
        "equity": eq,
        "stats": stats,
        "monthly": pd.DataFrame(monthly_rows),
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "pairs_metrics": pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(),
    }
