from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from config.params import UNIVERSES
from utils.monthly_backtest import run_monthly_batch, StrategyParams, BatchConfig


# =====================================================================
# GLOBAL WALK-FORWARD (MULTI-UNIVERSE, CAPITAL SEGMENTED)
# =====================================================================

def run_global_walkforward(
    base_cfg: BatchConfig,
    params: StrategyParams,
    universes: Optional[List[str]] = None,
    universe_weights: Optional[dict] = None,
) -> dict:

    if universes is None:
        universes = UNIVERSES

    # -----------------------------------------------------------------
    # Weights
    # -----------------------------------------------------------------
    if universe_weights is None:
        universe_weights = {u: 1.0 for u in universes}

    w_sum = sum(universe_weights.values())
    universe_weights = {u: w / w_sum for u, w in universe_weights.items()}

    universe_equities = []
    all_trades = []
    all_pairs = []
    monthly_rows = []

    # -----------------------------------------------------------------
    # Run WF per universe
    # -----------------------------------------------------------------
    for universe in universes:

        cfg = BatchConfig(
            data_path=base_cfg.data_path,
            monthly_universe_path=(
                Path(base_cfg.monthly_universe_path).parent / f"{universe}.parquet"
            ),
            out_dir=base_cfg.out_dir / universe,
            universe_name=universe,
            timeframe=base_cfg.timeframe,
            warmup_extra=base_cfg.warmup_extra,
            equal_weight=True,
        )

        print(f"\n=== GLOBAL WF | Universe: {universe} ===")

        res_u = _run_single_universe(cfg, params)
        if not res_u:
            continue

        eq_u = res_u["equity"].copy()
        eq_u["equity"] *= universe_weights[universe]
        eq_u["universe"] = universe

        universe_equities.append(eq_u)
        all_trades.append(res_u["trades"])
        all_pairs.append(res_u["pairs_metrics"])
        monthly_rows.append(res_u["monthly"].assign(universe=universe))

    if not universe_equities:
        return {}

    # =================================================================
    # 🔑 FIX CRITICAL PART: ALIGN CALENDAR + FORWARD FILL
    # =================================================================

    # Global calendar
    all_dates = sorted(
        set().union(*[df["datetime"] for df in universe_equities])
    )
    calendar = pd.DataFrame({"datetime": all_dates})

    aligned_equities = []

    for eq in universe_equities:
        tmp = calendar.merge(eq, on="datetime", how="left")
        tmp["equity"] = tmp["equity"].ffill()
        tmp["equity"] = tmp["equity"].fillna(1.0 * universe_weights[eq["universe"].iloc[0]])
        aligned_equities.append(tmp[["datetime", "equity"]])

    # Aggregate
    eq_all = aligned_equities[0].copy()
    eq_all["equity"] = 0.0

    for eq in aligned_equities:
        eq_all["equity"] += eq["equity"]

    # =================================================================
    # STATS
    # =================================================================

    rets = eq_all["equity"].pct_change().dropna()

    stats = {
        "Final Equity": eq_all["equity"].iloc[-1],
        "CAGR": (
            eq_all["equity"].iloc[-1] ** (252 / len(eq_all)) - 1.0
            if len(eq_all) > 2 else 0.0
        ),
        "Sharpe": (
            np.sqrt(252) * rets.mean() / rets.std(ddof=1)
            if rets.std(ddof=1) > 0 else 0.0
        ),
        "Max Drawdown": (eq_all["equity"] / eq_all["equity"].cummax() - 1.0).min(),
        "Nb Trades": int(sum(len(df) for df in all_trades if not df.empty)),
    }

    return {
        "equity": eq_all,
        "stats": stats,
        "monthly": pd.concat(monthly_rows, ignore_index=True),
        "trades": pd.concat(all_trades, ignore_index=True),
        "pairs_metrics": pd.concat(all_pairs, ignore_index=True),
    }


# =====================================================================
# INTERNAL: SINGLE UNIVERSE WALK-FORWARD
# =====================================================================

def _run_single_universe(cfg: BatchConfig, params: StrategyParams) -> dict:

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

    eq = (
        pd.concat(equity_chunks)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return {
        "equity": eq,
        "monthly": pd.DataFrame(monthly_rows),
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "pairs_metrics": pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(),
    }
