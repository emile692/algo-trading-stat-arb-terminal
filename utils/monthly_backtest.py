from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from utils.loader import load_price_csv
from utils.metrics import compute_hedge_ratio
from utils.backtest import (
    walk_forward_beta_spread_zscore,
    backtest_pair,
    compute_metrics,
)

# =====================================================================
# CONFIG
# =====================================================================

@dataclass(frozen=True)
class StrategyParams:
    z_entry: float = 2.0
    z_exit: float = 0.4
    z_stop: float = 4.0
    z_window: int = 60

    wf_train: int = 120
    wf_test: int = 30  # utilisé uniquement en beta_mode="wf"

    fees: float = 0.0002
    beta_mode: str = "monthly"  # "monthly" | "wf"


@dataclass(frozen=True)
class BatchConfig:
    data_path: Path
    monthly_universe_path: Path
    out_dir: Path
    universe_name: Optional[str] = None
    timeframe: str = "Daily"
    warmup_extra: int = 50
    equal_weight: bool = True


# =====================================================================
# HELPERS
# =====================================================================

def _prep_pair_merged(df1: pd.DataFrame, df2: pd.DataFrame, a1: str, a2: str) -> pd.DataFrame:
    df1 = df1.copy()
    df2 = df2.copy()

    df1["log"] = np.log(df1["close"])
    df2["log"] = np.log(df2["close"])

    df1["norm"] = df1["log"] - df1["log"].iloc[0]
    df2["norm"] = df2["log"] - df2["log"].iloc[0]

    merged = pd.merge(
        df1[["datetime", "norm"]],
        df2[["datetime", "norm"]],
        on="datetime",
        how="inner",
        suffixes=(f"_{a1}", f"_{a2}"),
    )
    return merged


def _slice_with_warmup(
    merged: pd.DataFrame,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    wf_train: int,
    wf_test: int,
    z_window: int,
    warmup_extra: int,
) -> Tuple[pd.DataFrame, pd.Series]:

    merged = merged.copy()
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged[merged["datetime"] <= trade_end].reset_index(drop=True)

    if merged.empty:
        return merged, pd.Series([], dtype=bool)

    warmup = wf_train + z_window + warmup_extra
    slice_len = max(warmup + wf_test + 5, warmup)

    merged_slice = merged.iloc[-slice_len:].reset_index(drop=True)
    mask = (merged_slice["datetime"] >= trade_start) & (merged_slice["datetime"] <= trade_end)

    return merged_slice, mask


def _equity_only_in_trade_window(equity: pd.Series, mask: pd.Series) -> pd.Series:
    rets = equity.pct_change().fillna(0.0)
    rets = rets.where(mask, 0.0)
    return (1.0 + rets).cumprod()


def _zscore_rolling(spread: pd.Series, z_window: int) -> pd.Series:
    mu = spread.rolling(z_window, min_periods=z_window).mean()
    sd = spread.rolling(z_window, min_periods=z_window).std(ddof=0).replace(0.0, np.nan)
    return ((spread - mu) / sd).fillna(0.0)


def _estimate_beta_monthly(
    merged: pd.DataFrame,
    a1: str,
    a2: str,
    scan_date: pd.Timestamp,
    lookback: int,
) -> float:

    df = merged.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] <= scan_date].tail(lookback)

    if len(df) < max(30, int(0.8 * lookback)):
        raise ValueError("Not enough data for monthly beta")

    beta = compute_hedge_ratio(df[f"norm_{a1}"], df[f"norm_{a2}"])
    if not np.isfinite(beta):
        raise ValueError("Invalid beta")

    return float(beta)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =====================================================================
# CORE
# =====================================================================

def run_monthly_batch(
    cfg: BatchConfig,
    params: StrategyParams,
    trade_month: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df_mu = pd.read_parquet(cfg.monthly_universe_path)
    df_mu["trade_start"] = pd.to_datetime(df_mu["trade_start"])
    df_mu["trade_end"] = pd.to_datetime(df_mu["trade_end"])
    df_mu["scan_date"] = pd.to_datetime(df_mu["scan_date"])

    if cfg.universe_name:
        df_mu = df_mu[df_mu["universe"] == cfg.universe_name]

    if trade_month:
        df_mu = df_mu[df_mu["trade_month"] == trade_month]

    price_cache: Dict[str, pd.DataFrame] = {}

    def load(asset: str) -> pd.DataFrame:
        asset = asset.upper()
        if asset not in price_cache:
            price_cache[asset] = load_price_csv(asset, cfg.data_path)
        return price_cache[asset]

    all_metrics, all_eq, all_trades, all_pf = [], [], [], []

    for month, df_m in df_mu.groupby("trade_month"):
        pair_returns = {}

        for r in df_m.itertuples(index=False):
            a1, a2 = str(r.asset_1).upper(), str(r.asset_2).upper()
            trade_start, trade_end, scan_date = r.trade_start, r.trade_end, r.scan_date

            df1, df2 = load(a1), load(a2)
            merged_full = _prep_pair_merged(df1, df2, a1, a2)

            merged, mask = _slice_with_warmup(
                merged_full, trade_start, trade_end,
                params.wf_train, params.wf_test, params.z_window, cfg.warmup_extra
            )

            if merged.empty or mask.sum() < 5:
                continue

            y = merged[f"norm_{a1}"]
            x = merged[f"norm_{a2}"]

            beta_wf = None

            if params.beta_mode == "monthly":
                try:
                    beta = _estimate_beta_monthly(
                        merged_full, a1, a2, scan_date, params.wf_train
                    )
                except Exception:
                    continue

                spread = y - beta * x
                z = _zscore_rolling(spread, params.z_window)

            else:
                beta = compute_hedge_ratio(y, x)
                if not np.isfinite(beta):
                    continue
                spread, z, beta_wf = walk_forward_beta_spread_zscore(
                    y, x, params.wf_train, params.wf_test, params.z_window
                )

            equity, trades, _ = backtest_pair(
                spread, z, y, x,
                beta=beta,
                beta_series=beta_wf,
                z_entry=params.z_entry,
                z_exit=params.z_exit,
                z_stop=params.z_stop,
                fees=params.fees,
            )

            dt_index = pd.to_datetime(merged["datetime"])
            equity = _equity_only_in_trade_window(equity, mask)
            equity.index = dt_index

            rets = equity.pct_change().fillna(0.0)
            rets = rets.loc[(rets.index >= trade_start) & (rets.index <= trade_end)]
            if len(rets) < 3:
                continue

            pair_id = f"{a1}-{a2}|{month}"
            pair_returns[pair_id] = rets

            m = compute_metrics(equity.loc[rets.index].reset_index(drop=True), [])

            all_metrics.append({
                "trade_month": month,
                "pair_id": pair_id,
                "asset_1": a1,
                "asset_2": a2,
                "beta_mode": params.beta_mode,
                "beta_ref": float(beta),
                "Final Equity": float(m["Final Equity"]),
                "Total Return": float(m["Total Return"]),
                "Sharpe": float(m["Sharpe"]),
                "Max Drawdown": float(m["Max Drawdown"]),
                "Trades": int(m["Trades"]),
            })

            all_eq.append(pd.DataFrame({
                "datetime": equity.index,
                "trade_month": month,
                "pair_id": pair_id,
                "equity": equity.values,
            }))

            # =======================
            # TRADE JOURNAL (AUDIT)
            # =======================
            if trades:
                tr = pd.DataFrame(trades)

                entry_col = _pick_col(tr, ["entry_index", "Entry_index", "entry_idx", "EntryIdx"])
                exit_col = _pick_col(tr, ["exit_index", "Exit_index", "exit_idx", "ExitIdx"])

                if entry_col is not None and exit_col is not None:
                    ei = pd.to_numeric(tr[entry_col], errors="coerce").astype("Int64")
                    xi = pd.to_numeric(tr[exit_col], errors="coerce").astype("Int64")

                    tr["entry_datetime"] = pd.NaT
                    tr["exit_datetime"] = pd.NaT

                    m_e = ei.notna() & (ei >= 0) & (ei < len(dt_index))
                    m_x = xi.notna() & (xi >= 0) & (xi < len(dt_index))

                    tr.loc[m_e, "entry_datetime"] = dt_index.iloc[ei[m_e].astype(int)].values
                    tr.loc[m_x, "exit_datetime"] = dt_index.iloc[xi[m_x].astype(int)].values

                    tr["entry_datetime"] = pd.to_datetime(tr["entry_datetime"])
                    tr["exit_datetime"] = pd.to_datetime(tr["exit_datetime"])

                    # 🔒 HARD FILTER : le trade doit ENTRER pendant le mois
                    tr = tr[
                        (tr["entry_datetime"].notna()) &
                        (tr["entry_datetime"] >= trade_start) &
                        (tr["entry_datetime"] <= trade_end)
                    ].copy()

                    # Durées auditables
                    tr["duration_days"] = (tr["exit_datetime"] - tr["entry_datetime"]).dt.days
                    tr["duration_bars_calc"] = (
                        pd.to_numeric(tr[exit_col], errors="coerce")
                        - pd.to_numeric(tr[entry_col], errors="coerce")
                    )

                # Contexte
                tr["trade_month"] = month
                tr["pair_id"] = pair_id
                tr["asset_1"] = a1
                tr["asset_2"] = a2
                tr["beta_mode"] = params.beta_mode
                tr["beta_ref"] = float(beta)
                tr["z_entry"] = float(params.z_entry)
                tr["z_exit"] = float(params.z_exit)
                tr["z_stop"] = float(params.z_stop)
                tr["z_window"] = int(params.z_window)
                tr["wf_train"] = int(params.wf_train)
                tr["wf_test"] = int(params.wf_test)
                tr["fees"] = float(params.fees)

                if not tr.empty:
                    all_trades.append(tr)

        if pair_returns:
            ret_mat = pd.DataFrame(pair_returns)
            pf_rets = ret_mat.mean(axis=1).fillna(0.0)
            pf_eq = (1.0 + pf_rets).cumprod()

            all_pf.append(pd.DataFrame({
                "datetime": pf_eq.index,
                "trade_month": month,
                "pf_equity": pf_eq.values,
            }))

    res = {
        "pairs_metrics": pd.DataFrame(all_metrics),
        "pairs_equity": pd.concat(all_eq, ignore_index=True) if all_eq else pd.DataFrame(),
        "pairs_trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "portfolio_equity": pd.concat(all_pf, ignore_index=True) if all_pf else pd.DataFrame(),
    }

    for k, v in res.items():
        if not v.empty:
            (cfg.out_dir / f"{k}.parquet").write_bytes(v.to_parquet(index=False))

    return res
