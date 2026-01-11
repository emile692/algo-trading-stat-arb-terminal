# utils/global_ranking_backtest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.monthly_backtest import BatchConfig, StrategyParams


# ============================================================
# Helpers — Market data & statistics (NO trading logic)
# ============================================================

def _read_price_csv(data_path: Path, asset: str) -> pd.DataFrame:
    fp = data_path / f"{asset}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing price file: {fp}")

    df = pd.read_csv(fp)
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{fp} must have columns: datetime, close")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna(subset=["close"])

    df["log_close"] = np.log(df["close"].astype(float))
    df["log_norm"] = df["log_close"] - df["log_close"].iloc[0]

    return df[["datetime", "close", "log_norm"]]


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 3:
        return 1.0
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return 1.0
    return float(np.dot(x, y - y.mean()) / denom)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=1)
    z = (s - m) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def _month_bounds(df_month: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.to_datetime(df_month["trade_start"].iloc[0]).normalize(),
        pd.to_datetime(df_month["trade_end"].iloc[0]).normalize(),
    )


def _pair_id(a1: str, a2: str, trade_month: str) -> str:
    return f"{a1}__{a2}__{trade_month}"


# ============================================================
# Core trading object
# ============================================================

@dataclass
class Position:
    pair_id: str
    asset_1: str
    asset_2: str
    side: str  # LONG_SPREAD / SHORT_SPREAD
    beta: float
    entry_datetime: pd.Timestamp
    entry_spread: float
    entry_z: float
    entry_y: float
    entry_x: float


# ============================================================
# Candidate pool (monthly, cross-universe)
# ============================================================

def build_global_month_candidates(
    monthly_universe_dir: Path,
    trade_month: str,
    universes: List[str],
    top_n: int,
) -> pd.DataFrame:

    frames = []
    for u in universes:
        fp = monthly_universe_dir / f"{u}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df = df[df["trade_month"] == trade_month]
        if df.empty:
            continue
        df["universe"] = u
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["a_min"] = df[["asset_1", "asset_2"]].min(axis=1)
    df["a_max"] = df[["asset_1", "asset_2"]].max(axis=1)

    df = (
        df.sort_values("eligibility_score", ascending=False)
        .drop_duplicates(["a_min", "a_max", "trade_month"])
        .head(top_n)
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Pair state precomputation (beta, spread, z-score)
# ============================================================

def precompute_pair_state_for_month(
    cfg: BatchConfig,
    params: StrategyParams,
    candidates: pd.DataFrame,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:

    warmup = int(max(params.z_window, params.wf_train) + 10)
    start_load = trade_start - pd.Timedelta(days=warmup)

    state = {}
    trade_month = str(candidates["trade_month"].iloc[0])

    for _, r in candidates.iterrows():
        a1, a2 = r["asset_1"].upper(), r["asset_2"].upper()
        pid = _pair_id(a1, a2, trade_month)

        try:
            df1 = _read_price_csv(Path(cfg.data_path), a1)
            df2 = _read_price_csv(Path(cfg.data_path), a2)
        except Exception:
            continue

        df = df1.merge(df2, on="datetime", suffixes=("_1", "_2"))
        df = df[(df["datetime"] >= start_load) & (df["datetime"] <= trade_end)]
        df = df.sort_values("datetime")

        if len(df) < params.z_window + 5:
            continue

        df["y"] = df["log_norm_1"]
        df["x"] = df["log_norm_2"]

        beta_df = df[df["datetime"] < trade_start].tail(params.wf_train)
        if len(beta_df) < 20:
            continue

        beta = _ols_beta(beta_df["y"].values, beta_df["x"].values)
        df["beta"] = beta
        df["spread"] = df["y"] - beta * df["x"]
        df["z"] = _rolling_zscore(df["spread"], params.z_window)

        df["eligible_today"] = (
            (~df["z"].isna())
            & (df["datetime"] >= trade_start)
            & (df["datetime"] <= trade_end)
        )

        df["pair_id"] = pid
        df["asset_1"] = a1
        df["asset_2"] = a2

        state[pid] = df.set_index("datetime")[
            ["pair_id", "asset_1", "asset_2", "y", "x", "beta", "spread", "z", "eligible_today"]
        ]

    return state


# ============================================================
# MONTHLY BACKTEST — Capital constrained, ranking for ENTRY ONLY
# ============================================================

def backtest_month_global_ranking(
    pair_state: dict,
    candidates: pd.DataFrame,
    params: StrategyParams,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    K: int,
) -> dict:
    """
    Monthly global ranking backtest.
    - Equal-weight capital allocation
    - Max K concurrent positions
    - Trades recorded explicitly
    """

    # ------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------
    all_dates = sorted(set().union(*[df.index for df in pair_state.values()]))
    all_dates = [d for d in all_dates if trade_start <= d <= trade_end]

    if not all_dates:
        return {}

    trade_month = str(candidates["trade_month"].iloc[0])

    equity = 1.0
    prev_dt = None

    open_positions = {}
    equity_rows = []
    trades = []

    # ------------------------------------------------------------
    # Helper to find open trade row
    # ------------------------------------------------------------
    def _find_trade_idx(pair_id, entry_dt):
        for i in range(len(trades) - 1, -1, -1):
            if (
                trades[i]["pair_id"] == pair_id
                and trades[i]["entry_datetime"] == entry_dt
            ):
                return i
        return None

    # ------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------
    for dt in all_dates:

        # ========================================================
        # 0) DAILY MTM (equal-weight)
        # ========================================================
        if prev_dt and open_positions:
            rets = []
            for pid, pos in open_positions.items():
                df = pair_state[pid]
                if prev_dt not in df.index or dt not in df.index:
                    continue

                dY = df.loc[dt, "y"] - df.loc[prev_dt, "y"]
                dX = df.loc[dt, "x"] - df.loc[prev_dt, "x"]
                sign = 1.0 if pos.side == "LONG_SPREAD" else -1.0

                rets.append(sign * (dY - pos.beta * dX))

            if rets:
                equity *= (1.0 + np.mean(rets))

        # ========================================================
        # 1) EXITS
        # ========================================================
        to_close = []

        for pid, pos in open_positions.items():
            df = pair_state[pid]
            if dt not in df.index:
                continue

            z = df.loc[dt, "z"]

            exit_tp = (z > -params.z_exit) if pos.side == "LONG_SPREAD" else (z < params.z_exit)
            exit_sl = abs(z) >= params.z_stop
            exit_tm = dt >= trade_end

            if exit_tp or exit_sl or exit_tm:
                reason = "TP" if exit_tp else ("SL" if exit_sl else "TIME")

                idx = _find_trade_idx(pid, pos.entry_datetime)
                if idx is not None:
                    entry_spread = pos.entry_spread
                    exit_spread = float(df.loc[dt, "spread"])
                    sign = 1.0 if pos.side == "LONG_SPREAD" else -1.0

                    trades[idx].update({
                        "exit_datetime": dt,
                        "exit_z": float(z),
                        "exit_spread": exit_spread,
                        "pnl_spread": sign * (exit_spread - entry_spread),
                        "reason": reason,
                        "duration_days": int((dt.normalize() - pos.entry_datetime.normalize()).days),
                    })

                to_close.append(pid)

        for pid in to_close:
            del open_positions[pid]

        # ========================================================
        # 2) ENTRIES (ranking-based)
        # ========================================================
        slots = max(0, K - len(open_positions))

        if slots > 0:
            candidates_today = []

            for pid, df in pair_state.items():
                if pid in open_positions or dt not in df.index:
                    continue
                if not bool(df.loc[dt, "eligible_today"]):
                    continue

                z = df.loc[dt, "z"]
                if pd.isna(z):
                    continue

                if z <= -params.z_entry:
                    side = "LONG_SPREAD"
                elif z >= params.z_entry:
                    side = "SHORT_SPREAD"
                else:
                    continue

                candidates_today.append((abs(float(z)), pid, side))

            for _, pid, side in sorted(candidates_today, reverse=True)[:slots]:
                df = pair_state[pid]

                open_positions[pid] = Position(
                    pair_id=pid,
                    asset_1=str(df.loc[dt, "asset_1"]),
                    asset_2=str(df.loc[dt, "asset_2"]),
                    side=side,
                    beta=float(df.loc[dt, "beta"]),
                    entry_datetime=dt,
                    entry_spread=float(df.loc[dt, "spread"]),
                    entry_z=float(df.loc[dt, "z"]),
                    entry_y=float(df.loc[dt, "y"]),
                    entry_x=float(df.loc[dt, "x"]),
                )

                trades.append({
                    "trade_month": trade_month,
                    "pair_id": pid,
                    "asset_1": str(df.loc[dt, "asset_1"]),
                    "asset_2": str(df.loc[dt, "asset_2"]),
                    "side": side,
                    "beta": float(df.loc[dt, "beta"]),
                    "entry_datetime": dt,
                    "entry_z": float(df.loc[dt, "z"]),
                    "entry_spread": float(df.loc[dt, "spread"]),
                    "exit_datetime": pd.NaT,
                    "exit_z": np.nan,
                    "exit_spread": np.nan,
                    "pnl_spread": np.nan,
                    "reason": None,
                    "duration_days": np.nan,
                })

        # ========================================================
        # 3) RECORD EQUITY
        # ========================================================
        equity_rows.append({
            "trade_month": trade_month,
            "datetime": dt,
            "equity": equity,
            "n_open_positions": len(open_positions),
        })

        prev_dt = dt

    return {
        "equity": pd.DataFrame(equity_rows),
        "trades": pd.DataFrame(trades),
    }


# ============================================================
# WALK-FORWARD ORCHESTRATION
# ============================================================

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

        res = backtest_month_global_ranking(
            pair_state, candidates, params, trade_start, trade_end, max_positions
        )
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
        "Final Equity": final_eq,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Nb Trades": int(len(trades_df)) if isinstance(trades_df, pd.DataFrame) else 0,
    }

    return {
        "equity": eq,
        "stats": stats,
        "monthly": monthly,
        "trades": trades_df,
    }


