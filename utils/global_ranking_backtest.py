# utils/global_ranking_backtest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.monthly_backtest import BatchConfig, StrategyParams


# ============================================================
# Helpers
# ============================================================

def _read_price_csv(data_path: Path, asset: str) -> pd.DataFrame:
    """
    Minimal loader:
    expects CSV with at least: datetime, close
    """
    fp = data_path / f"{asset}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing price file: {fp}")

    df = pd.read_csv(fp)
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{fp} must have columns: datetime, close")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").dropna(subset=["close"])

    # log price + log-normalized (start at 0)
    df["log_close"] = np.log(df["close"].astype(float))
    df["log_norm"] = df["log_close"] - df["log_close"].iloc[0]

    return df[["datetime", "close", "log_norm"]]


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    """
    OLS slope of y on x with intercept:
    beta = cov(x,y)/var(x)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return 1.0

    x_ = x - x.mean()
    denom = np.dot(x_, x_)
    if denom <= 0:
        return 1.0
    return float(np.dot(x_, y - y.mean()) / denom)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=1)
    z = (s - m) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def _month_bounds(df_month: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    trade_start = pd.to_datetime(df_month["trade_start"].iloc[0])
    trade_end = pd.to_datetime(df_month["trade_end"].iloc[0])
    return trade_start.normalize(), trade_end.normalize()


def _pair_id(a1: str, a2: str, trade_month: str) -> str:
    return f"{a1}__{a2}__{trade_month}"


# ============================================================
# Core objects
# ============================================================

@dataclass
class Position:
    pair_id: str
    asset_1: str
    asset_2: str
    side: str  # "LONG_SPREAD" or "SHORT_SPREAD"
    beta: float

    entry_datetime: pd.Timestamp
    entry_spread: float
    entry_z: float

    entry_y: float
    entry_x: float

    is_open: bool = True


# ============================================================
# Candidate pool builder (cross-univers)
# ============================================================

def build_global_month_candidates(
    monthly_universe_dir: Path,
    trade_month: str,
    universes: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Reads data/universe/{universe}.parquet for all universes,
    keeps rows for trade_month, concatenates, sorts by eligibility_score,
    returns Top-N candidates.
    """
    frames = []
    for u in universes:
        fp = monthly_universe_dir / f"{u}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if df.empty:
            continue
        df = df[df["trade_month"] == trade_month].copy()
        if df.empty:
            continue
        if "universe" not in df.columns:
            df["universe"] = u
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    # Deduplicate within month (same pair across multiple universes)
    df_all["a_min"] = df_all[["asset_1", "asset_2"]].min(axis=1)
    df_all["a_max"] = df_all[["asset_1", "asset_2"]].max(axis=1)
    df_all = (
        df_all.sort_values("eligibility_score", ascending=False)
        .drop_duplicates(subset=["a_min", "a_max", "trade_month"], keep="first")
        .drop(columns=["a_min", "a_max"])
    )

    df_all = df_all.sort_values("eligibility_score", ascending=False).head(top_n).copy()
    df_all.reset_index(drop=True, inplace=True)
    return df_all


# ============================================================
# Pair precompute for a month (beta fixed, z rolling)
# ============================================================

def precompute_pair_state_for_month(
    cfg: BatchConfig,
    params: StrategyParams,
    candidates: pd.DataFrame,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    For each candidate pair:
    - load prices up to trade_end with warmup
    - compute beta using wf_train lookback ending at trade_start-1
    - compute spread = y - beta*x  (y/x are log-normalized)
    - compute zscore rolling z_window
    Returns dict pair_id -> df indexed by datetime with:
      y, x, spread, z, eligible_today
    """
    if candidates.empty:
        return {}

    warmup_days = int(max(params.z_window, params.wf_train) + 10)
    start_load = trade_start - pd.Timedelta(days=warmup_days)

    out: Dict[str, pd.DataFrame] = {}
    trade_month = str(candidates["trade_month"].iloc[0])

    for _, row in candidates.iterrows():
        a1 = str(row["asset_1"]).upper()
        a2 = str(row["asset_2"]).upper()
        pid = _pair_id(a1, a2, trade_month)

        try:
            df1 = _read_price_csv(Path(cfg.data_path), a1)
            df2 = _read_price_csv(Path(cfg.data_path), a2)
        except Exception:
            continue

        df = df1.merge(df2, on="datetime", how="inner", suffixes=("_1", "_2"))
        df = df[(df["datetime"] >= start_load) & (df["datetime"] <= trade_end)].copy()
        df.sort_values("datetime", inplace=True)

        if len(df) < (int(params.z_window) + 5):
            continue

        df["y"] = df["log_norm_1"].astype(float)
        df["x"] = df["log_norm_2"].astype(float)

        df_beta = df[df["datetime"] < trade_start].copy()
        if len(df_beta) < max(20, int(params.wf_train) // 2):
            continue
        df_beta = df_beta.tail(int(params.wf_train))

        beta = _ols_beta(df_beta["y"].values, df_beta["x"].values)
        if not np.isfinite(beta):
            continue

        df["beta"] = float(beta)
        df["spread"] = df["y"] - df["beta"] * df["x"]
        df["z"] = _rolling_zscore(df["spread"], int(params.z_window))

        df["eligible_today"] = (
            (~df["z"].isna())
            & (df["datetime"] >= trade_start)
            & (df["datetime"] <= trade_end)
        )

        df["asset_1"] = a1
        df["asset_2"] = a2
        df["pair_id"] = pid
        df["trade_month"] = trade_month

        out[pid] = df.set_index("datetime")[[
            "pair_id", "asset_1", "asset_2",
            "y", "x", "beta", "spread", "z", "eligible_today"
        ]]

    return out


# ============================================================
# Monthly backtest (ranking used ONLY for entry selection)
# ============================================================

def backtest_month_global_ranking(
    pair_state: Dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    params: StrategyParams,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    top_k_positions: int = 5,  # K
) -> dict:
    """
    Book rules (aligned with method #1 intent):
    - K is a hard capacity constraint (max concurrent positions).
    - Ranking is used ONLY to choose which NEW positions to open when there are
      more eligible entry signals than free slots.
    - Once a position is open, it stays open until:
        * TP: LONG exits when z > -z_exit, SHORT exits when z < +z_exit
        * SL: |z| >= z_stop
        * TIMEOUT: month end
      (No exits because it is no longer top-K.)

    Portfolio aggregation:
    - Equal-weight across OPEN positions each day:
        port_ret = (1/n_open) * sum(daily_ret_pos)
      This is what you expect for “1/M capital per active trade”.
    - Equity multiplicative: equity *= (1 + port_ret)

    Fees:
    - Applied on close as an equity haircut proportional to the *current*
      equal-weight (1/n_open_before_close). This is consistent with equal-weight.
    """
    if not pair_state:
        return {}

    # Global calendar: union of all dates, restricted to the month
    all_dates = sorted(set().union(*[df.index for df in pair_state.values()]))
    all_dates = [d for d in all_dates if (d >= trade_start) and (d <= trade_end)]
    if not all_dates:
        return {}

    trade_month = str(candidates["trade_month"].iloc[0])

    pid_to_universe = {}
    for _, r in candidates.iterrows():
        pid_to_universe[_pair_id(str(r["asset_1"]).upper(), str(r["asset_2"]).upper(), trade_month)] = str(r.get("universe", ""))

    K = int(top_k_positions)
    fees_rt = float(getattr(params, "fees", 0.0))

    equity = 1.0
    equity_rows = []

    open_positions: Dict[str, Position] = {}
    trades = []

    prev_dt: Optional[pd.Timestamp] = None

    for dt in all_dates:

        # ------------------------------------------------------------
        # 0) MTM PnL from OPEN positions (equal-weight across open)
        # ------------------------------------------------------------
        port_ret = 0.0
        n_open_prev = len(open_positions)

        if prev_dt is not None and n_open_prev > 0:
            sum_ret = 0.0
            n_valid = 0

            for pid, pos in list(open_positions.items()):
                dfp = pair_state.get(pid)
                if dfp is None:
                    continue
                if (dt not in dfp.index) or (prev_dt not in dfp.index):
                    continue

                y_now = float(dfp.loc[dt, "y"])
                x_now = float(dfp.loc[dt, "x"])
                y_prev = float(dfp.loc[prev_dt, "y"])
                x_prev = float(dfp.loc[prev_dt, "x"])
                beta = float(dfp.loc[dt, "beta"])  # fixed per month anyway

                dY = y_now - y_prev
                dX = x_now - x_prev

                position = +1.0 if pos.side == "LONG_SPREAD" else -1.0
                daily_ret_pos = position * (dY - beta * dX)

                if np.isfinite(daily_ret_pos):
                    sum_ret += daily_ret_pos
                    n_valid += 1

            if n_valid > 0:
                # Equal-weight across positions with valid marks
                port_ret = sum_ret / float(n_valid)

        new_equity = equity * (1.0 + port_ret)
        if not np.isfinite(new_equity) or new_equity <= 0:
            new_equity = 1e-8
        equity = new_equity

        # ------------------------------------------------------------
        # 1) Exits (TP/SL/MonthEnd) — NO ranking exits
        # ------------------------------------------------------------
        to_close = []

        for pid, pos in open_positions.items():
            dfp = pair_state.get(pid)
            if dfp is None or dt not in dfp.index:
                continue

            z = float(dfp.loc[dt, "z"])
            spread = float(dfp.loc[dt, "spread"])
            y_now = float(dfp.loc[dt, "y"])
            x_now = float(dfp.loc[dt, "x"])
            beta = float(dfp.loc[dt, "beta"])

            if pos.side == "LONG_SPREAD":
                exit_signal = (z > -float(params.z_exit))
            else:
                exit_signal = (z < float(params.z_exit))

            stop_signal = (abs(z) >= float(params.z_stop))
            is_month_end = (dt >= trade_end)

            if exit_signal or stop_signal or is_month_end:
                reason = "TP" if exit_signal else ("SL" if stop_signal else "MTH_END")
                to_close.append((pid, z, spread, y_now, x_now, beta, reason))

        # Close positions (apply fees proportionally to equal-weight at time of close)
        for pid, exit_z, exit_spread, y_exit, x_exit, beta_exit, reason in to_close:
            if pid not in open_positions:
                continue

            # equal-weight before removing this position
            n_open_before = max(1, len(open_positions))
            w_close = 1.0 / float(n_open_before)

            # Apply fees as equity haircut
            if fees_rt > 0:
                equity = equity * (1.0 - w_close * fees_rt)
                if not np.isfinite(equity) or equity <= 0:
                    equity = 1e-8

            pos = open_positions[pid]

            position = +1.0 if pos.side == "LONG_SPREAD" else -1.0
            trade_ret_gross = position * ((y_exit - pos.entry_y) - beta_exit * (x_exit - pos.entry_x))

            trades.append({
                "trade_month": trade_month,
                "pair_id": pos.pair_id,
                "asset_1": pos.asset_1,
                "asset_2": pos.asset_2,
                "universe": pid_to_universe.get(pid, ""),

                "beta_mode": getattr(params, "beta_mode", "monthly"),
                "beta_entry": pos.beta,

                "z_entry_th": float(params.z_entry),
                "z_exit_th": float(params.z_exit),
                "z_stop_th": float(params.z_stop),
                "z_window": int(params.z_window),
                "fees_round_trip": float(fees_rt),

                # audit: what weight was used at close (equal-weight)
                "weight_close": float(w_close),

                "entry_datetime": pos.entry_datetime,
                "exit_datetime": dt,
                "side": pos.side,
                "exit_reason": reason,

                "entry_spread": pos.entry_spread,
                "exit_spread": exit_spread,
                "entry_z": pos.entry_z,
                "exit_z": exit_z,

                "duration_days": int((dt - pos.entry_datetime).days),

                "entry_y": pos.entry_y,
                "exit_y": y_exit,
                "entry_x": pos.entry_x,
                "exit_x": x_exit,

                "trade_return_gross": float(trade_ret_gross),
            })

            del open_positions[pid]

        # ------------------------------------------------------------
        # 2) Entries (ranking ONLY for new entries), capacity K
        # ------------------------------------------------------------
        slots = K - len(open_positions)
        if slots > 0:
            entry_list = []

            for pid, dfp in pair_state.items():
                if pid in open_positions:
                    continue
                if dt not in dfp.index:
                    continue
                if not bool(dfp.loc[dt, "eligible_today"]):
                    continue

                z = float(dfp.loc[dt, "z"])
                if not np.isfinite(z):
                    continue

                if z <= -float(params.z_entry):
                    side = "LONG_SPREAD"
                elif z >= float(params.z_entry):
                    side = "SHORT_SPREAD"
                else:
                    continue

                score = abs(z)
                entry_list.append((score, pid, side, z))

            entry_list.sort(key=lambda x: x[0], reverse=True)
            entry_list = entry_list[:slots]

            for _, pid, side, z in entry_list:
                dfp = pair_state[pid]
                spread = float(dfp.loc[dt, "spread"])
                beta = float(dfp.loc[dt, "beta"])
                a1 = str(dfp.loc[dt, "asset_1"])
                a2 = str(dfp.loc[dt, "asset_2"])
                y_now = float(dfp.loc[dt, "y"])
                x_now = float(dfp.loc[dt, "x"])

                open_positions[pid] = Position(
                    pair_id=pid,
                    asset_1=a1,
                    asset_2=a2,
                    side=side,
                    beta=beta,
                    entry_datetime=dt,
                    entry_spread=spread,
                    entry_z=float(z),
                    entry_y=y_now,
                    entry_x=x_now,
                )

        # ------------------------------------------------------------
        # 3) Record daily equity
        # ------------------------------------------------------------
        equity_rows.append({
            "datetime": dt,
            "equity": float(equity),
            "n_open_positions": int(len(open_positions)),
        })

        prev_dt = dt

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

    return {"equity": equity_df, "trades": trades_df}


# ============================================================
# Walk-forward over months (global ranking)
# ============================================================

def run_global_ranking_walkforward(
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    top_n_candidates: int = 20,   # N
    max_positions: int = 5,       # K
) -> dict:
    """
    Global book, monthly pool (Top-N), daily ranking (used only for entry selection),
    chains equity across months.
    """
    monthly_universe_dir = Path(cfg.monthly_universe_path).parent

    # months available across universes
    months = set()
    for u in universes:
        fp = monthly_universe_dir / f"{u}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=["trade_month"])
        months.update(df["trade_month"].unique().tolist())

    months = sorted(months)
    if not months:
        return {}

    global_equity_chunks = []
    global_trades = []
    monthly_summary = []

    last_equity = 1.0

    for m in months:

        candidates = build_global_month_candidates(
            monthly_universe_dir=monthly_universe_dir,
            trade_month=m,
            universes=universes,
            top_n=top_n_candidates,
        )
        if candidates.empty:
            continue

        trade_start, trade_end = _month_bounds(candidates)

        pair_state = precompute_pair_state_for_month(
            cfg=cfg,
            params=params,
            candidates=candidates,
            trade_start=trade_start,
            trade_end=trade_end,
        )
        if not pair_state:
            continue

        res_m = backtest_month_global_ranking(
            pair_state=pair_state,
            candidates=candidates,
            params=params,
            trade_start=trade_start,
            trade_end=trade_end,
            top_k_positions=max_positions,
        )
        if not res_m:
            continue

        eq_m = res_m["equity"].copy().sort_values("datetime")

        # chain across months (multiplicative)
        eq_m["equity"] = eq_m["equity"] * last_equity
        last_equity = float(eq_m["equity"].iloc[-1])

        global_equity_chunks.append(eq_m[["datetime", "equity", "n_open_positions"]])

        trades_m = res_m["trades"]
        if not trades_m.empty:
            global_trades.append(trades_m)

        monthly_ret = (eq_m["equity"].iloc[-1] / eq_m["equity"].iloc[0]) - 1.0 if len(eq_m) > 1 else 0.0
        monthly_summary.append({
            "trade_month": m,
            "monthly_return": float(monthly_ret),
            "n_candidates": int(len(candidates)),
            "K": int(max_positions),
            "N": int(top_n_candidates),
        })

    if not global_equity_chunks:
        return {}

    eq = (
        pd.concat(global_equity_chunks, ignore_index=True)
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    trades = pd.concat(global_trades, ignore_index=True) if global_trades else pd.DataFrame()
    monthly = pd.DataFrame(monthly_summary)

    rets = eq["equity"].pct_change().dropna()
    stats = {
        "Final Equity": float(eq["equity"].iloc[-1]),
        "CAGR": float(eq["equity"].iloc[-1] ** (252 / len(eq)) - 1.0) if len(eq) > 2 else 0.0,
        "Sharpe": float(np.sqrt(252) * rets.mean() / rets.std(ddof=1)) if rets.std(ddof=1) > 0 else 0.0,
        "Max Drawdown": float((eq["equity"] / eq["equity"].cummax() - 1.0).min()),
        "Nb Trades": int(len(trades)),
    }

    return {"equity": eq, "trades": trades, "monthly": monthly, "stats": stats}
