# backtesting/engine.py

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from object.class_file import StrategyParams, Position


GetRankedPairsFn = Callable[[pd.Timestamp], List[Tuple[str, str]]]
GetPairStateFn = Callable[[pd.Timestamp, List[Tuple[str, str]]], Dict[str, pd.DataFrame]]


def _pid(a1: str, a2: str) -> str:
    return f"{a1.upper()}_{a2.upper()}"


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std(ddof=1)


def _spread_and_z_at_dt(
    dfp: pd.DataFrame,
    dt: pd.Timestamp,
    beta: float,
    z_window: int,
) -> tuple[float, float]:
    """
    Calcule spread(t) et z(t) avec un bêta fourni (typiquement beta d'entrée).
    dfp doit contenir au moins les colonnes y, x et un historique suffisant.
    """
    if dt not in dfp.index:
        return np.nan, np.nan

    y_dt = dfp.loc[dt, "y"]
    x_dt = dfp.loc[dt, "x"]
    if pd.isna(y_dt) or pd.isna(x_dt):
        return np.nan, np.nan

    spread_dt = float(float(y_dt) - beta * float(x_dt))

    # Only z(dt) is needed: use the trailing window up to dt, not a full rolling series.
    hist = dfp.loc[:dt, ["y", "x"]].tail(int(z_window))
    if len(hist) < int(z_window):
        return spread_dt, np.nan

    arr = hist.to_numpy(dtype=float)
    spread_hist = arr[:, 0] - beta * arr[:, 1]
    mu = float(np.mean(spread_hist))
    sd = float(np.std(spread_hist, ddof=1))
    if (not np.isfinite(sd)) or sd <= 0.0:
        return spread_dt, np.nan

    z_dt = float((spread_dt - mu) / sd)
    return spread_dt, z_dt


def run_daily_portfolio_engine(
    params: StrategyParams,
    start: pd.Timestamp,
    end: pd.Timestamp,
    get_ranked_pairs: GetRankedPairsFn,
    get_pair_state: GetPairStateFn,
) -> Dict[str, pd.DataFrame]:
    """
    Event-driven portfolio engine (single timeline).
    - Scan can be daily (handled by get_ranked_pairs)
    - Positions are unique per pair_id
    - Time stop via params.max_holding_days (business days)
    - Trade-level capital metrics: capital_at_entry/exit, trade_return

    Option A (cohérente) : bêta figé par trade.
    - MTM: utilise pos.beta
    - TP/SL: z-score recalculé avec pos.beta (pas le beta rolling du jour)
    - exit_spread/pnl_spread: spread recalculé avec pos.beta
    """

    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    trade_dates = pd.bdate_range(start=start, end=end)
    if len(trade_dates) == 0:
        return {}

    pair_return_cap = (
        float(params.pair_return_cap)
        if params.pair_return_cap is not None and float(params.pair_return_cap) > 0.0
        else None
    )
    trade_return_isolated_cap = (
        float(params.trade_return_isolated_cap)
        if params.trade_return_isolated_cap is not None and float(params.trade_return_isolated_cap) > 0.0
        else None
    )
    vol_target = (
        float(params.portfolio_vol_target)
        if params.portfolio_vol_target is not None and float(params.portfolio_vol_target) > 0.0
        else None
    )
    vol_lookback = max(5, int(params.portfolio_vol_lookback))
    vol_max_scale = max(0.0, float(params.portfolio_vol_max_scale))

    equity = 1.0
    equity_rows: List[dict] = []
    trades: List[dict] = []
    diagnostics: List[dict] = []
    anomaly_flags: List[str] = []
    raw_port_ret_hist: List[float] = []

    open_positions: Dict[str, Position] = {}
    open_meta: Dict[str, dict] = {}  # pid -> {"expiry": Timestamp, "trade_idx": int}

    prev_dt = None

    for dt in trade_dates:

        # ---------- ranked pairs for today (from scan_date = dt - exec_lag)
        ranked_pairs = get_ranked_pairs(dt)
        ranked_pairs = [(a1.upper(), a2.upper()) for (a1, a2) in ranked_pairs]

        # ---------- universe needed today = open positions + ranked candidates
        universe_pairs = set(ranked_pairs)
        for pos in open_positions.values():
            universe_pairs.add((pos.asset_1.upper(), pos.asset_2.upper()))

        mtm_return_raw = np.nan
        mtm_return_effective = np.nan
        vol_scale = 1.0
        max_abs_pair_ret_raw = np.nan
        mean_abs_pair_ret_raw = np.nan
        n_pairs_mtm = 0

        if not universe_pairs:
            diagnostics.append(
                {
                    "datetime": dt,
                    "n_pairs_mtm": 0,
                    "mtm_return_raw": np.nan,
                    "mtm_return_effective": np.nan,
                    "vol_scale": 1.0,
                    "max_abs_pair_ret_raw": np.nan,
                    "mean_abs_pair_ret_raw": np.nan,
                }
            )
            equity_rows.append({"datetime": dt, "equity": equity, "n_open_positions": 0})
            prev_dt = dt
            continue

        universe_pairs_list = sorted(universe_pairs)
        pair_state = get_pair_state(dt, universe_pairs_list)  # pid -> df indexed by datetime

        # ---------- MTM (close-to-close) using entry beta
        if prev_dt is not None and open_positions:
            rets = []
            for pid, pos in open_positions.items():
                dfp = pair_state.get(pid)
                if dfp is None:
                    continue
                if prev_dt not in dfp.index or dt not in dfp.index:
                    continue

                dY = float(dfp.loc[dt, "y"] - dfp.loc[prev_dt, "y"])
                dX = float(dfp.loc[dt, "x"] - dfp.loc[prev_dt, "x"])
                sign = 1.0 if pos.side == "LONG_SPREAD" else -1.0
                rets.append(sign * (dY - pos.beta * dX))

            if rets:
                arr_raw = np.asarray(rets, dtype=float)
                arr_raw = arr_raw[np.isfinite(arr_raw)]
                if arr_raw.size > 0:
                    n_pairs_mtm = int(arr_raw.size)
                    max_abs_pair_ret_raw = float(np.max(np.abs(arr_raw)))
                    mean_abs_pair_ret_raw = float(np.mean(np.abs(arr_raw)))

                    arr_eff = arr_raw
                    if pair_return_cap is not None:
                        arr_eff = np.clip(arr_raw, -pair_return_cap, pair_return_cap)

                    mtm_return_raw = float(np.mean(arr_eff))

                    if vol_target is not None:
                        hist = np.asarray(raw_port_ret_hist[-vol_lookback:], dtype=float)
                        hist = hist[np.isfinite(hist)]
                        if hist.size >= max(5, vol_lookback // 2):
                            realized = float(hist.std(ddof=1))
                            if np.isfinite(realized) and realized > 1e-12:
                                vol_scale = float(np.clip(vol_target / realized, 0.0, vol_max_scale))

                    mtm_return_effective = float(mtm_return_raw * vol_scale)
                    equity *= (1.0 + mtm_return_effective)
                    raw_port_ret_hist.append(float(mtm_return_raw))

        # ---------- EXITS (TP / SL / TIME) using ENTRY beta for z/spread
        to_close = []
        for pid, pos in open_positions.items():
            dfp = pair_state.get(pid)
            if dfp is None or dt not in dfp.index:
                continue

            exit_spread, z = _spread_and_z_at_dt(dfp, dt, pos.beta, params.z_window)

            exit_tp = (z > -params.z_exit) if pos.side == "LONG_SPREAD" else (z < params.z_exit)
            exit_sl = (abs(z) >= params.z_stop) if not np.isnan(z) else False
            exit_tm = dt >= open_meta[pid]["expiry"]

            if exit_tp or exit_sl or exit_tm:
                idx = open_meta[pid]["trade_idx"]
                sign = 1.0 if pos.side == "LONG_SPREAD" else -1.0

                # capital snapshot BEFORE exit fee
                capital_at_exit_pre_fee = equity
                isolated_ret = sign * (exit_spread - pos.entry_spread) - 2.0 * params.fees
                if trade_return_isolated_cap is not None:
                    isolated_ret = float(
                        np.clip(isolated_ret, -trade_return_isolated_cap, trade_return_isolated_cap)
                    )

                trades[idx].update(
                    {
                        "exit_datetime": dt,
                        "exit_z": z,
                        "exit_spread": exit_spread,
                        "pnl_spread": sign * (exit_spread - pos.entry_spread),
                        "reason": "TP" if exit_tp else ("SL" if exit_sl else "TIME"),
                        "duration_days": int((dt - pos.entry_datetime).days),
                        "capital_at_exit": capital_at_exit_pre_fee,  # pre-fee snapshot
                        "trade_return": (capital_at_exit_pre_fee / trades[idx]["capital_at_entry"]) - 1.0,
                        "trade_return_isolated": isolated_ret,
                    }
                )

                equity *= (1.0 - params.fees)  # fee at exit
                to_close.append(pid)

        for pid in to_close:
            del open_positions[pid]
            del open_meta[pid]

        # ---------- ENTRIES (from ranked pairs only)
        slots = max(0, params.max_positions - len(open_positions))
        if slots > 0 and ranked_pairs:
            ranked_today = []

            for a1, a2 in ranked_pairs:
                pid = _pid(a1, a2)
                if pid in open_positions:
                    continue

                dfp = pair_state.get(pid)
                if dfp is None or dt not in dfp.index:
                    continue

                # signal uses z computed in pair_state (beta(t) du jour)
                z_val = dfp.loc[dt, "z"]
                if pd.isna(z_val):
                    continue

                z_sig = float(z_val)
                if z_sig <= -params.z_entry:
                    side = "LONG_SPREAD"
                elif z_sig >= params.z_entry:
                    side = "SHORT_SPREAD"
                else:
                    continue

                ranked_today.append((abs(z_sig), pid, side))

            # strongest signals first
            for _, pid, side in sorted(ranked_today, reverse=True)[:slots]:
                dfp = pair_state[pid]

                beta_entry = float(dfp.loc[dt, "beta"])
                entry_spread, entry_z = _spread_and_z_at_dt(dfp, dt, beta_entry, params.z_window)

                pos = Position(
                    pair_id=pid,
                    asset_1=str(dfp.loc[dt, "asset_1"]),
                    asset_2=str(dfp.loc[dt, "asset_2"]),
                    side=side,
                    beta=beta_entry,
                    entry_datetime=dt,
                    entry_spread=entry_spread,
                    entry_z=entry_z,
                    entry_y=float(dfp.loc[dt, "y"]),
                    entry_x=float(dfp.loc[dt, "x"]),
                )

                open_positions[pid] = pos

                expiry = (dt + BDay(params.max_holding_days)).normalize()
                trade_idx = len(trades)
                open_meta[pid] = {"expiry": expiry, "trade_idx": trade_idx}

                # capital snapshot BEFORE entry fee
                capital_at_entry_pre_fee = equity

                equity *= (1.0 - params.fees)  # fee at entry

                trades.append(
                    {
                        "pair_id": pid,
                        "asset_1": pos.asset_1,
                        "asset_2": pos.asset_2,
                        "side": side,
                        "beta": pos.beta,
                        "entry_datetime": dt,
                        "entry_z": pos.entry_z,
                        "entry_spread": pos.entry_spread,
                        "exit_datetime": pd.NaT,
                        "exit_z": np.nan,
                        "exit_spread": np.nan,
                        "pnl_spread": np.nan,
                        "reason": None,
                        "duration_days": np.nan,
                        "expiry_dt": expiry,
                        "capital_at_entry": capital_at_entry_pre_fee,  # pre-fee snapshot
                        "capital_at_exit": np.nan,  # filled at exit
                        "trade_return": np.nan,  # portfolio-level return snapshot
                        "trade_return_isolated": np.nan,  # spread-return proxy net round-trip fees
                    }
                )

        diagnostics.append(
            {
                "datetime": dt,
                "n_pairs_mtm": n_pairs_mtm,
                "mtm_return_raw": mtm_return_raw,
                "mtm_return_effective": mtm_return_effective,
                "vol_scale": float(vol_scale),
                "max_abs_pair_ret_raw": max_abs_pair_ret_raw,
                "mean_abs_pair_ret_raw": mean_abs_pair_ret_raw,
            }
        )

        equity_rows.append({"datetime": dt, "equity": equity, "n_open_positions": len(open_positions)})
        prev_dt = dt

        if not np.isfinite(equity):
            anomaly_flags.append(f"non_finite_equity@{dt.date()}")
            break
        if equity <= 0.0:
            anomaly_flags.append(f"non_positive_equity@{dt.date()}")
            break

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    diagnostics_df = pd.DataFrame(diagnostics)

    # Safety: no duplicated (pair_id, entry_datetime)
    if not trades_df.empty:
        dup = trades_df.duplicated(subset=["pair_id", "entry_datetime"], keep=False)
        if dup.any():
            raise RuntimeError(f"Duplicated trades found: {int(dup.sum())}")

    return {
        "equity": equity_df,
        "trades": trades_df,
        "diagnostics": diagnostics_df,
        "anomaly_flags": anomaly_flags,
    }
