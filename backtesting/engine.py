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
    s = dfp["y"] - beta * dfp["x"]
    z = _rolling_zscore(s, z_window)

    spread_dt = float(s.loc[dt]) if dt in s.index and pd.notna(s.loc[dt]) else np.nan
    z_dt = float(z.loc[dt]) if dt in z.index and pd.notna(z.loc[dt]) else np.nan
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

    equity = 1.0
    equity_rows: List[dict] = []
    trades: List[dict] = []

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

        if not universe_pairs:
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
                equity *= (1.0 + float(np.mean(rets)))

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

                trades[idx].update({
                    "exit_datetime": dt,
                    "exit_z": z,
                    "exit_spread": exit_spread,
                    "pnl_spread": sign * (exit_spread - pos.entry_spread),
                    "reason": "TP" if exit_tp else ("SL" if exit_sl else "TIME"),
                    "duration_days": int((dt - pos.entry_datetime).days),
                    "capital_at_exit": capital_at_exit_pre_fee,  # pre-fee snapshot
                    "trade_return": (capital_at_exit_pre_fee / trades[idx]["capital_at_entry"]) - 1.0,
                    "trade_return_isolated": sign * (exit_spread - pos.entry_spread) - 2.0 * params.fees,
                })

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

                trades.append({
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
                    "capital_at_exit": np.nan,                    # filled at exit
                    "trade_return": np.nan,                       # portfolio-level return snapshot
                    "trade_return_isolated": np.nan,              # spread-return proxy net round-trip fees
                })

        equity_rows.append({"datetime": dt, "equity": equity, "n_open_positions": len(open_positions)})
        prev_dt = dt

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

    # Safety: no duplicated (pair_id, entry_datetime)
    if not trades_df.empty:
        dup = trades_df.duplicated(subset=["pair_id", "entry_datetime"], keep=False)
        if dup.any():
            raise RuntimeError(f"Duplicated trades found: {int(dup.sum())}")

    return {"equity": equity_df, "trades": trades_df}