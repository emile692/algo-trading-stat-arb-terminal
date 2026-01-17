# ============================================================
# MONTHLY BACKTEST — Capital constrained, ranking for ENTRY ONLY
# ============================================================
import numpy as np
import pandas as pd

from object.class_file import StrategyParams, Position


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
