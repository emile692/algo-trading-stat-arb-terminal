from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from backtesting.helpers import _pair_id, _read_price_csv
from object.class_file import BatchConfig, StrategyParams


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