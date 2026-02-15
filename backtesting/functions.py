# backtesting/functions.py

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

from backtesting.helpers import _read_price_csv
from object.class_file import BatchConfig, StrategyParams


def _ols_beta(y, x):
    x = x - x.mean()
    denom = np.dot(x, x)
    return 1.0 if denom <= 0 else float(np.dot(x, y - y.mean()) / denom)


def _rolling_zscore(s, window):
    return (s - s.rolling(window).mean()) / s.rolling(window).std(ddof=1)


def precompute_pair_state_for_window(
    cfg: BatchConfig,
    params: StrategyParams,
    candidates: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:

    warmup = int(params.z_window + 10)
    start_load = start - pd.Timedelta(days=warmup)

    state = {}

    for _, r in candidates.iterrows():
        a1, a2 = r["asset_1"].upper(), r["asset_2"].upper()
        pid = f"{a1}_{a2}"

        try:
            df1 = _read_price_csv(Path(cfg.data_path), a1)
            df2 = _read_price_csv(Path(cfg.data_path), a2)
        except Exception:
            continue

        df = df1.merge(df2, on="datetime", suffixes=("_1", "_2"))
        df = df[(df["datetime"] >= start_load) & (df["datetime"] <= end)]
        df = df.sort_values("datetime")

        if len(df) < params.z_window + 5:
            continue

        df["y"] = df["log_norm_1"]
        df["x"] = df["log_norm_2"]

        beta_df = df[df["datetime"] < start].tail(params.z_window)
        if len(beta_df) < params.z_window:
            continue

        beta = _ols_beta(beta_df["y"].values, beta_df["x"].values)

        df["beta"] = beta
        df["spread"] = df["y"] - beta * df["x"]
        df["z"] = _rolling_zscore(df["spread"], params.z_window)

        df["eligible_today"] = (~df["z"].isna()) & (df["datetime"] >= start)

        df["pair_id"] = pid
        df["asset_1"] = a1
        df["asset_2"] = a2

        state[pid] = (
            df.set_index("datetime")[
                [
                    "pair_id",
                    "asset_1",
                    "asset_2",
                    "y",
                    "x",
                    "beta",
                    "spread",
                    "z",
                    "eligible_today",
                ]
            ]
        )

    return state

