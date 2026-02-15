# backtesting/helpers.py

from pathlib import Path
from typing import Tuple
from functools import lru_cache

import numpy as np
import pandas as pd

# ============================================================
# Helpers — Market data & statistics (NO trading logic)
# ============================================================

@lru_cache(maxsize=512)
def _read_price_csv_cached(data_path_str: str, asset: str) -> pd.DataFrame:
    data_path = Path(data_path_str)
    asset = asset.upper()
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


def _read_price_csv(data_path: Path, asset: str) -> pd.DataFrame:
    # IMPORTANT: return a copy to avoid accidental mutation of cached DF
    return _read_price_csv_cached(str(data_path), asset).copy()


def _month_bounds(df_month: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.to_datetime(df_month["trade_start"].iloc[0]).normalize(),
        pd.to_datetime(df_month["trade_end"].iloc[0]).normalize(),
    )


def _pair_id(a1: str, a2: str, trade_month: str) -> str:
    return f"{a1}__{a2}__{trade_month}"
