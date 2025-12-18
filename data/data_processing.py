from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUT_FOLDER = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class ProcessingConfig:
    keep_bid_ask: bool = True
    # Si True: on ne garde que datetime, open, high, low, close (et éventuellement spread_close)
    ohlc_only: bool = False
    drop_non_positive: bool = True


def _process_one_file(path: Path, cfg: ProcessingConfig) -> None:
    df = pd.read_csv(path)

    if "datetime" not in df.columns:
        raise ValueError(f"{path} missing required column: datetime")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Required OHLC
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")

    if cfg.drop_non_positive:
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

    # Minimal column set for your current terminal loader
    base_cols = ["datetime", "open", "high", "low", "close"]
    extra_cols = []
    if "spread_close" in df.columns:
        extra_cols.append("spread_close")

    if cfg.ohlc_only:
        df = df[base_cols + extra_cols]
    else:
        if not cfg.keep_bid_ask:
            # Drop bid/ask columns if present
            drop_cols = [c for c in df.columns if c.startswith("bid_") or c.startswith("ask_")]
            df = df.drop(columns=drop_cols, errors="ignore")
        # Otherwise keep everything

    df = df.reset_index(drop=True)
    df.to_csv(path, index=False)


def process_all_timeframes(raw_root: Path = RAW_OUT_FOLDER, cfg: ProcessingConfig = ProcessingConfig()) -> None:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw folder not found: {raw_root}")

    csv_files: list[Path] = []
    for tf_dir in raw_root.glob("*"):
        if tf_dir.is_dir():
            csv_files.extend(sorted(tf_dir.glob("*.csv")))

    if not csv_files:
        raise RuntimeError(f"No CSV files found under {raw_root}/<tf>/")

    for p in csv_files:
        _process_one_file(p, cfg)

    print(f"[OK] Processed {len(csv_files)} files under {raw_root}")


if __name__ == "__main__":
    # Par défaut: conserve bid/ask, ne réduit pas à OHLC-only
    process_all_timeframes()
