from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import time

from utils.loader import load_price_csv
from utils.scanner import scan_universe


@dataclass(frozen=True)
class InlineScannerConfig:
    raw_data_path: Path                 # ex: PROJECT_ROOT / "data/raw/d1"
    asset_registry_path: Path           # ex: PROJECT_ROOT / "data/asset_registry.csv"
    lookback_days: int = 504            # 2 ans par défaut
    min_obs: int = 100
    liquidity_lookback: int = 20        # rolling window
    liquidity_min_moves: float = 0.0    # si 0 => filtre "pas totalement flat"


@lru_cache(maxsize=256)
def _load_asset_csv(asset: str, raw_data_path_str: str) -> pd.DataFrame:
    raw_data_path = Path(raw_data_path_str)
    df = load_price_csv(asset, raw_data_path).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_universe_assets(asset_registry_path: Path, universe: str) -> list[str]:
    reg = pd.read_csv(asset_registry_path)
    return (
        reg.loc[reg["category_id"] == universe, "asset"]
           .astype(str)
           .str.upper()
           .tolist()
    )


def load_price_asof_norm(
    asset: str,
    asof_date: pd.Timestamp,
    cfg: InlineScannerConfig,
) -> pd.Series | None:

    try:
        df = _load_asset_csv(asset.upper(), str(cfg.raw_data_path))
    except Exception:
        return None

    asof_date = pd.to_datetime(asof_date).normalize()
    start_date = asof_date - pd.Timedelta(days=cfg.lookback_days)

    dfw = df[(df["datetime"] >= start_date) & (df["datetime"] <= asof_date)].copy()
    if len(dfw) < cfg.min_obs:
        return None

    dfw["log"] = np.log(dfw["close"])

    # liquidity/activity filter
    price_diff = dfw["close"].diff().abs()
    if price_diff.rolling(cfg.liquidity_lookback).sum().iloc[-1] <= cfg.liquidity_min_moves:
        return None

    # normalized log-price (as-of)
    dfw["norm"] = dfw["log"] - dfw["log"].iloc[-1]

    return dfw.set_index("datetime")["norm"]


def scan_universe_asof(
    universe: str,
    scan_date: pd.Timestamp,
    cfg: InlineScannerConfig,
) -> pd.DataFrame:

    assets = sorted(set(load_universe_assets(cfg.asset_registry_path, universe)))
    series: dict[str, pd.Series] = {}

    for asset in assets:
        s = load_price_asof_norm(asset, scan_date, cfg)
        if s is not None:
            series[asset] = s

    if len(series) < 2:
        return pd.DataFrame()

    prices = pd.DataFrame(series).dropna(how="all")
    if prices.shape[1] < 2:
        return pd.DataFrame()

    df_scan = scan_universe(price_df=prices, universe_name=universe)
    if df_scan.empty:
        return df_scan

    df_scan["scan_date"] = pd.to_datetime(scan_date).normalize()
    df_scan["universe"] = universe
    return df_scan


def build_scans_inline(
    universes: list[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    freq: str,
    cfg: InlineScannerConfig,
    print_every: int = 10, 
) -> pd.DataFrame:

    scan_dates = pd.date_range(
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date),
        freq=freq,
    )

    frames = []
    t0 = time.time()

    for u in universes:
        print(f"\n[SCAN] Universe={u} | dates={len(scan_dates)} | {scan_dates[0].date()} -> {scan_dates[-1].date()}")
        for i, d in enumerate(scan_dates, start=1):

            df_day = scan_universe_asof(u, d, cfg)
            if not df_day.empty:
                frames.append(df_day)

            if (i % print_every == 0) or (i == 1) or (i == len(scan_dates)):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else float("inf")
                remaining = (len(scan_dates) - i) / rate if rate > 0 else float("inf")

                print(
                    f"[SCAN] {u} | {i:>5}/{len(scan_dates)} ({i/len(scan_dates):>6.1%}) "
                    f"| last={d.date()} | frames={len(frames)} "
                    f"| {rate:,.2f} dates/s | ETA ~ {remaining/60:,.1f} min"
                )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

