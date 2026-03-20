from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import time

from utils.loader import load_price_csv
from utils.scanner import scan_universe


_VALID_SCAN_WEEKDAYS = {"MON", "TUE", "WED", "THU", "FRI"}


@dataclass(frozen=True)
class InlineScannerConfig:
    raw_data_path: Path                 # ex: PROJECT_ROOT / "data/raw/d1"
    asset_registry_path: Path           # ex: PROJECT_ROOT / "data/asset_registry.csv"
    lookback_days: int = 504            # 2 years by default
    min_obs: int = 100
    liquidity_lookback: int = 20        # rolling window
    liquidity_min_moves: float = 0.0    # if 0 => only rejects fully flat windows


@lru_cache(maxsize=4096)
def _load_asset_csv(asset: str, raw_data_path_str: str) -> pd.DataFrame:
    """
    Cached, minimal schema loader for scanner use.
    Large cache size avoids repeated CSV disk reads on broad universes.
    """
    raw_data_path = Path(raw_data_path_str)
    raw = load_price_csv(asset, raw_data_path)

    if "datetime" not in raw.columns or "close" not in raw.columns:
        raise ValueError(f"Missing required columns in {asset}: needs datetime/close")

    df = raw.loc[:, ["datetime", "close"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(float)
    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)

    # Precompute log once per asset/process instead of once per asof date.
    df["log_close"] = np.log(df["close"])
    return df


@lru_cache(maxsize=512)
def _load_universe_assets_cached(asset_registry_path_str: str, universe: str) -> tuple[str, ...]:
    reg = pd.read_csv(asset_registry_path_str, usecols=["category_id", "asset"])
    assets = (
        reg.loc[reg["category_id"] == universe, "asset"]
           .astype(str)
           .str.upper()
           .tolist()
    )
    return tuple(assets)


def load_universe_assets(asset_registry_path: Path, universe: str) -> list[str]:
    return list(_load_universe_assets_cached(str(asset_registry_path), universe))


@lru_cache(maxsize=128)
def _load_universe_trade_dates_cached(
    asset_registry_path_str: str,
    raw_data_path_str: str,
    universe: str,
) -> tuple[pd.Timestamp, ...]:
    assets = _load_universe_assets_cached(asset_registry_path_str, universe)
    dates: set[pd.Timestamp] = set()

    for asset in assets:
        try:
            df = _load_asset_csv(asset.upper(), raw_data_path_str)
        except Exception:
            continue
        if "datetime" not in df.columns or df.empty:
            continue
        vals = pd.to_datetime(df["datetime"], errors="coerce").dropna().dt.normalize()
        dates.update(pd.Timestamp(x) for x in vals.unique().tolist())

    return tuple(sorted(dates))


def load_universe_trade_dates(asset_registry_path: Path, raw_data_path: Path, universe: str) -> pd.DatetimeIndex:
    vals = _load_universe_trade_dates_cached(str(asset_registry_path), str(raw_data_path), universe)
    return pd.DatetimeIndex(vals)


def _normalize_scan_frequency(freq: str) -> str:
    f = str(freq).strip().upper()
    if f in {"WEEKLY", "WEEK", "W"}:
        return "weekly"
    if f in {"DAILY", "DAY", "D", "B"}:
        return "daily"
    return str(freq)


def _normalize_scan_weekday(scan_weekday: str) -> str:
    wd = str(scan_weekday).strip().upper()
    if wd not in _VALID_SCAN_WEEKDAYS:
        raise ValueError(f"Unsupported scan_weekday='{scan_weekday}'. Use one of {sorted(_VALID_SCAN_WEEKDAYS)}.")
    return wd


def _build_weekly_scan_dates(
    trade_dates: pd.DatetimeIndex,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    scan_weekday: str,
) -> pd.DatetimeIndex:
    if len(trade_dates) == 0:
        return pd.DatetimeIndex([])

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    wd = _normalize_scan_weekday(scan_weekday)

    dates = pd.DatetimeIndex(trade_dates).normalize()
    dates = dates[(dates >= start) & (dates <= end)]
    dates = dates[dates.dayofweek <= 4]
    if len(dates) == 0:
        return pd.DatetimeIndex([])

    buckets = dates.to_series(index=dates).groupby(dates.to_period(f"W-{wd}")).max()
    return pd.DatetimeIndex(buckets.sort_values().tolist()).normalize()


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

    mask = (df["datetime"] >= start_date) & (df["datetime"] <= asof_date)
    if not bool(mask.any()):
        return None

    dfw = df.loc[mask, ["datetime", "close", "log_close"]]
    if len(dfw) < cfg.min_obs:
        return None

    # liquidity/activity filter
    close_vals = dfw["close"].to_numpy(dtype=float, copy=False)
    lb = int(cfg.liquidity_lookback)
    if close_vals.size <= lb:
        return None
    last_move_sum = float(np.abs(np.diff(close_vals))[-lb:].sum())
    if last_move_sum <= cfg.liquidity_min_moves:
        return None

    # normalized log-price (as-of)
    log_vals = dfw["log_close"].to_numpy(dtype=float, copy=False)
    norm_vals = log_vals - log_vals[-1]

    return pd.Series(
        norm_vals,
        index=dfw["datetime"].to_numpy(copy=False),
        name="norm",
    )


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
    scan_weekday: str = "FRI",
    print_every: int = 10,
) -> pd.DataFrame:
    freq_mode = _normalize_scan_frequency(freq)
    scan_weekday = _normalize_scan_weekday(scan_weekday)

    frames = []
    t0 = time.time()

    for u in universes:
        if freq_mode == "weekly":
            trade_dates = load_universe_trade_dates(cfg.asset_registry_path, cfg.raw_data_path, u)
            scan_dates = _build_weekly_scan_dates(
                trade_dates=trade_dates,
                start_date=start_date,
                end_date=end_date,
                scan_weekday=scan_weekday,
            )
        else:
            scan_dates = pd.date_range(
                start=pd.to_datetime(start_date),
                end=pd.to_datetime(end_date),
                freq=freq,
            )

        if len(scan_dates) == 0:
            continue

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
