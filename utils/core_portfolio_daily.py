from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_sweden_filter_ablation_campaign import (
    FilterThresholds,
    build_ablation_configs,
    run_config as run_sweden_config,
)
from scripts.run_sweden_edge_decomposition_campaign import (
    build_universe_assets as build_sweden_assets,
    load_or_build_scans as load_or_build_sweden_scans,
)
from utils.country_research_pipeline import (
    BASE_DATA_PATH,
    DEFAULT_ABS_Z_THRESHOLD,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_ZSPEED_EWMA_THRESHOLD,
    PROJECT_ROOT,
    baseline_variant,
    build_country_assets,
    compute_market_regime_features,
    load_or_build_country_scans,
    load_price_panel,
    run_variant,
    select_country_reference,
)
from utils.germany_phase2 import GERMANY_BETA_THRESHOLD, make_research_variant
from utils.germany_phase3 import (
    apply_phase3_scan_filter,
    fixed_pair_filter,
    mitigation_filter,
    reference_filter,
)


def load_or_build_core_daily_returns(
    config: dict[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long and wide daily book returns for the frozen core portfolio."""

    start = str(config.get("period", {}).get("start", DEFAULT_START))
    end = str(config.get("period", {}).get("end", DEFAULT_END))
    frames = [
        load_or_build_book_daily_returns(
            book_cfg,
            root=root,
            start=start,
            end=end,
            cache_dir=cache_dir,
            rebuild=rebuild,
        )
        for book_cfg in config.get("books", [])
    ]
    daily = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if daily.empty:
        return daily, pd.DataFrame()

    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
    daily["daily_return"] = pd.to_numeric(daily["daily_return"], errors="coerce")
    period_start = pd.Timestamp(start).normalize()
    period_end = pd.Timestamp(end).normalize()
    daily = daily[(daily["trade_date"] >= period_start) & (daily["trade_date"] <= period_end)].copy()

    wide = (
        daily.pivot_table(index="trade_date", columns="book", values="daily_return", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    return daily.sort_values(["trade_date", "book"]).reset_index(drop=True), wide


def load_or_build_book_daily_returns(
    book_cfg: dict[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    cache_path = _cache_path(cache_dir, book_cfg, start=start, end=end)
    if cache_path is not None and cache_path.exists() and not rebuild:
        return _read_daily_cache(cache_path)

    daily = build_book_daily_returns(book_cfg, root=root, start=start, end=end)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(cache_path, index=False)
    return daily


def build_book_daily_returns(
    book_cfg: dict[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    country = str(book_cfg["country"]).lower().strip()
    config_name = str(book_cfg["config_name"]).strip()
    if country == "sweden":
        return _build_sweden_daily(book_cfg=book_cfg, root=root, start=start, end=end)
    if country == "germany":
        run = _run_germany_book(config_name=config_name, root=root, start=start, end=end)
    else:
        run = _run_country_reference(country=country, root=root, start=start, end=end)

    equity = run["result"].get("equity", pd.DataFrame()).copy()
    if equity.empty:
        raise RuntimeError(f"No daily equity generated for book={book_cfg.get('book')}")
    return equity_to_daily_returns(equity, book_cfg)


def _build_sweden_daily(
    *,
    book_cfg: dict[str, Any],
    root: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    config_name = str(book_cfg["config_name"])
    configs = {cfg.name: cfg for cfg in build_ablation_configs()}
    if config_name not in configs:
        raise ValueError(f"Unsupported Sweden config for daily rebuild: {config_name}")

    scan_start = "2018-01-05" if pd.Timestamp(start) <= pd.Timestamp("2018-01-05") else start
    scans = load_or_build_sweden_scans(start=scan_start, end=end, rebuild=False)
    assets = build_sweden_assets(scans)
    price_panel = load_price_panel(assets, BASE_DATA_PATH, start=start, end=end, buffer_days=520)
    market_features = compute_market_regime_features(price_panel)

    frames: list[pd.DataFrame] = []
    for split_start, split_end in _sweden_reference_splits(start=start, end=end):
        run = run_sweden_config(
            config=configs[config_name],
            base_scans=scans,
            thresholds=_sweden_thresholds(root),
            market_features=market_features,
            start=split_start,
            end=split_end,
        )
        equity = run["result"].get("equity", pd.DataFrame()).copy()
        if equity.empty:
            continue
        frames.append(equity_to_daily_returns(equity, book_cfg))
    if not frames:
        raise RuntimeError(f"No Sweden daily equity generated for config={config_name}")
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("trade_date").reset_index(drop=True)


def equity_to_daily_returns(equity: pd.DataFrame, book_cfg: dict[str, Any]) -> pd.DataFrame:
    required = {"datetime", "equity"}
    missing = required.difference(equity.columns)
    if missing:
        raise ValueError(f"Missing columns in equity output: {sorted(missing)}")

    out = equity.copy()
    out["trade_date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["trade_date", "equity"]).sort_values("trade_date").reset_index(drop=True)
    out["daily_return"] = out["equity"].pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    out["book"] = str(book_cfg["book"])
    out["country"] = str(book_cfg["country"])
    out["config_name"] = str(book_cfg["config_name"])
    if "n_open_positions" not in out.columns:
        out["n_open_positions"] = pd.NA
    return out[
        [
            "trade_date",
            "book",
            "country",
            "config_name",
            "daily_return",
            "equity",
            "n_open_positions",
        ]
    ]


def monthly_returns_from_daily(daily_returns: pd.DataFrame) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame()
    equity = (1.0 + daily_returns.fillna(0.0)).cumprod()
    period = pd.DatetimeIndex(equity.index).to_period("M")
    first = equity.groupby(period).first()
    last = equity.groupby(period).last()
    monthly = last / first - 1.0
    monthly.index = monthly.index.to_timestamp()
    monthly.index.name = "trade_month"
    return monthly


def _run_country_reference(
    *,
    country: str,
    root: Path,
    start: str,
    end: str,
) -> dict[str, Any]:
    reference = select_country_reference(country)
    scans = load_or_build_country_scans(reference, start=start, end=end, rebuild=False)
    assets = build_country_assets(country, scans)
    price_panel = load_price_panel(assets, BASE_DATA_PATH, start=start, end=end, buffer_days=520)
    market_features = compute_market_regime_features(price_panel)
    return run_variant(
        variant=baseline_variant(reference),
        base_scans=scans,
        thresholds=_thresholds(root),
        market_features=market_features,
        start=start,
        end=end,
    )


def _run_germany_book(
    *,
    config_name: str,
    root: Path,
    start: str,
    end: str,
) -> dict[str, Any]:
    reference = select_country_reference("germany")
    scans = load_or_build_country_scans(reference, start=start, end=end, rebuild=False)
    assets = build_country_assets("germany", scans)
    price_panel = load_price_panel(assets, BASE_DATA_PATH, start=start, end=end, buffer_days=520)
    market_features = compute_market_regime_features(price_panel)
    spec = _germany_spec(config_name)
    filtered_scans, _diag = apply_phase3_scan_filter(scans, spec, market_features)
    return run_variant(
        variant=make_research_variant(reference, spec.base),
        base_scans=filtered_scans,
        thresholds=_thresholds(root),
        market_features=market_features,
        start=start,
        end=end,
    )


def _germany_spec(config_name: str):
    if config_name == "reference":
        return reference_filter()
    if config_name == "pair_filter_corr_abs_le_0p75":
        return fixed_pair_filter()
    if config_name == "pair_filter_corr_abs_le_0p75_bypass_scan_stress_trending":
        return mitigation_filter()
    raise ValueError(f"Unsupported Germany config for daily rebuild: {config_name}")


def _sweden_reference_splits(*, start: str, end: str) -> list[tuple[str, str]]:
    period_start = pd.Timestamp(start).normalize()
    period_end = pd.Timestamp(end).normalize()
    source_splits = (
        ("2018-01-01", "2020-12-31"),
        ("2021-01-01", "2023-12-31"),
        ("2024-01-01", "2025-12-31"),
    )
    out: list[tuple[str, str]] = []
    for split_start, split_end in source_splits:
        left = max(period_start, pd.Timestamp(split_start))
        right = min(period_end, pd.Timestamp(split_end))
        if left <= right:
            out.append((left.strftime("%Y-%m-%d"), right.strftime("%Y-%m-%d")))
    if out:
        return out
    return [(period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"))]


def _thresholds(root: Path) -> FilterThresholds:
    return FilterThresholds(
        abs_z_extreme_min=DEFAULT_ABS_Z_THRESHOLD,
        zspeed_ewma_extreme_min=DEFAULT_ZSPEED_EWMA_THRESHOLD,
        beta_stability_degraded_min=GERMANY_BETA_THRESHOLD,
        source_dir=Path(root),
    )


def _sweden_thresholds(root: Path) -> FilterThresholds:
    source_dir = (
        Path(root)
        / "data"
        / "experiments"
        / "sweden_filter_ablation_20180101_20251231_20260418_224839"
    )
    meta_path = source_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        thresholds = meta.get("thresholds", {})
        if thresholds:
            return FilterThresholds(
                abs_z_extreme_min=float(thresholds["abs_z_extreme_min"]),
                zspeed_ewma_extreme_min=float(thresholds["zspeed_ewma_extreme_min"]),
                beta_stability_degraded_min=float(thresholds["beta_stability_degraded_min"]),
                source_dir=source_dir,
            )
    return _thresholds(root)


def _cache_path(
    cache_dir: Path | None,
    book_cfg: dict[str, Any],
    *,
    start: str,
    end: str,
) -> Path | None:
    if cache_dir is None:
        return None
    name = "_".join(
        [
            _safe_slug(book_cfg.get("book", "book")),
            _safe_slug(book_cfg.get("config_name", "config")),
            pd.Timestamp(start).strftime("%Y%m%d"),
            pd.Timestamp(end).strftime("%Y%m%d"),
            "engine_daily_returns_v4.csv",
        ]
    )
    return Path(cache_dir) / name


def _read_daily_cache(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path)
    if "trade_date" not in out.columns or "daily_return" not in out.columns:
        raise ValueError(f"Invalid daily-return cache: {path}")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["daily_return"] = pd.to_numeric(out["daily_return"], errors="coerce")
    return out.dropna(subset=["trade_date", "daily_return"]).sort_values("trade_date").reset_index(drop=True)


def _safe_slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"
