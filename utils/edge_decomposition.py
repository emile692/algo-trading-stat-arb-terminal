from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from object.class_file import StrategyParams
from utils.loader import load_price_csv


REGIME_RULES_DESCRIPTION = (
    "stress_regime = market_vol_20d above its expanding 75th percentile "
    "computed with one-day lag and min_history observations; "
    "trending_regime = abs(market_return_20d) above its expanding 75th "
    "percentile computed with one-day lag and min_history observations; "
    "neutral_regime = not stress and not trending. These rules use only "
    "history available before each date."
)


def load_price_panel(
    assets: Iterable[str],
    data_path: Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    buffer_days: int = 420,
) -> pd.DataFrame:
    """Load a close-price panel for the requested assets."""
    start_ts = pd.to_datetime(start).normalize() - pd.Timedelta(days=int(buffer_days))
    end_ts = pd.to_datetime(end).normalize()
    frames: dict[str, pd.Series] = {}

    for asset in sorted({str(a).upper() for a in assets if str(a).strip()}):
        try:
            raw = load_price_csv(asset, Path(data_path))
        except Exception:
            continue
        if "datetime" not in raw.columns or "close" not in raw.columns:
            continue

        df = raw.loc[:, ["datetime", "close"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["datetime", "close"])
        df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)]
        df = df[df["close"] > 0.0]
        if df.empty:
            continue

        df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
        frames[asset] = df.set_index("datetime")["close"].astype(float)

    if not frames:
        return pd.DataFrame()

    out = pd.DataFrame(frames).sort_index()
    out.index = pd.to_datetime(out.index).normalize()
    return out.dropna(how="all")


def compute_market_regime_features(
    price_panel: pd.DataFrame,
    *,
    min_history: int = 126,
    stress_quantile: float = 0.75,
    trend_quantile: float = 0.75,
) -> pd.DataFrame:
    """Build no-lookahead Sweden market-regime proxies from an asset panel."""
    if price_panel.empty:
        return pd.DataFrame()

    close = price_panel.copy().sort_index()
    close.index = pd.to_datetime(close.index).normalize()
    rets = close.pct_change(fill_method=None)
    market_ret = rets.mean(axis=1, skipna=True)
    daily_dispersion = rets.std(axis=1, skipna=True)
    daily_breadth = (rets > 0.0).sum(axis=1) / rets.notna().sum(axis=1).replace(0, np.nan)

    out = pd.DataFrame(index=close.index)
    out["market_return_5d"] = (1.0 + market_ret).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
    out["market_return_20d"] = (1.0 + market_ret).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    out["market_vol_20d"] = market_ret.rolling(20, min_periods=20).std(ddof=1)
    out["market_trend_20d"] = out["market_return_20d"]
    out["cross_sectional_dispersion_5d"] = daily_dispersion.rolling(5, min_periods=5).mean()
    out["cross_sectional_dispersion_20d"] = daily_dispersion.rolling(20, min_periods=20).mean()
    out["cross_sectional_breadth"] = daily_breadth.rolling(5, min_periods=5).mean()

    stress_threshold = (
        out["market_vol_20d"]
        .expanding(min_periods=int(min_history))
        .quantile(float(stress_quantile))
        .shift(1)
    )
    trend_threshold = (
        out["market_return_20d"]
        .abs()
        .expanding(min_periods=int(min_history))
        .quantile(float(trend_quantile))
        .shift(1)
    )

    out["stress_threshold_vol_20d"] = stress_threshold
    out["trend_threshold_abs_return_20d"] = trend_threshold
    out["stress_regime"] = (
        out["market_vol_20d"].notna()
        & stress_threshold.notna()
        & (out["market_vol_20d"] > stress_threshold)
    )
    out["trending_regime"] = (
        out["market_return_20d"].notna()
        & trend_threshold.notna()
        & (out["market_return_20d"].abs() > trend_threshold)
    )
    out["neutral_regime"] = ~(out["stress_regime"] | out["trending_regime"])

    out["market_regime"] = np.select(
        [
            out["stress_regime"] & out["trending_regime"],
            out["stress_regime"],
            out["trending_regime"],
        ],
        ["stress_trending", "stress", "trending"],
        default="neutral",
    )
    out.index.name = "datetime"
    return out.reset_index()


def compute_pair_profile_features(
    row: pd.Series,
    price_panel: pd.DataFrame,
    params: StrategyParams,
    *,
    spread_percentile_lookback: int = 252,
) -> dict[str, Any]:
    """Compute entry-time pair features and realized MAE/MFE from raw prices."""
    a1 = str(row.get("asset_left", row.get("asset_1", ""))).upper()
    a2 = str(row.get("asset_right", row.get("asset_2", ""))).upper()
    entry_dt = pd.to_datetime(row.get("entry_datetime"), errors="coerce")
    exit_dt = pd.to_datetime(row.get("exit_datetime"), errors="coerce")
    beta = _safe_float(row.get("beta_entry", row.get("beta")), np.nan)

    base = {
        "spread_speed_1d": np.nan,
        "spread_speed_3d": np.nan,
        "z_speed_1d": np.nan,
        "z_speed_3d": np.nan,
        "z_speed_ewma": np.nan,
        "spread_vol_20d": np.nan,
        "z_vol_20d": np.nan,
        "distance_to_mean": np.nan,
        "distance_to_mean_z_units": np.nan,
        "price_ratio": np.nan,
        "spread_percentile_lookback": np.nan,
        "recent_corr_drop": np.nan,
        "recent_vol_jump": np.nan,
        "mae": np.nan,
        "mfe": np.nan,
    }

    if (
        price_panel.empty
        or a1 not in price_panel.columns
        or a2 not in price_panel.columns
        or pd.isna(entry_dt)
        or not np.isfinite(beta)
    ):
        return base

    px = price_panel[[a1, a2]].copy().dropna(how="any")
    px = px[(px[a1] > 0.0) & (px[a2] > 0.0)]
    if px.empty:
        return base

    entry_dt = entry_dt.normalize()
    if entry_dt not in px.index:
        px_to_entry = px.loc[:entry_dt]
        if px_to_entry.empty:
            return base
        entry_dt = px_to_entry.index[-1]

    y = np.log(px[a1].astype(float))
    x = np.log(px[a2].astype(float))
    spread = y - float(beta) * x

    z_window = max(2, int(params.z_window))
    spread_mean = spread.rolling(z_window, min_periods=z_window).mean()
    spread_std = spread.rolling(z_window, min_periods=z_window).std(ddof=1)
    z = (spread - spread_mean) / spread_std.replace(0.0, np.nan)

    rets = px[[a1, a2]].pct_change(fill_method=None)
    corr_20 = rets[a1].rolling(20, min_periods=15).corr(rets[a2])
    corr_126 = rets[a1].rolling(126, min_periods=80).corr(rets[a2])

    spread_diff = spread.diff()
    spread_diff_vol_20 = spread_diff.rolling(20, min_periods=15).std(ddof=1)
    spread_diff_vol_120 = spread_diff.rolling(120, min_periods=80).std(ddof=1)

    z_diff_abs = z.diff().abs()
    span = max(1, int(getattr(params, "zspeed_ewma_span", 5)))
    z_ewma = z_diff_abs.ewm(span=span, adjust=False, min_periods=1).mean()

    def at(series: pd.Series, default: float = np.nan) -> float:
        if entry_dt not in series.index:
            return default
        return _safe_float(series.loc[entry_dt], default)

    entry_spread_local = at(spread)
    entry_z_local = at(z)
    price_2 = _safe_float(px.at[entry_dt, a2], np.nan) if entry_dt in px.index else np.nan
    if np.isfinite(price_2) and price_2 != 0.0:
        base["price_ratio"] = _safe_float(px.at[entry_dt, a1], np.nan) / price_2

    lookback = max(20, int(spread_percentile_lookback))
    spread_hist = spread.loc[:entry_dt].tail(lookback).dropna()
    if len(spread_hist) >= 20 and np.isfinite(entry_spread_local):
        base["spread_percentile_lookback"] = float((spread_hist <= entry_spread_local).mean())

    base["spread_speed_1d"] = at(spread.diff(1).abs())
    base["spread_speed_3d"] = at((spread - spread.shift(3)).abs())
    base["z_speed_1d"] = at(z.diff(1).abs())
    base["z_speed_3d"] = at((z - z.shift(3)).abs())
    base["z_speed_ewma"] = at(z_ewma)
    base["spread_vol_20d"] = at(spread.rolling(20, min_periods=15).std(ddof=1))
    base["z_vol_20d"] = at(z.rolling(20, min_periods=15).std(ddof=1))
    base["distance_to_mean"] = abs(entry_spread_local - at(spread_mean)) if np.isfinite(entry_spread_local) else np.nan
    base["distance_to_mean_z_units"] = abs(entry_z_local) if np.isfinite(entry_z_local) else np.nan
    base["recent_corr_drop"] = at(corr_20) - at(corr_126)

    vol_120 = at(spread_diff_vol_120)
    if np.isfinite(vol_120) and vol_120 > 0.0:
        base["recent_vol_jump"] = at(spread_diff_vol_20) / vol_120

    if pd.notna(exit_dt):
        exit_dt = exit_dt.normalize()
        path = spread.loc[entry_dt:exit_dt].dropna()
        if len(path) >= 1:
            side = str(row.get("side", row.get("pair_direction_at_entry", ""))).upper()
            sign = 1.0 if side == "LONG_SPREAD" else -1.0
            directed = sign * (path - path.iloc[0])
            base["mae"] = float(directed.min())
            base["mfe"] = float(directed.max())

    return base


def build_trade_diagnostics(
    *,
    trades: pd.DataFrame,
    config_name: str,
    params: StrategyParams,
    scans: pd.DataFrame,
    scan_usage: pd.DataFrame,
    price_panel: pd.DataFrame,
    market_features: pd.DataFrame,
    ranking_mode: str,
    asset_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich a run's trade journal with entry, scan, pair and regime features."""
    if trades is None or trades.empty:
        return pd.DataFrame()

    out = trades.copy()
    out["config_name"] = str(config_name)
    out["asset_left"] = out.get("asset_1", pd.Series(index=out.index, dtype=object)).astype(str).str.upper()
    out["asset_right"] = out.get("asset_2", pd.Series(index=out.index, dtype=object)).astype(str).str.upper()
    out["entry_datetime"] = pd.to_datetime(out["entry_datetime"], errors="coerce").dt.normalize()
    out["exit_datetime"] = pd.to_datetime(out.get("exit_datetime"), errors="coerce").dt.normalize()
    out["holding_days"] = pd.to_numeric(out.get("duration_days"), errors="coerce")
    missing_holding = out["holding_days"].isna() & out["exit_datetime"].notna()
    out.loc[missing_holding, "holding_days"] = (
        out.loc[missing_holding, "exit_datetime"] - out.loc[missing_holding, "entry_datetime"]
    ).dt.days
    out["exit_reason"] = out.get("reason", pd.Series(index=out.index, dtype=object)).astype("object")
    out["exit_reason_norm"] = out["exit_reason"].map(_normalize_exit_reason)
    out["beta_entry"] = pd.to_numeric(out.get("beta"), errors="coerce")
    out["hedge_ratio"] = out["beta_entry"]
    out["z_entry_observed"] = pd.to_numeric(out.get("entry_z"), errors="coerce")
    out["abs_z_entry"] = out["z_entry_observed"].abs()
    out["z_exit_threshold"] = float(params.z_exit)
    out["z_stop_threshold"] = float(params.z_stop)
    out["spread_value_entry"] = pd.to_numeric(out.get("entry_spread"), errors="coerce")
    out["pair_direction_at_entry"] = out.get("side", pd.Series(index=out.index, dtype=object)).astype(str).str.lower()
    out["signal_space"] = str(params.signal_space)
    out["ranking_mode"] = str(ranking_mode)

    out["return_pct"] = pd.to_numeric(out.get("trade_return"), errors="coerce")
    out["pnl"] = _pick_pnl(out)

    out = _attach_scan_dates(out, scan_usage)
    out = _attach_scan_features(out, scans)
    out = _attach_asset_metadata(out, asset_metadata)

    pair_features = [
        compute_pair_profile_features(row, price_panel, params)
        for _, row in out.iterrows()
    ]
    pair_df = pd.DataFrame(pair_features, index=out.index)
    out = pd.concat([out, pair_df], axis=1)

    out["reversion_potential_proxy"] = _safe_series_divide(
        out["abs_z_entry"],
        pd.to_numeric(out.get("half_life_6m"), errors="coerce"),
    )

    out = _attach_market_features(out, market_features)
    return bucketize_trade_features(out)


def bucketize_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Add stable diagnostic buckets across the combined trade population."""
    if trades.empty:
        return trades.copy()

    out = trades.copy()
    out["market_regime"] = out.get("market_regime", "missing").fillna("missing").astype(str)
    out["stress_bucket"] = np.where(_bool_column(out, "stress_regime"), "stress", "non_stress")
    out["trending_bucket"] = np.where(_bool_column(out, "trending_regime"), "trending", "non_trending")
    out["neutral_bucket"] = np.where(_bool_column(out, "neutral_regime"), "neutral", "non_neutral")

    out["abs_z_entry_quintile"] = _quantile_bucket(out.get("abs_z_entry"), 5, "abs_z")
    out["z_speed_1d_quintile"] = _quantile_bucket(out.get("z_speed_1d"), 5, "zspeed1d")
    out["z_speed_ewma_quintile"] = _quantile_bucket(out.get("z_speed_ewma"), 5, "zspeed_ewma")
    out["spread_vol_20d_bucket"] = _quantile_bucket(out.get("spread_vol_20d"), 4, "spreadvol20")

    out["half_life_6m_bucket"] = _quantile_bucket(out.get("half_life_6m"), 3, "half_life6m")
    out["nb_windows_passed_bucket"] = (
        pd.to_numeric(out.get("nb_windows_passed"), errors="coerce")
        .round()
        .astype("Int64")
        .astype(str)
        .radd("windows_")
        .replace("windows_<NA>", "missing")
    )
    out["corr_6m_abs_bucket"] = _quantile_bucket(pd.to_numeric(out.get("corr_6m"), errors="coerce").abs(), 3, "corr6m")
    out["recent_corr_drop_bucket"] = _quantile_bucket(out.get("recent_corr_drop"), 3, "corr_drop")
    out["beta_stability_bucket"] = _quantile_bucket(out.get("beta_stability_score"), 3, "beta_stability")

    half_life = pd.to_numeric(out.get("half_life_6m"), errors="coerce")
    out["half_life_type"] = pd.cut(
        half_life,
        bins=[-np.inf, 20.0, 60.0, np.inf],
        labels=["short_half_life", "medium_half_life", "long_half_life"],
    ).astype("object").fillna("missing")

    corr_abs = pd.to_numeric(out.get("corr_6m"), errors="coerce").abs()
    out["corr_type"] = pd.cut(
        corr_abs,
        bins=[-np.inf, 0.50, 0.75, np.inf],
        labels=["low_corr", "medium_corr", "high_corr"],
    ).astype("object").fillna("missing")
    out["pair_quality_bucket"] = out["half_life_type"].astype(str) + "__" + out["corr_type"].astype(str)

    same_sector = out.get("same_sector")
    if same_sector is None:
        out["sector_bucket"] = "sector_unknown"
    else:
        out["sector_bucket"] = np.select(
            [same_sector == True, same_sector == False],
            ["same_sector", "cross_sector"],
            default="sector_unknown",
        )

    out["exit_reason_bucket"] = out["exit_reason_norm"].fillna("missing").astype(str)
    return out


def summarize_edge_by_segment(
    trades: pd.DataFrame,
    segment_cols: Sequence[str],
) -> pd.DataFrame:
    """Return one long-form summary table for several segment columns."""
    if trades.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for col in segment_cols:
        if col not in trades.columns:
            continue
        work = trades.copy()
        work["_segment_value"] = work[col].astype("object").where(work[col].notna(), "missing").astype(str)
        grouped = work.groupby(["config_name", "_segment_value"], dropna=False)
        summary = grouped.apply(_segment_metrics).reset_index()
        summary = summary.rename(columns={"_segment_value": "segment_value"})
        summary.insert(0, "segment_type", col)
        frames.append(summary)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    totals = (
        trades.groupby("config_name", dropna=False)
        .agg(_total_pnl=("pnl", lambda s: pd.to_numeric(s, errors="coerce").sum()), _total_trades=("pnl", "size"))
        .reset_index()
    )
    out = out.merge(totals, on="config_name", how="left")
    out["contribution_to_total_pnl"] = _safe_series_divide(out["total_pnl"], out["_total_pnl"])
    out["contribution_to_total_trades"] = _safe_series_divide(out["nb_trades"], out["_total_trades"])
    return out.drop(columns=["_total_pnl", "_total_trades"])


def compare_configs_by_segment(
    segment_summary: pd.DataFrame,
    *,
    best_config: str,
    baseline_config: str,
) -> pd.DataFrame:
    """Compare best_config and baseline_config on a long-form segment summary."""
    if segment_summary.empty:
        return pd.DataFrame()

    key_cols = ["segment_type", "segment_value"]
    metrics = [
        "nb_trades",
        "win_rate",
        "avg_pnl",
        "median_pnl",
        "total_pnl",
        "avg_holding_days",
        "avg_mae",
        "avg_mfe",
        "tp_rate",
        "sl_rate",
        "timeout_rate",
        "profit_factor",
        "trade_sharpe_like",
    ]
    present_metrics = [m for m in metrics if m in segment_summary.columns]

    best = (
        segment_summary[segment_summary["config_name"] == best_config][key_cols + present_metrics]
        .rename(columns={m: f"best_{m}" for m in present_metrics})
    )
    base = (
        segment_summary[segment_summary["config_name"] == baseline_config][key_cols + present_metrics]
        .rename(columns={m: f"baseline_{m}" for m in present_metrics})
    )
    out = best.merge(base, on=key_cols, how="outer", indicator=True)
    out["segment_presence"] = out["_merge"].map(
        {"both": "common", "left_only": "best_only", "right_only": "baseline_only"}
    )
    out = out.drop(columns=["_merge"])

    for metric in present_metrics:
        b = pd.to_numeric(out.get(f"best_{metric}"), errors="coerce")
        a = pd.to_numeric(out.get(f"baseline_{metric}"), errors="coerce")
        out[f"delta_{metric}"] = b - a

    return out.sort_values(
        ["segment_type", "segment_presence", "delta_total_pnl"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_pair_level_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate diagnostics by config and pair_id."""
    if trades.empty:
        return pd.DataFrame()

    group_cols = ["config_name", "pair_id", "asset_left", "asset_right"]
    work = trades.copy()
    grouped = work.groupby(group_cols, dropna=False)
    out = grouped.apply(_segment_metrics).reset_index()
    return out.sort_values(["config_name", "total_pnl"], ascending=[True, False]).reset_index(drop=True)


def _attach_scan_dates(trades: pd.DataFrame, scan_usage: pd.DataFrame) -> pd.DataFrame:
    if scan_usage is None or scan_usage.empty:
        trades["applied_scan_date"] = pd.NaT
        return trades

    usage = scan_usage.copy()
    usage["trade_date"] = pd.to_datetime(usage["trade_date"], errors="coerce").dt.normalize()
    usage["applied_scan_date"] = pd.to_datetime(usage["applied_scan_date"], errors="coerce").dt.normalize()
    keep = ["trade_date", "applied_scan_date", "scan_age_bdays", "lookahead_ok"]
    keep = [c for c in keep if c in usage.columns]
    return trades.merge(
        usage[keep].drop_duplicates("trade_date"),
        left_on="entry_datetime",
        right_on="trade_date",
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")


def _attach_scan_features(trades: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    if scans is None or scans.empty:
        return trades

    scan = scans.copy()
    scan["scan_date"] = pd.to_datetime(scan["scan_date"], errors="coerce").dt.normalize()
    scan["asset_1"] = scan["asset_1"].astype(str).str.upper()
    scan["asset_2"] = scan["asset_2"].astype(str).str.upper()
    scan["pair_id"] = scan["asset_1"] + "_" + scan["asset_2"]
    scan = scan.sort_values(["scan_date", "pair_id", "eligibility_score"], ascending=[True, True, False])
    scan = scan.drop_duplicates(["scan_date", "pair_id"], keep="first")

    rename: dict[str, str] = {
        "eligibility": "eligibility_label",
        "eligibility_score": "ranking_score",
        "n_valid_windows": "nb_windows_passed",
        "beta_std": "beta_stability_score",
    }
    for win in ("3m", "6m", "12m"):
        rename.update(
            {
                f"{win}_corr": f"corr_{win}",
                f"{win}_half_life": f"half_life_{win}",
                f"{win}_adf_p": f"adf_pvalue_{win}",
                f"{win}_eg_p": f"eg_pvalue_{win}",
            }
        )

    cols = ["scan_date", "pair_id"] + [c for c in rename if c in scan.columns]
    scan = scan[cols].rename(columns=rename)

    return trades.merge(
        scan,
        left_on=["applied_scan_date", "pair_id"],
        right_on=["scan_date", "pair_id"],
        how="left",
    ).drop(columns=["scan_date"], errors="ignore")


def _attach_market_features(trades: pd.DataFrame, market_features: pd.DataFrame) -> pd.DataFrame:
    if market_features is None or market_features.empty:
        return trades

    m = market_features.copy()
    m["datetime"] = pd.to_datetime(m["datetime"], errors="coerce").dt.normalize()
    m = m.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
    return trades.merge(m, left_on="entry_datetime", right_on="datetime", how="left").drop(columns=["datetime"])


def _attach_asset_metadata(trades: pd.DataFrame, asset_metadata: pd.DataFrame | None) -> pd.DataFrame:
    out = trades.copy()
    out["sector_left"] = np.nan
    out["sector_right"] = np.nan
    out["same_sector"] = pd.NA

    if asset_metadata is None or asset_metadata.empty or "asset" not in asset_metadata.columns:
        return out

    sector_col = next((c for c in ("sector", "sector_name", "industry", "industry_group") if c in asset_metadata.columns), None)
    if sector_col is None:
        return out

    meta = asset_metadata[["asset", sector_col]].copy()
    meta["asset"] = meta["asset"].astype(str).str.upper()
    lookup = meta.drop_duplicates("asset").set_index("asset")[sector_col].to_dict()
    out["sector_left"] = out["asset_left"].map(lookup)
    out["sector_right"] = out["asset_right"].map(lookup)
    both = out["sector_left"].notna() & out["sector_right"].notna()
    out.loc[both, "same_sector"] = out.loc[both, "sector_left"] == out.loc[both, "sector_right"]
    return out


def _pick_pnl(trades: pd.DataFrame) -> pd.Series:
    for col in ("trade_return_isolated", "pnl_spread", "trade_return"):
        if col in trades.columns:
            s = pd.to_numeric(trades[col], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=trades.index, dtype=float)


def _segment_metrics(group: pd.DataFrame) -> pd.Series:
    pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
    pnl_valid = pnl.dropna()
    holding = pd.to_numeric(group.get("holding_days"), errors="coerce")
    mae = pd.to_numeric(group.get("mae"), errors="coerce")
    mfe = pd.to_numeric(group.get("mfe"), errors="coerce")
    reasons = group.get("exit_reason_norm", pd.Series(index=group.index, dtype=object)).fillna("missing").astype(str)

    wins = pnl_valid[pnl_valid > 0.0]
    losses = pnl_valid[pnl_valid < 0.0]
    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    std = float(pnl_valid.std(ddof=1)) if len(pnl_valid) > 1 else np.nan
    mean = float(pnl_valid.mean()) if not pnl_valid.empty else np.nan

    return pd.Series(
        {
            "nb_trades": int(len(group)),
            "win_rate": float((pnl_valid > 0.0).mean()) if not pnl_valid.empty else np.nan,
            "avg_pnl": mean,
            "median_pnl": float(pnl_valid.median()) if not pnl_valid.empty else np.nan,
            "total_pnl": float(pnl_valid.sum()) if not pnl_valid.empty else np.nan,
            "avg_holding_days": float(holding.mean()) if holding.notna().any() else np.nan,
            "median_holding_days": float(holding.median()) if holding.notna().any() else np.nan,
            "avg_mae": float(mae.mean()) if mae.notna().any() else np.nan,
            "avg_mfe": float(mfe.mean()) if mfe.notna().any() else np.nan,
            "tp_rate": float((reasons == "TP").mean()) if len(reasons) else np.nan,
            "sl_rate": float((reasons == "SL").mean()) if len(reasons) else np.nan,
            "timeout_rate": float((reasons == "TIME").mean()) if len(reasons) else np.nan,
            "expectancy": mean,
            "profit_factor": _profit_factor(gross_win, gross_loss),
            "trade_sharpe_like": mean / std if np.isfinite(std) and std > 0.0 else np.nan,
        }
    )


def _profit_factor(gross_win: float, gross_loss: float) -> float:
    if gross_loss < 0.0:
        return gross_win / abs(gross_loss)
    if gross_win > 0.0:
        return np.inf
    return np.nan


def _normalize_exit_reason(value: Any) -> str:
    txt = str(value).strip().upper()
    if txt in {"TP", "TAKE_PROFIT"}:
        return "TP"
    if txt in {"SL", "STOP", "STOP_LOSS"}:
        return "SL"
    if txt in {"TIME", "TIMEOUT", "TIME_STOP"}:
        return "TIME"
    if txt in {"", "NONE", "NAN", "NAT"}:
        return "missing"
    return txt


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _safe_series_divide(num: Any, den: Any) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce").replace(0.0, np.nan)
    return n / d


def _quantile_bucket(values: Any, q: int, prefix: str) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    out = pd.Series("missing", index=s.index, dtype=object)
    valid = s.dropna()
    if valid.nunique() < 2:
        return out
    bins = min(int(q), int(valid.nunique()))
    labels = [f"{prefix}_q{i}" for i in range(1, bins + 1)]
    try:
        bucketed = pd.qcut(valid.rank(method="first"), q=bins, labels=labels)
    except ValueError:
        return out
    out.loc[bucketed.index] = bucketed.astype(str)
    return out


def _bool_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)
