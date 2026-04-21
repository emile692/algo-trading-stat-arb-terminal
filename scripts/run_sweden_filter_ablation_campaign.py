from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.engine import _spread_and_z_at_dt, run_daily_portfolio_engine
from backtesting.global_loop import (
    _get_or_build_global_context,
    _get_or_build_pair_states_for_window,
)
from object.class_file import BatchConfig, StrategyParams
from scripts.run_sweden_edge_decomposition_campaign import (
    ASSET_REGISTRY_PATH,
    BASE_FEES,
    BASE_MAX_POSITIONS,
    BASE_SELECTION_MODE,
    BASE_SELECTION_VARIANT,
    BASE_SIGNAL_SPACE,
    BASE_TOP_N,
    DATA_PATH,
    DEFAULT_END,
    DEFAULT_START,
    SCAN_FREQUENCY,
    SCAN_WEEKDAY,
    UNIVERSE,
    build_universe_assets,
    load_or_build_scans,
    segment_scans,
)
from utils.edge_decomposition import (
    REGIME_RULES_DESCRIPTION,
    build_pair_level_summary,
    build_trade_diagnostics,
    compare_configs_by_segment,
    compute_market_regime_features,
    load_price_panel,
    summarize_edge_by_segment,
)


LOGGER = logging.getLogger("sweden_filter_ablation")

REFERENCE_EDGE_DIR_NAME = "sweden_edge_decomposition_20180101_20251231_20260418_174859"

MARKET_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
)
STRUCTURE_SEGMENT_COLS = (
    "market_regime",
    "corr_type",
    "beta_stability_bucket",
    "abs_z_entry_quintile",
    "z_speed_ewma_quintile",
    "exit_reason_bucket",
)
REGIME_EXPOSURE_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
)


@dataclass(frozen=True)
class FilterThresholds:
    abs_z_extreme_min: float
    zspeed_ewma_extreme_min: float
    beta_stability_degraded_min: float
    source_dir: Path


@dataclass(frozen=True)
class AblationConfig:
    name: str
    label: str
    role: str
    letter: str
    entry_mode: str
    z_entry: float = 1.8
    z_window: int = 60
    max_holding_days: int = 30
    zspeed_ewma_span: int | None = None
    zspeed_ewma_cap: float | None = None
    use_h1_regime_filter: bool = False
    use_h2_entry_filter: bool = False
    use_h3_pair_filter: bool = False
    notes: str = ""


@dataclass
class FilterCounters:
    h1_blocked: int = 0
    h2_abs_z_blocked: int = 0
    h2_zspeed_ewma_blocked: int = 0
    h2_both_blocked: int = 0
    candidate_days_seen: int = 0
    threshold_hits_seen: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "h1_blocked_candidate_days": int(self.h1_blocked),
            "h2_abs_z_blocked_candidate_days": int(self.h2_abs_z_blocked),
            "h2_zspeed_ewma_blocked_candidate_days": int(self.h2_zspeed_ewma_blocked),
            "h2_both_blocked_candidate_days": int(self.h2_both_blocked),
            "filter_candidate_days_seen": int(self.candidate_days_seen),
            "filter_threshold_hits_seen": int(self.threshold_hits_seen),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prudent Sweden filter ablation campaign around the diagnosed best config."
    )
    parser.add_argument("--start", default=None, help="Backtest start date. Defaults to reference metadata.")
    parser.add_argument("--end", default=None, help="Backtest end date. Defaults to reference metadata.")
    parser.add_argument(
        "--reference-output",
        default=None,
        help="Reference Sweden edge-decomposition output folder used to derive cautious thresholds.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "experiments"),
        help="Root folder for campaign outputs.",
    )
    parser.add_argument("--output-suffix", default=None, help="Optional suffix appended to output directory.")
    parser.add_argument("--rebuild-scans", action="store_true", help="Rebuild Sweden scans.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def build_ablation_configs() -> list[AblationConfig]:
    best_kwargs = {
        "entry_mode": "entry_zspeed_ewma_cap",
        "zspeed_ewma_span": 5,
        "zspeed_ewma_cap": 1.3,
    }
    return [
        AblationConfig(
            letter="A",
            name="baseline_reference",
            label="baseline_reference",
            role="baseline",
            entry_mode="baseline_entry",
            notes="Comparable baseline from the edge decomposition campaign.",
        ),
        AblationConfig(
            letter="B",
            name="best_reference",
            label="best_reference",
            role="best_reference",
            notes="Current Sweden best config: zspeed EWMA span 5 cap 1.3.",
            **best_kwargs,
        ),
        AblationConfig(
            letter="C",
            name="best_plus_regime_filter",
            label="best_plus_regime_filter",
            role="h1_regime",
            use_h1_regime_filter=True,
            notes="H1: block stress/stress_trending entries only for medium_corr or degraded beta profiles.",
            **best_kwargs,
        ),
        AblationConfig(
            letter="D",
            name="best_plus_entry_filter",
            label="best_plus_entry_filter",
            role="h2_entry",
            use_h2_entry_filter=True,
            notes="H2: block only the prior worst extreme abs_z and zspeed_ewma entry buckets.",
            **best_kwargs,
        ),
        AblationConfig(
            letter="E",
            name="best_plus_pair_filter",
            label="best_plus_pair_filter",
            role="h3_pair",
            use_h3_pair_filter=True,
            notes="H3: scan-time pair-quality filter excluding medium_corr, degraded beta, and non-short half-life.",
            **best_kwargs,
        ),
        AblationConfig(
            letter="F",
            name="best_plus_regime_entry",
            label="best_plus_regime_entry",
            role="h1_h2",
            use_h1_regime_filter=True,
            use_h2_entry_filter=True,
            notes="Combined H1 + H2.",
            **best_kwargs,
        ),
        AblationConfig(
            letter="G",
            name="best_plus_regime_entry_pair",
            label="best_plus_regime_entry_pair",
            role="h1_h2_h3",
            use_h1_regime_filter=True,
            use_h2_entry_filter=True,
            use_h3_pair_filter=True,
            notes="Combined H1 + H2 + H3.",
            **best_kwargs,
        ),
    ]


def build_strategy_params(config: AblationConfig) -> StrategyParams:
    kwargs: dict[str, Any] = {
        "z_entry": float(config.z_entry),
        "z_exit": float(config.z_entry) / 3.0,
        "z_stop": 2.0 * float(config.z_entry),
        "z_window": int(config.z_window),
        "beta_mode": "static",
        "fees": BASE_FEES,
        "top_n_candidates": BASE_TOP_N,
        "max_positions": BASE_MAX_POSITIONS,
        "max_holding_days": int(config.max_holding_days),
        "exec_lag_days": 1,
        "scan_frequency": SCAN_FREQUENCY,
        "scan_weekday": SCAN_WEEKDAY,
        "signal_space": BASE_SIGNAL_SPACE,
        "selection_mode": BASE_SELECTION_MODE,
        "selection_score_variant": BASE_SELECTION_VARIANT,
        "eligibility_labels": ("ELIGIBLE",),
        "entry_mode": str(config.entry_mode),
    }
    if config.zspeed_ewma_span is not None:
        kwargs["zspeed_ewma_span"] = int(config.zspeed_ewma_span)
    if config.zspeed_ewma_cap is not None:
        kwargs["zspeed_ewma_cap"] = float(config.zspeed_ewma_cap)
    return StrategyParams(**kwargs)


def find_reference_output(exp_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Reference output not found: {path}")
        return path

    preferred = exp_root / REFERENCE_EDGE_DIR_NAME
    if preferred.exists():
        return preferred

    candidates: list[tuple[int, float, Path]] = []
    for path in exp_root.glob("sweden_edge_decomposition_*"):
        meta_path = path / "metadata.json"
        trades_path = path / "trades_enriched.csv"
        if not meta_path.exists() or not trades_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            start = pd.to_datetime(meta.get("start"))
            end = pd.to_datetime(meta.get("end"))
        except Exception:
            continue
        if pd.isna(start) or pd.isna(end):
            continue
        duration = int((end - start).days)
        candidates.append((duration, path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError("No previous Sweden edge decomposition output folder found.")
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def load_reference_metadata(reference_dir: Path) -> dict[str, Any]:
    meta_path = reference_dir / "metadata.json"
    if not meta_path.exists():
        return {"start": DEFAULT_START, "end": DEFAULT_END}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def derive_filter_thresholds(reference_dir: Path) -> FilterThresholds:
    trades_path = reference_dir / "trades_enriched.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing reference trades_enriched.csv: {trades_path}")

    trades = pd.read_csv(trades_path)
    meta = load_reference_metadata(reference_dir)
    best_name = str(meta.get("best_config_name", "best_config_zewma_s5_c1p3"))
    best = trades[trades["config_name"].astype(str) == best_name].copy()
    if best.empty:
        best = trades.copy()

    abs_bucket = best[best.get("abs_z_entry_quintile").astype(str) == "abs_z_q5"]
    zspeed_bucket = best[best.get("z_speed_ewma_quintile").astype(str) == "zspeed_ewma_q5"]
    beta_bucket = best[best.get("beta_stability_bucket").astype(str) == "beta_stability_q3"]

    abs_thr = _min_valid(abs_bucket.get("abs_z_entry"), fallback=3.144060)
    zspeed_thr = _min_valid(zspeed_bucket.get("z_speed_ewma"), fallback=0.933395)
    beta_thr = _min_valid(beta_bucket.get("beta_stability_score"), fallback=0.221501)

    return FilterThresholds(
        abs_z_extreme_min=float(abs_thr),
        zspeed_ewma_extreme_min=float(zspeed_thr),
        beta_stability_degraded_min=float(beta_thr),
        source_dir=reference_dir,
    )


def _min_valid(values: Any, *, fallback: float) -> float:
    s = pd.to_numeric(values, errors="coerce")
    if int(s.notna().sum()) == 0:
        return float(fallback)
    out = float(s.min())
    return out if np.isfinite(out) else float(fallback)


def build_output_dir(
    *,
    output_root: Path,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    suffix: str | None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_s = pd.to_datetime(start).strftime("%Y%m%d")
    end_s = pd.to_datetime(end).strftime("%Y%m%d")
    name = f"sweden_filter_ablation_{start_s}_{end_s}_{stamp}"
    if suffix:
        name = f"{name}_{suffix}"
    out = Path(output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def annotate_scan_quality(scans: pd.DataFrame, thresholds: FilterThresholds) -> pd.DataFrame:
    out = scans.copy()
    corr_abs = pd.to_numeric(out.get("6m_corr"), errors="coerce").abs()
    out["_corr_type"] = pd.cut(
        corr_abs,
        bins=[-np.inf, 0.50, 0.75, np.inf],
        labels=["low_corr", "medium_corr", "high_corr"],
    ).astype("object").fillna("missing")

    beta = pd.to_numeric(out.get("beta_std"), errors="coerce")
    out["_beta_stability_degraded"] = beta.notna() & (beta >= float(thresholds.beta_stability_degraded_min))
    out["_beta_stability_bucket_proxy"] = np.where(
        out["_beta_stability_degraded"],
        "beta_stability_q3",
        "beta_stability_not_q3",
    )

    half_life = pd.to_numeric(out.get("6m_half_life"), errors="coerce")
    out["_half_life_type"] = pd.cut(
        half_life,
        bins=[-np.inf, 20.0, 60.0, np.inf],
        labels=["short_half_life", "medium_half_life", "long_half_life"],
    ).astype("object").fillna("missing")
    out["_pair_quality_block_h3"] = (
        (out["_corr_type"].astype(str) == "medium_corr")
        | out["_beta_stability_degraded"].astype(bool)
        | out["_half_life_type"].astype(str).isin(["medium_half_life", "long_half_life"])
    )
    return out


def apply_h3_scan_filter(
    scans: pd.DataFrame,
    thresholds: FilterThresholds,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotated = annotate_scan_quality(scans, thresholds)
    before_rows = int(len(annotated))
    before_dates = int(pd.to_datetime(annotated["scan_date"], errors="coerce").nunique())
    block = annotated["_pair_quality_block_h3"].fillna(False).astype(bool)
    filtered = annotated.loc[~block].drop(
        columns=[
            "_corr_type",
            "_beta_stability_degraded",
            "_beta_stability_bucket_proxy",
            "_half_life_type",
            "_pair_quality_block_h3",
        ],
        errors="ignore",
    )
    diag = {
        "h3_scan_rows_before": before_rows,
        "h3_scan_rows_after": int(len(filtered)),
        "h3_scan_rows_removed": int(block.sum()),
        "h3_scan_removed_pct": float(block.mean()) if before_rows else np.nan,
        "h3_scan_dates_before": before_dates,
        "h3_scan_dates_after": int(pd.to_datetime(filtered["scan_date"], errors="coerce").nunique()),
    }
    return filtered.reset_index(drop=True), diag


def build_scan_feature_lookup(scans: pd.DataFrame, thresholds: FilterThresholds) -> dict[tuple[pd.Timestamp, str], dict[str, Any]]:
    annotated = annotate_scan_quality(scans, thresholds)
    annotated["scan_date"] = pd.to_datetime(annotated["scan_date"], errors="coerce").dt.normalize()
    annotated["asset_1"] = annotated["asset_1"].astype(str).str.upper()
    annotated["asset_2"] = annotated["asset_2"].astype(str).str.upper()
    annotated["_pair_id"] = annotated["asset_1"] + "_" + annotated["asset_2"]
    keep_cols = [
        "scan_date",
        "_pair_id",
        "_corr_type",
        "_beta_stability_degraded",
        "_beta_stability_bucket_proxy",
        "_half_life_type",
        "6m_corr",
        "6m_half_life",
        "beta_std",
    ]
    keep_cols = [c for c in keep_cols if c in annotated.columns]
    lookup: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    for payload in annotated[keep_cols].to_dict("records"):
        scan_dt = pd.to_datetime(payload.pop("scan_date"), errors="coerce")
        pair_id = str(payload.pop("_pair_id"))
        if pd.isna(scan_dt):
            continue
        lookup[(pd.Timestamp(scan_dt).normalize(), pair_id)] = payload
    return lookup


def build_market_regime_lookup(market_features: pd.DataFrame) -> dict[pd.Timestamp, str]:
    if market_features.empty or "datetime" not in market_features.columns:
        return {}
    work = market_features.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce").dt.normalize()
    return {
        pd.Timestamp(r.datetime).normalize(): str(r.market_regime)
        for r in work.dropna(subset=["datetime"]).itertuples(index=False)
        if hasattr(r, "market_regime")
    }


def make_filtered_ranked_pairs_fn(
    *,
    config: AblationConfig,
    params: StrategyParams,
    ctx: Any,
    pair_state_cache: dict[str, pd.DataFrame],
    scan_lookup: dict[tuple[pd.Timestamp, str], dict[str, Any]],
    market_lookup: dict[pd.Timestamp, str],
    thresholds: FilterThresholds,
    counters: FilterCounters,
):
    def get_ranked_pairs(dt: pd.Timestamp) -> list[tuple[str, str]]:
        dt = pd.to_datetime(dt).normalize()
        base_pairs = ctx.ranked_pairs_by_date.get(dt, [])
        if not base_pairs:
            return []

        if not (config.use_h1_regime_filter or config.use_h2_entry_filter):
            return list(base_pairs)

        scan_dt = pd.to_datetime(ctx.scan_date_by_trade_date.get(dt, pd.NaT), errors="coerce")
        scan_dt = pd.Timestamp(scan_dt).normalize() if pd.notna(scan_dt) else pd.NaT
        regime = str(market_lookup.get(dt, "missing"))

        kept: list[tuple[str, str]] = []
        for a1, a2 in base_pairs:
            a1 = str(a1).upper()
            a2 = str(a2).upper()
            pair_id = f"{a1}_{a2}"
            counters.candidate_days_seen += 1
            dfp = pair_state_cache.get(pair_id)
            if dfp is None or dt not in dfp.index:
                kept.append((a1, a2))
                continue

            z_signal = _safe_float(dfp.at[dt, "z"])
            if not np.isfinite(z_signal) or abs(z_signal) < float(params.z_entry):
                kept.append((a1, a2))
                continue

            counters.threshold_hits_seen += 1
            scan_features = scan_lookup.get((scan_dt, pair_id), {}) if pd.notna(scan_dt) else {}
            h1_block = _should_block_h1(config, regime, scan_features)
            h2_abs_block, h2_speed_block = _should_block_h2(config, dfp, dt, params, thresholds)

            if h1_block:
                counters.h1_blocked += 1
            if h2_abs_block and h2_speed_block:
                counters.h2_both_blocked += 1
            elif h2_abs_block:
                counters.h2_abs_z_blocked += 1
            elif h2_speed_block:
                counters.h2_zspeed_ewma_blocked += 1

            if h1_block or h2_abs_block or h2_speed_block:
                continue
            kept.append((a1, a2))
        return kept

    return get_ranked_pairs


def _should_block_h1(config: AblationConfig, regime: str, scan_features: dict[str, Any]) -> bool:
    if not config.use_h1_regime_filter:
        return False
    if str(regime) not in {"stress", "stress_trending"}:
        return False
    corr_type = str(scan_features.get("_corr_type", "missing"))
    beta_degraded = bool(scan_features.get("_beta_stability_degraded", False))
    return corr_type == "medium_corr" or beta_degraded


def _should_block_h2(
    config: AblationConfig,
    dfp: pd.DataFrame,
    dt: pd.Timestamp,
    params: StrategyParams,
    thresholds: FilterThresholds,
) -> tuple[bool, bool]:
    if not config.use_h2_entry_filter:
        return (False, False)

    beta_entry = _safe_float(dfp.at[dt, "beta"] if dt in dfp.index else np.nan)
    _, entry_z = _spread_and_z_at_dt(dfp, dt, beta_entry, int(params.z_window))
    abs_z = abs(_safe_float(entry_z))
    zspeed_ewma = _zspeed_ewma_at(dfp, dt, span=int(params.zspeed_ewma_span))
    abs_block = np.isfinite(abs_z) and abs_z >= float(thresholds.abs_z_extreme_min)
    speed_block = np.isfinite(zspeed_ewma) and zspeed_ewma >= float(thresholds.zspeed_ewma_extreme_min)
    return (bool(abs_block), bool(speed_block))


def _zspeed_ewma_at(dfp: pd.DataFrame, dt: pd.Timestamp, *, span: int) -> float:
    if dt not in dfp.index or "z" not in dfp.columns:
        return np.nan
    z_hist = pd.to_numeric(dfp.loc[:dt, "z"], errors="coerce").dropna()
    if len(z_hist) < 2:
        return np.nan
    abs_dz = z_hist.diff().dropna().abs()
    if abs_dz.empty:
        return np.nan
    lookback = max(6, 5 * max(1, int(span)))
    value = abs_dz.tail(lookback).ewm(span=max(1, int(span)), adjust=False, min_periods=1).mean().iloc[-1]
    return _safe_float(value)


def run_config(
    *,
    config: AblationConfig,
    base_scans: pd.DataFrame,
    thresholds: FilterThresholds,
    market_features: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    LOGGER.info("Running %s (%s)", config.name, config.role)
    params = build_strategy_params(config)
    cfg = BatchConfig(data_path=DATA_PATH, start_date=pd.Timestamp(start), end_date=pd.Timestamp(end))

    run_scans = segment_scans(base_scans, start=start, end=end)
    h3_diag: dict[str, Any] = {}
    if config.use_h3_pair_filter:
        run_scans, h3_diag = apply_h3_scan_filter(run_scans, thresholds)

    ctx = _get_or_build_global_context(cfg=cfg, params=params, universes=[UNIVERSE], scans=run_scans)
    if ctx is None:
        raise RuntimeError(f"No global context returned for {config.name}")
    pair_state_cache = _get_or_build_pair_states_for_window(ctx, int(params.z_window))
    counters = FilterCounters()

    scan_lookup = build_scan_feature_lookup(run_scans, thresholds)
    market_lookup = build_market_regime_lookup(market_features)
    get_ranked_pairs = make_filtered_ranked_pairs_fn(
        config=config,
        params=params,
        ctx=ctx,
        pair_state_cache=pair_state_cache,
        scan_lookup=scan_lookup,
        market_lookup=market_lookup,
        thresholds=thresholds,
        counters=counters,
    )

    def get_pair_state(dt: pd.Timestamp, pairs: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
        dt = pd.to_datetime(dt).normalize()
        out: dict[str, pd.DataFrame] = {}
        for a1, a2 in pairs:
            pid = f"{str(a1).upper()}_{str(a2).upper()}"
            dfp = pair_state_cache.get(pid)
            if dfp is None or dt not in dfp.index:
                continue
            if not bool(dfp.at[dt, "state_available"]):
                continue
            out[pid] = dfp
        return out

    engine_res = run_daily_portfolio_engine(
        params=params,
        start=ctx.start,
        end=ctx.end,
        get_ranked_pairs=get_ranked_pairs,
        get_pair_state=get_pair_state,
    )
    if not engine_res:
        raise RuntimeError(f"No engine result for {config.name}")

    result = finalize_engine_result(
        raw=engine_res,
        ctx=ctx,
        params=params,
        config=config,
    )
    filter_diag = {"config_name": config.name, **h3_diag, **counters.as_dict()}
    result["filter_diagnostics"] = pd.DataFrame([filter_diag])
    return {"config": config, "params": params, "result": result, "scans": run_scans}


def finalize_engine_result(
    *,
    raw: dict[str, Any],
    ctx: Any,
    params: StrategyParams,
    config: AblationConfig,
) -> dict[str, Any]:
    equity = raw["equity"].copy()
    trades = raw["trades"].copy()
    diagnostics = raw.get("diagnostics", pd.DataFrame()).copy()
    anomaly_flags = list(raw.get("anomaly_flags", []))

    trade_dates = pd.DatetimeIndex(ctx.trade_dates).normalize()
    trade_pos = {pd.Timestamp(dt).normalize(): i for i, dt in enumerate(trade_dates)}
    scan_usage = pd.DataFrame({"trade_date": trade_dates})
    scan_usage["scan_target_date"] = pd.DatetimeIndex(
        trade_dates - BDay(int(params.exec_lag_days))
    ).normalize()
    scan_usage["applied_scan_date"] = [
        pd.to_datetime(ctx.scan_date_by_trade_date.get(pd.Timestamp(dt).normalize(), pd.NaT))
        for dt in trade_dates
    ]
    scan_usage["scan_age_bdays"] = [
        (
            float(trade_pos[pd.Timestamp(dt).normalize()] - trade_pos[pd.Timestamp(sd).normalize()])
            if pd.notna(sd) and pd.Timestamp(sd).normalize() in trade_pos
            else np.nan
        )
        for dt, sd in zip(scan_usage["trade_date"], scan_usage["applied_scan_date"])
    ]
    scan_usage["lookahead_ok"] = [
        pd.isna(sd) or pd.Timestamp(sd).normalize() < pd.Timestamp(dt).normalize()
        for dt, sd in zip(scan_usage["trade_date"], scan_usage["applied_scan_date"])
    ]

    monthly = build_monthly_returns(equity)
    stats = build_portfolio_stats(
        config=config,
        equity=equity,
        trades=trades,
        diagnostics=diagnostics,
        scan_usage=scan_usage,
        anomaly_flags=anomaly_flags,
        params=params,
    )
    return {
        "equity": equity,
        "monthly": monthly,
        "trades": trades,
        "stats": stats,
        "diagnostics": diagnostics,
        "entry_filter_summary": raw.get("entry_filter_summary", pd.DataFrame()).copy(),
        "scan_usage": scan_usage,
        "anomaly_flags": anomaly_flags,
    }


def build_monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    out = equity.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["trade_month"] = out["datetime"].dt.strftime("%Y-%m")
    monthly = (
        out.groupby("trade_month", as_index=False)
        .agg(
            start_equity=("equity", "first"),
            end_equity=("equity", "last"),
            n_days=("equity", "size"),
            avg_open_positions=("n_open_positions", "mean"),
            max_open_positions=("n_open_positions", "max"),
        )
    )
    monthly["month_return"] = monthly["end_equity"] / monthly["start_equity"] - 1.0
    return monthly


def build_portfolio_stats(
    *,
    config: AblationConfig,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    diagnostics: pd.DataFrame,
    scan_usage: pd.DataFrame,
    anomaly_flags: list[str],
    params: StrategyParams,
) -> dict[str, Any]:
    if equity.empty:
        return {}
    eq = pd.to_numeric(equity["equity"], errors="coerce")
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    final_eq = float(eq.iloc[-1])
    total_return = final_eq - 1.0
    n = int(eq.notna().sum())
    cagr = (final_eq ** (252.0 / n) - 1.0) if n > 0 and final_eq > 0.0 else np.nan
    daily_vol = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    annualized_vol = daily_vol * float(np.sqrt(252.0)) if np.isfinite(daily_vol) else np.nan
    sharpe = float(np.sqrt(252.0) * returns.mean() / daily_vol) if np.isfinite(daily_vol) and daily_vol > 0 else np.nan
    drawdown = eq / eq.cummax() - 1.0
    mdd = float(drawdown.min()) if not drawdown.empty else np.nan
    calmar = float(cagr / abs(mdd)) if np.isfinite(cagr) and np.isfinite(mdd) and mdd < 0 else np.nan

    max_abs_pair_ret_raw = np.nan
    if not diagnostics.empty and "max_abs_pair_ret_raw" in diagnostics.columns:
        s = pd.to_numeric(diagnostics["max_abs_pair_ret_raw"], errors="coerce")
        max_abs_pair_ret_raw = float(s.max()) if int(s.notna().sum()) else np.nan

    anomaly_reasons = list(anomaly_flags)
    if np.isfinite(max_abs_pair_ret_raw) and max_abs_pair_ret_raw > 0.20:
        anomaly_reasons.append("extreme_pair_mtm_raw_gt_20pct")
    anomaly_reasons = sorted(set(str(x) for x in anomaly_reasons if str(x).strip()))

    n_open = pd.to_numeric(equity.get("n_open_positions"), errors="coerce")
    return {
        "config_name": config.name,
        "variant": config.letter,
        "role": config.role,
        "entry_mode": params.entry_mode,
        "final_equity": final_eq,
        "total_return_engine": total_return,
        "engine_sharpe": sharpe,
        "engine_cagr": cagr,
        "engine_annualized_vol": annualized_vol,
        "engine_max_drawdown": mdd,
        "engine_calmar": calmar,
        "nb_observations": n,
        "nb_days_with_positions": int((n_open > 0).sum()) if n_open.notna().any() else 0,
        "avg_open_positions": float(n_open.mean()) if n_open.notna().any() else np.nan,
        "max_open_positions": int(n_open.max()) if n_open.notna().any() else 0,
        "pct_days_fully_invested": float((n_open >= int(params.max_positions)).mean()) if n_open.notna().any() else np.nan,
        "engine_nb_trades": int(len(trades)) if isinstance(trades, pd.DataFrame) else 0,
        "lookahead_violations": int((~scan_usage["lookahead_ok"]).sum()) if not scan_usage.empty else np.nan,
        "max_abs_pair_mtm_raw": max_abs_pair_ret_raw,
        "anomaly_flag": bool(anomaly_reasons),
        "anomaly_reasons": ";".join(anomaly_reasons),
    }


def write_outputs(
    *,
    out_dir: Path,
    enriched: pd.DataFrame,
    runs: list[dict[str, Any]],
    thresholds: FilterThresholds,
    reference_meta: dict[str, Any],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Path]:
    paths = {
        "trades_enriched": out_dir / "trades_enriched.csv",
        "campaign_summary": out_dir / "campaign_summary.txt",
        "config_stats": out_dir / "config_stats.csv",
        "config_manifest": out_dir / "config_manifest.csv",
        "metadata": out_dir / "metadata.json",
        "comparison_trade_level": out_dir / "comparison_trade_level.csv",
        "comparison_portfolio_level": out_dir / "comparison_portfolio_level.csv",
        "comparison_concentration": out_dir / "comparison_concentration.csv",
        "comparison_exit_behavior": out_dir / "comparison_exit_behavior.csv",
        "comparison_regime_exposure": out_dir / "comparison_regime_exposure.csv",
        "conclusion": out_dir / "conclusion.txt",
        "pair_level_summary": out_dir / "pair_level_summary.csv",
        "monthly_returns_by_config": out_dir / "monthly_returns_by_config.csv",
        "regime_breakdown_by_config": out_dir / "regime_breakdown_by_config.csv",
        "slot_utilization_by_config": out_dir / "slot_utilization_by_config.csv",
        "best_vs_baseline_by_segment": out_dir / "best_vs_baseline_by_segment.csv",
        "variant_vs_best_by_segment": out_dir / "variant_vs_best_by_segment.csv",
        "segment_breakdown_by_config": out_dir / "segment_breakdown_by_config.csv",
        "filter_diagnostics": out_dir / "filter_diagnostics.csv",
        "entry_filter_summary": out_dir / "entry_filter_summary.csv",
    }

    enriched = enriched.copy()
    if "regime_pair_interaction" not in enriched.columns and {"market_regime", "pair_quality_bucket"}.issubset(enriched.columns):
        enriched["regime_pair_interaction"] = (
            enriched["market_regime"].astype(str) + "__" + enriched["pair_quality_bucket"].astype(str)
        )

    trade_level = build_trade_level_comparison(enriched)
    portfolio_level = pd.DataFrame([r["result"]["stats"] for r in runs])
    concentration = build_concentration_comparison(enriched)
    exit_behavior = summarize_edge_by_segment(enriched, ("exit_reason_bucket",))
    regime_exposure = summarize_edge_by_segment(enriched, REGIME_EXPOSURE_COLS)
    segment_breakdown = summarize_edge_by_segment(enriched, STRUCTURE_SEGMENT_COLS)
    pair_level = build_pair_level_summary(enriched)
    monthly = build_monthly_output(runs)
    slot_utilization = build_slot_utilization_output(runs)
    filter_diagnostics = pd.concat(
        [r["result"].get("filter_diagnostics", pd.DataFrame()) for r in runs],
        ignore_index=True,
        sort=False,
    )
    entry_filter_summary = build_entry_filter_summary_output(runs)

    best_vs_baseline = compare_configs_by_segment(
        segment_breakdown,
        best_config="best_reference",
        baseline_config="baseline_reference",
    )
    variant_vs_best = build_variant_vs_best_comparison(segment_breakdown, best_config="best_reference")

    config_manifest = pd.DataFrame([config_to_dict(r["config"]) for r in runs])
    config_stats = (
        config_manifest[["name", "letter", "role"]]
        .rename(columns={"name": "config_name", "letter": "variant"})
        .merge(trade_level, on=["config_name", "variant", "role"], how="left")
        .merge(portfolio_level, on=["config_name", "variant", "role"], how="left")
    )

    enriched.to_csv(paths["trades_enriched"], index=False)
    trade_level.to_csv(paths["comparison_trade_level"], index=False)
    portfolio_level.to_csv(paths["comparison_portfolio_level"], index=False)
    concentration.to_csv(paths["comparison_concentration"], index=False)
    exit_behavior.to_csv(paths["comparison_exit_behavior"], index=False)
    regime_exposure.to_csv(paths["comparison_regime_exposure"], index=False)
    segment_breakdown.to_csv(paths["segment_breakdown_by_config"], index=False)
    pair_level.to_csv(paths["pair_level_summary"], index=False)
    monthly.to_csv(paths["monthly_returns_by_config"], index=False)
    slot_utilization.to_csv(paths["slot_utilization_by_config"], index=False)
    best_vs_baseline.to_csv(paths["best_vs_baseline_by_segment"], index=False)
    variant_vs_best.to_csv(paths["variant_vs_best_by_segment"], index=False)
    filter_diagnostics.to_csv(paths["filter_diagnostics"], index=False)
    entry_filter_summary.to_csv(paths["entry_filter_summary"], index=False)
    config_manifest.to_csv(paths["config_manifest"], index=False)
    config_stats.to_csv(paths["config_stats"], index=False)

    metadata = {
        "universe": UNIVERSE,
        "start": str(pd.to_datetime(start).date()),
        "end": str(pd.to_datetime(end).date()),
        "reference_output": str(thresholds.source_dir),
        "reference_metadata": reference_meta,
        "thresholds": {
            "abs_z_extreme_min": thresholds.abs_z_extreme_min,
            "zspeed_ewma_extreme_min": thresholds.zspeed_ewma_extreme_min,
            "beta_stability_degraded_min": thresholds.beta_stability_degraded_min,
        },
        "regime_rules": REGIME_RULES_DESCRIPTION,
        "method_notes": [
            "H1 and H2 are implemented as pre-entry ranked-pair filters, so open-position MTM and exits remain unchanged.",
            "H3 is implemented as a scan-time filter before top-N selection, allowing backfill by other scan candidates.",
            "H2 thresholds are derived from the prior best_config destructive q5 buckets, not optimized in this script.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = build_campaign_summary(
        trade_level=trade_level,
        portfolio_level=portfolio_level,
        concentration=concentration,
        exit_behavior=exit_behavior,
        filter_diagnostics=filter_diagnostics,
        thresholds=thresholds,
        start=start,
        end=end,
    )
    conclusion = build_conclusion(
        trade_level=trade_level,
        portfolio_level=portfolio_level,
        concentration=concentration,
        exit_behavior=exit_behavior,
        filter_diagnostics=filter_diagnostics,
    )
    paths["campaign_summary"].write_text(summary, encoding="utf-8")
    paths["conclusion"].write_text(conclusion, encoding="utf-8")
    return paths


def build_trade_level_comparison(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_name, group in enriched.groupby("config_name", dropna=False):
        pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
        pnl_valid = pnl.dropna()
        holding = pd.to_numeric(group.get("holding_days"), errors="coerce")
        reasons = group.get("exit_reason_norm", pd.Series(index=group.index, dtype=object)).fillna("missing").astype(str)
        wins = pnl_valid[pnl_valid > 0]
        losses = pnl_valid[pnl_valid < 0]
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(losses.sum()) if not losses.empty else 0.0
        std = float(pnl_valid.std(ddof=1)) if len(pnl_valid) > 1 else np.nan
        avg = float(pnl_valid.mean()) if not pnl_valid.empty else np.nan
        rows.append(
            {
                "config_name": str(config_name),
                "nb_trades": int(len(group)),
                "nb_closed_trades": int(pnl.notna().sum()),
                "total_pnl": float(pnl_valid.sum()) if not pnl_valid.empty else np.nan,
                "avg_pnl_per_trade": avg,
                "median_pnl_per_trade": float(pnl_valid.median()) if not pnl_valid.empty else np.nan,
                "win_rate": float((pnl_valid > 0).mean()) if not pnl_valid.empty else np.nan,
                "avg_holding_days": float(holding.mean()) if holding.notna().any() else np.nan,
                "nb_tp": int((reasons == "TP").sum()),
                "nb_sl": int((reasons == "SL").sum()),
                "nb_time": int((reasons == "TIME").sum()),
                "nb_missing_exit": int((reasons == "missing").sum()),
                "total_tp_pnl": float(pnl[reasons == "TP"].sum()),
                "total_sl_pnl": float(pnl[reasons == "SL"].sum()),
                "total_time_pnl": float(pnl[reasons == "TIME"].sum()),
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf,
                "trade_sharpe_like": avg / std if np.isfinite(std) and std > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    manifest = pd.DataFrame([config_to_dict(c) for c in build_ablation_configs()])[
        ["name", "letter", "role"]
    ].rename(columns={"name": "config_name", "letter": "variant"})
    out = manifest.merge(out, on="config_name", how="left")
    return add_vs_best_deltas(out, "best_reference", ["nb_trades", "total_pnl", "avg_pnl_per_trade", "win_rate", "nb_sl", "nb_time"])


def add_vs_best_deltas(df: pd.DataFrame, best_name: str, metric_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    base = out[out["config_name"] == best_name]
    if base.empty:
        return out
    row = base.iloc[0]
    for col in metric_cols:
        if col in out.columns:
            out[f"delta_vs_best_{col}"] = pd.to_numeric(out[col], errors="coerce") - _safe_float(row.get(col))
    return out


def build_concentration_comparison(enriched: pd.DataFrame) -> pd.DataFrame:
    pair_level = build_pair_level_summary(enriched)
    rows: list[dict[str, Any]] = []
    for config_name, group in pair_level.groupby("config_name", dropna=False):
        pnl = pd.to_numeric(group["total_pnl"], errors="coerce")
        valid = group[pnl.notna()].copy()
        valid["_pnl"] = pnl[pnl.notna()].values
        sorted_pairs = valid.sort_values("_pnl", ascending=False)
        total = float(valid["_pnl"].sum()) if not valid.empty else np.nan
        gross_profit = float(valid.loc[valid["_pnl"] > 0, "_pnl"].sum()) if not valid.empty else 0.0
        gross_loss = float(valid.loc[valid["_pnl"] < 0, "_pnl"].sum()) if not valid.empty else 0.0
        top5 = float(sorted_pairs.head(5)["_pnl"].sum()) if not sorted_pairs.empty else np.nan
        top10 = float(sorted_pairs.head(10)["_pnl"].sum()) if not sorted_pairs.empty else np.nan
        bottom10 = float(sorted_pairs.tail(10)["_pnl"].sum()) if not sorted_pairs.empty else np.nan
        rows.append(
            {
                "config_name": str(config_name),
                "nb_paires_tradees": int(group["pair_id"].nunique()),
                "nb_paires_positives": int((pnl > 0).sum()),
                "nb_paires_negatives": int((pnl < 0).sum()),
                "nb_paires_nan_pnl": int(pnl.isna().sum()),
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "net_total_pnl": total,
                "top5_pnl": top5,
                "top10_pnl": top10,
                "bottom10_pnl": bottom10,
                "top5_share_net_pnl": top5 / total if np.isfinite(total) and total != 0 else np.nan,
                "top10_share_net_pnl": top10 / total if np.isfinite(total) and total != 0 else np.nan,
                "top5_share_gross_profit": top5 / gross_profit if gross_profit > 0 else np.nan,
                "top10_share_gross_profit": top10 / gross_profit if gross_profit > 0 else np.nan,
                "bottom10_share_gross_loss_abs": abs(bottom10) / abs(gross_loss) if gross_loss < 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    manifest = pd.DataFrame([config_to_dict(c) for c in build_ablation_configs()])[
        ["name", "letter", "role"]
    ].rename(columns={"name": "config_name", "letter": "variant"})
    out = manifest.merge(out, on="config_name", how="left")
    return add_vs_best_deltas(out, "best_reference", ["nb_paires_tradees", "gross_profit", "gross_loss", "net_total_pnl"])


def build_monthly_output(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        cfg = run["config"]
        monthly = run["result"].get("monthly", pd.DataFrame()).copy()
        if monthly.empty:
            continue
        monthly.insert(0, "config_name", cfg.name)
        monthly.insert(1, "variant", cfg.letter)
        frames.append(monthly)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_slot_utilization_output(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        cfg = run["config"]
        equity = run["result"].get("equity", pd.DataFrame()).copy()
        if equity.empty:
            continue
        n_open = pd.to_numeric(equity.get("n_open_positions"), errors="coerce")
        rows.append(
            {
                "config_name": cfg.name,
                "variant": cfg.letter,
                "avg_open_positions": float(n_open.mean()) if n_open.notna().any() else np.nan,
                "median_open_positions": float(n_open.median()) if n_open.notna().any() else np.nan,
                "max_open_positions": int(n_open.max()) if n_open.notna().any() else 0,
                "pct_days_with_positions": float((n_open > 0).mean()) if n_open.notna().any() else np.nan,
                "pct_days_fully_invested": float((n_open >= BASE_MAX_POSITIONS).mean()) if n_open.notna().any() else np.nan,
                "nb_observations": int(len(equity)),
            }
        )
    return pd.DataFrame(rows)


def build_entry_filter_summary_output(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        cfg = run["config"]
        entry = run["result"].get("entry_filter_summary", pd.DataFrame()).copy()
        if entry.empty:
            continue
        entry.insert(0, "config_name", cfg.name)
        entry.insert(1, "variant", cfg.letter)
        frames.append(entry)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_variant_vs_best_comparison(segment_summary: pd.DataFrame, *, best_config: str) -> pd.DataFrame:
    if segment_summary.empty:
        return pd.DataFrame()
    frames = []
    for config_name in sorted(segment_summary["config_name"].dropna().unique()):
        if config_name == best_config:
            continue
        comp = compare_configs_by_segment(
            segment_summary,
            best_config=str(config_name),
            baseline_config=best_config,
        )
        if comp.empty:
            continue
        comp.insert(0, "variant_config", str(config_name))
        comp = comp.rename(
            columns={
                "best_nb_trades": "variant_nb_trades",
                "best_win_rate": "variant_win_rate",
                "best_avg_pnl": "variant_avg_pnl",
                "best_total_pnl": "variant_total_pnl",
                "best_sl_rate": "variant_sl_rate",
                "best_timeout_rate": "variant_timeout_rate",
                "baseline_nb_trades": "best_reference_nb_trades",
                "baseline_win_rate": "best_reference_win_rate",
                "baseline_avg_pnl": "best_reference_avg_pnl",
                "baseline_total_pnl": "best_reference_total_pnl",
                "baseline_sl_rate": "best_reference_sl_rate",
                "baseline_timeout_rate": "best_reference_timeout_rate",
            }
        )
        frames.append(comp)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def config_to_dict(config: AblationConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "label": config.label,
        "letter": config.letter,
        "role": config.role,
        "z_entry": config.z_entry,
        "z_window": config.z_window,
        "z_exit": config.z_entry / 3.0,
        "z_stop": 2.0 * config.z_entry,
        "max_holding_days": config.max_holding_days,
        "entry_mode": config.entry_mode,
        "zspeed_ewma_span": config.zspeed_ewma_span,
        "zspeed_ewma_cap": config.zspeed_ewma_cap,
        "use_h1_regime_filter": config.use_h1_regime_filter,
        "use_h2_entry_filter": config.use_h2_entry_filter,
        "use_h3_pair_filter": config.use_h3_pair_filter,
        "notes": config.notes,
    }


def build_campaign_summary(
    *,
    trade_level: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    concentration: pd.DataFrame,
    exit_behavior: pd.DataFrame,
    filter_diagnostics: pd.DataFrame,
    thresholds: FilterThresholds,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> str:
    lines = [
        "Sweden filter ablation campaign",
        "",
        f"Period: {pd.to_datetime(start).date()} -> {pd.to_datetime(end).date()}",
        f"Reference output: {thresholds.source_dir}",
        "",
        "Thresholds derived from previous best_config buckets:",
        f"- abs_z_entry extreme threshold: >= {thresholds.abs_z_extreme_min:.6f}",
        f"- z_speed_ewma extreme threshold: >= {thresholds.zspeed_ewma_extreme_min:.6f}",
        f"- beta_stability degraded threshold: >= {thresholds.beta_stability_degraded_min:.6f}",
        "",
        "Regime rules:",
        f"- {REGIME_RULES_DESCRIPTION}",
        "",
        "Trade-level comparison:",
        _compact_table(trade_level, ["variant", "config_name", "nb_trades", "total_pnl", "avg_pnl_per_trade", "win_rate", "nb_sl", "nb_time"]),
        "",
        "Portfolio-level comparison:",
        _compact_table(portfolio_level, ["variant", "config_name", "total_return_engine", "engine_sharpe", "engine_cagr", "engine_max_drawdown", "avg_open_positions"]),
        "",
        "Concentration comparison:",
        _compact_table(concentration, ["variant", "config_name", "nb_paires_tradees", "gross_profit", "gross_loss", "top5_share_net_pnl", "top10_share_net_pnl"]),
        "",
        "Filter diagnostics:",
        _compact_table(filter_diagnostics, list(filter_diagnostics.columns)),
        "",
        "Methodological limitations:",
        "- H2 abs_z uses the engine entry-date z reconstruction and thresholds derived from prior diagnostic buckets.",
        "- H1 uses same-date market regime proxies, with lagged expanding thresholds as in the diagnostic campaign.",
        "- H3 is scan-time and can backfill top-N candidates; H1/H2 are entry-time blocks and do not backfill outside the selected top-N scan list.",
    ]
    if not exit_behavior.empty:
        lines.extend(["", "Exit behavior:", _compact_table(exit_behavior, ["config_name", "segment_value", "nb_trades", "total_pnl", "avg_pnl", "tp_rate", "sl_rate", "timeout_rate"])])
    return "\n".join(lines) + "\n"


def build_conclusion(
    *,
    trade_level: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    concentration: pd.DataFrame,
    exit_behavior: pd.DataFrame,
    filter_diagnostics: pd.DataFrame,
) -> str:
    best_trade = _row_by_config(trade_level, "best_reference")
    best_port = _row_by_config(portfolio_level, "best_reference")

    def verdict(config_name: str) -> str:
        t = _row_by_config(trade_level, config_name)
        p = _row_by_config(portfolio_level, config_name)
        if t.empty or p.empty or best_trade.empty or best_port.empty:
            return "ambiguous: missing comparison row."
        avg_delta = _safe_float(t.get("avg_pnl_per_trade")) - _safe_float(best_trade.get("avg_pnl_per_trade"))
        sharpe_delta = _safe_float(p.get("engine_sharpe")) - _safe_float(best_port.get("engine_sharpe"))
        total_delta = _safe_float(t.get("total_pnl")) - _safe_float(best_trade.get("total_pnl"))
        trades_delta = _safe_float(t.get("nb_trades")) - _safe_float(best_trade.get("nb_trades"))
        if avg_delta > 0 and sharpe_delta > 0:
            return f"validated on trade-level and portfolio-level: avg_pnl_delta={avg_delta:.6f}, sharpe_delta={sharpe_delta:.3f}, trades_delta={trades_delta:.0f}."
        if avg_delta > 0 and sharpe_delta <= 0:
            return f"ambiguous: trade-level improves but portfolio translation does not; avg_pnl_delta={avg_delta:.6f}, total_pnl_delta={total_delta:.6f}, sharpe_delta={sharpe_delta:.3f}."
        if avg_delta <= 0 and sharpe_delta > 0:
            return f"ambiguous: portfolio improves without average trade quality uplift; avg_pnl_delta={avg_delta:.6f}, sharpe_delta={sharpe_delta:.3f}."
        return f"not validated versus best_reference: avg_pnl_delta={avg_delta:.6f}, total_pnl_delta={total_delta:.6f}, sharpe_delta={sharpe_delta:.3f}."

    h1 = verdict("best_plus_regime_filter")
    h2 = verdict("best_plus_entry_filter")
    h3 = verdict("best_plus_pair_filter")

    best_sl_time = None
    if not best_trade.empty:
        base_bad = _safe_float(best_trade.get("nb_sl")) + _safe_float(best_trade.get("nb_time"))
        rows = []
        for row in trade_level.itertuples(index=False):
            if row.config_name == "best_reference":
                continue
            bad = _safe_float(getattr(row, "nb_sl", np.nan)) + _safe_float(getattr(row, "nb_time", np.nan))
            rows.append((str(row.config_name), base_bad - bad))
        if rows:
            best_sl_time = max(rows, key=lambda x: x[1])

    sharpe_rows = []
    if not best_port.empty:
        base_sharpe = _safe_float(best_port.get("engine_sharpe"))
        for row in portfolio_level.itertuples(index=False):
            if row.config_name == "best_reference":
                continue
            sharpe_rows.append((str(row.config_name), _safe_float(getattr(row, "engine_sharpe", np.nan)) - base_sharpe))
    best_sharpe = max(sharpe_rows, key=lambda x: x[1]) if sharpe_rows else None

    lines = [
        "Conclusion",
        "",
        "H1 regime filter:",
        f"- {h1}",
        "",
        "H2 entry anti-extreme filter:",
        f"- {h2}",
        "",
        "H3 pair-quality filter:",
        f"- {h3}",
        "",
        "Trade-level improvement vs portfolio-level translation:",
        "- A filter should not be promoted only because avg_pnl/trade improves; promotion requires portfolio Sharpe/return/drawdown to improve without excessive breadth loss.",
    ]
    if best_sl_time is not None:
        lines.append(f"- Largest SL/TIME count reduction versus best_reference: {best_sl_time[0]} ({best_sl_time[1]:.0f} fewer SL/TIME trades).")
    if best_sharpe is not None:
        lines.append(f"- Largest Sharpe delta versus best_reference: {best_sharpe[0]} ({best_sharpe[1]:.3f}).")
    lines.extend(
        [
            "",
            "Scientific next step:",
            "- Treat this campaign as an ablation screen. Any promoted filter should next be retested out-of-sample or by time-split before becoming a strategy rule.",
        ]
    )
    return "\n".join(lines) + "\n"


def _compact_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "(empty)"
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return df.to_string(index=False)
    return df[keep].to_string(index=False)


def _row_by_config(df: pd.DataFrame, config_name: str) -> pd.Series:
    if df is None or df.empty or "config_name" not in df.columns:
        return pd.Series(dtype=object)
    row = df[df["config_name"].astype(str) == str(config_name)]
    return row.iloc[0] if not row.empty else pd.Series(dtype=object)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_root = Path(args.output_root)
    reference_dir = find_reference_output(output_root, args.reference_output)
    reference_meta = load_reference_metadata(reference_dir)
    start = args.start or reference_meta.get("start", DEFAULT_START)
    end = args.end or reference_meta.get("end", DEFAULT_END)
    thresholds = derive_filter_thresholds(reference_dir)
    out_dir = build_output_dir(output_root=output_root, start=start, end=end, suffix=args.output_suffix)

    LOGGER.info("Reference output: %s", reference_dir)
    LOGGER.info("Output directory: %s", out_dir)
    LOGGER.info(
        "Thresholds: abs_z>=%.6f zspeed_ewma>=%.6f beta_std>=%.6f",
        thresholds.abs_z_extreme_min,
        thresholds.zspeed_ewma_extreme_min,
        thresholds.beta_stability_degraded_min,
    )

    scans = load_or_build_scans(start=start, end=end, rebuild=bool(args.rebuild_scans))
    if scans.empty:
        raise RuntimeError("No Sweden scans available.")

    assets = build_universe_assets(scans)
    LOGGER.info("Loading Sweden price panel for %d assets.", len(assets))
    price_panel = load_price_panel(assets, DATA_PATH, start=start, end=end, buffer_days=520)
    if price_panel.empty:
        raise RuntimeError("No price panel available for Sweden diagnostics.")

    market_features = compute_market_regime_features(price_panel)
    asset_metadata = pd.read_csv(ASSET_REGISTRY_PATH) if ASSET_REGISTRY_PATH.exists() else pd.DataFrame()

    runs: list[dict[str, Any]] = []
    enriched_frames: list[pd.DataFrame] = []
    for config in build_ablation_configs():
        run = run_config(
            config=config,
            base_scans=scans,
            thresholds=thresholds,
            market_features=market_features,
            start=start,
            end=end,
        )
        runs.append(run)
        enriched = build_trade_diagnostics(
            trades=run["result"]["trades"],
            config_name=config.name,
            params=run["params"],
            scans=run["scans"],
            scan_usage=run["result"].get("scan_usage", pd.DataFrame()),
            price_panel=price_panel,
            market_features=market_features,
            ranking_mode=f"{BASE_SELECTION_MODE}:{BASE_SELECTION_VARIANT}",
            asset_metadata=asset_metadata,
        )
        enriched_frames.append(enriched)
        LOGGER.info("Config %s enriched trades: %d", config.name, len(enriched))

    enriched_all = pd.concat(enriched_frames, ignore_index=True) if enriched_frames else pd.DataFrame()
    if enriched_all.empty:
        raise RuntimeError("No enriched trades produced.")

    paths = write_outputs(
        out_dir=out_dir,
        enriched=enriched_all,
        runs=runs,
        thresholds=thresholds,
        reference_meta=reference_meta,
        start=start,
        end=end,
    )

    LOGGER.info("Campaign completed.")
    for name, path in paths.items():
        LOGGER.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
