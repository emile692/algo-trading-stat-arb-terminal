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

from backtesting.global_loop import run_global_ranking_daily_portfolio
from object.class_file import BatchConfig, StrategyParams
from utils.edge_decomposition import (
    REGIME_RULES_DESCRIPTION,
    build_pair_level_summary,
    build_trade_diagnostics,
    compare_configs_by_segment,
    compute_market_regime_features,
    load_price_panel,
    summarize_edge_by_segment,
)
from utils.inline_scanner import InlineScannerConfig, build_scans_inline


LOGGER = logging.getLogger("sweden_edge_decomposition")

UNIVERSE = "sweden"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"
SCAN_FREQUENCY = "weekly"
SCAN_WEEKDAY = "FRI"

BASE_TOP_N = 20
BASE_MAX_POSITIONS = 5
BASE_FEES = 0.0002
BASE_SIGNAL_SPACE = "raw"
BASE_SELECTION_MODE = "legacy"
BASE_SELECTION_VARIANT = "baseline"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"

LEGACY_BEST_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "sweden_weekly_speedfilter_campaign_2018_2025"
    / "best_config_row.csv"
)

SCAN_CACHE_CANDIDATES = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "sweden_weekly_speedfilter_campaign_2018_2025"
    / "scans"
    / "sweden_weekly_fri.parquet",
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "sweden_weekly_entry_speed_campaign_2018_2025"
    / "scan_cache"
    / "sweden_weekly_fri_scans.parquet",
)

MARKET_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
)
SIGNAL_SEGMENT_COLS = (
    "abs_z_entry_quintile",
    "z_speed_1d_quintile",
    "z_speed_ewma_quintile",
    "spread_vol_20d_bucket",
)
PAIR_QUALITY_SEGMENT_COLS = (
    "half_life_6m_bucket",
    "nb_windows_passed_bucket",
    "corr_6m_abs_bucket",
    "recent_corr_drop_bucket",
    "beta_stability_bucket",
    "half_life_type",
    "corr_type",
    "pair_quality_bucket",
)
EXIT_SEGMENT_COLS = ("exit_reason_bucket",)


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    label: str
    role: str
    z_entry: float
    z_window: int
    max_holding_days: int
    entry_mode: str
    zspeed_ewma_span: int | None = None
    zspeed_ewma_cap: float | None = None
    spread_speed_cap: float | None = None
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only Sweden edge decomposition campaign."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Backtest start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Backtest end date.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "experiments"),
        help="Root folder for campaign outputs.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix appended to the output directory name.",
    )
    parser.add_argument(
        "--rebuild-scans",
        action="store_true",
        help="Rebuild Sweden weekly scans instead of using a compatible cache.",
    )
    parser.add_argument(
        "--no-extra-config",
        action="store_true",
        help="Run only best_config and baseline, skipping the compatible extra variant.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def normalize_scans(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"], errors="coerce").dt.normalize()
    out["asset_1"] = out["asset_1"].astype(str).str.upper()
    out["asset_2"] = out["asset_2"].astype(str).str.upper()
    out["eligibility"] = out.get("eligibility", "").astype(str).str.upper()
    out["universe"] = UNIVERSE
    if "eligibility_score" in out.columns:
        out["eligibility_score"] = pd.to_numeric(out["eligibility_score"], errors="coerce")
    else:
        out["eligibility_score"] = np.nan
    return (
        out.sort_values(
            ["scan_date", "asset_1", "asset_2", "eligibility_score"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=["scan_date", "asset_1", "asset_2"], keep="first")
        .reset_index(drop=True)
    )


def load_or_build_scans(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    rebuild: bool,
) -> pd.DataFrame:
    if not rebuild:
        for cache_path in SCAN_CACHE_CANDIDATES:
            if cache_path.exists():
                LOGGER.info("Loading scan cache: %s", cache_path)
                scans = normalize_scans(pd.read_parquet(cache_path))
                if _covers_period(scans, start=start, end=end):
                    return scans
                LOGGER.warning("Scan cache does not fully cover requested period: %s", cache_path)

    LOGGER.info("Building Sweden weekly scans from raw data.")
    inline_cfg = InlineScannerConfig(
        raw_data_path=DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
    )
    scans = build_scans_inline(
        universes=[UNIVERSE],
        start_date=start,
        end_date=end,
        freq=SCAN_FREQUENCY,
        scan_weekday=SCAN_WEEKDAY,
        cfg=inline_cfg,
        print_every=20,
    )
    return normalize_scans(scans)


def build_campaign_configs(include_extra: bool = True) -> list[CampaignConfig]:
    best = _best_config_from_history()
    configs = [
        CampaignConfig(
            name="baseline_entry",
            label="baseline_entry",
            role="baseline",
            z_entry=best.z_entry,
            z_window=best.z_window,
            max_holding_days=best.max_holding_days,
            entry_mode="baseline_entry",
            notes="Comparable threshold-only baseline with same z/window/hold settings as best_config.",
        ),
        best,
    ]
    if include_extra:
        configs.append(
            CampaignConfig(
                name="spread_speed_filter_cap_0p50",
                label="entry_with_spread_speed_filter_cap_0.50",
                role="compatible_reference",
                z_entry=best.z_entry,
                z_window=best.z_window,
                max_holding_days=best.max_holding_days,
                entry_mode="entry_with_spread_speed_filter",
                spread_speed_cap=0.50,
                notes="Single compatible challenger already present in recent Sweden entry-speed research.",
            )
        )
    return configs


def build_strategy_params(config: CampaignConfig) -> StrategyParams:
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
        "entry_mode": config.entry_mode,
    }
    if config.zspeed_ewma_span is not None:
        kwargs["zspeed_ewma_span"] = int(config.zspeed_ewma_span)
    if config.zspeed_ewma_cap is not None:
        kwargs["zspeed_ewma_cap"] = float(config.zspeed_ewma_cap)
    if config.spread_speed_cap is not None:
        kwargs["spread_speed_cap"] = float(config.spread_speed_cap)
    return StrategyParams(**kwargs)


def run_config(
    *,
    config: CampaignConfig,
    scans: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    LOGGER.info("Running config=%s (%s)", config.name, config.entry_mode)
    params = build_strategy_params(config)
    cfg = BatchConfig(data_path=DATA_PATH, start_date=pd.Timestamp(start), end_date=pd.Timestamp(end))
    run_scans = segment_scans(scans, start=start, end=end)
    result = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=[UNIVERSE],
        scans=run_scans,
    )
    if not result:
        raise RuntimeError(f"No result returned for config={config.name}")
    return {"config": config, "params": params, "result": result, "scans": run_scans}


def segment_scans(
    scans: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    buffer_bdays: int = 30,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    buffer_start = (start_ts - BDay(int(buffer_bdays))).normalize()
    out = scans.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"], errors="coerce").dt.normalize()
    return out[(out["scan_date"] >= buffer_start) & (out["scan_date"] <= end_ts)].reset_index(drop=True)


def build_universe_assets(scans: pd.DataFrame) -> list[str]:
    assets: set[str] = set()
    if ASSET_REGISTRY_PATH.exists():
        reg = pd.read_csv(ASSET_REGISTRY_PATH)
        if {"category_id", "asset"}.issubset(reg.columns):
            assets.update(
                reg.loc[reg["category_id"].astype(str).str.lower() == UNIVERSE, "asset"]
                .astype(str)
                .str.upper()
                .tolist()
            )
    if not scans.empty:
        assets.update(scans["asset_1"].astype(str).str.upper().tolist())
        assets.update(scans["asset_2"].astype(str).str.upper().tolist())
    return sorted(assets)


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
    name = f"sweden_edge_decomposition_{start_s}_{end_s}_{stamp}"
    if suffix:
        name = f"{name}_{suffix}"
    out = Path(output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_outputs(
    *,
    out_dir: Path,
    enriched: pd.DataFrame,
    runs: list[dict[str, Any]],
    best_config_name: str,
    baseline_config_name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Path]:
    enriched = enriched.copy()
    enriched["regime_pair_interaction"] = (
        enriched["market_regime"].astype(str) + "__" + enriched["pair_quality_bucket"].astype(str)
    )

    market_summary = summarize_edge_by_segment(enriched, MARKET_SEGMENT_COLS)
    signal_summary = summarize_edge_by_segment(enriched, SIGNAL_SEGMENT_COLS)
    pair_quality_summary = summarize_edge_by_segment(enriched, PAIR_QUALITY_SEGMENT_COLS)
    exit_summary = summarize_edge_by_segment(enriched, EXIT_SEGMENT_COLS)
    interaction_summary = summarize_edge_by_segment(enriched, ("regime_pair_interaction",))

    all_segment_summary = pd.concat(
        [
            market_summary,
            signal_summary,
            pair_quality_summary,
            exit_summary,
            interaction_summary,
        ],
        ignore_index=True,
    )
    comparison = compare_configs_by_segment(
        all_segment_summary,
        best_config=best_config_name,
        baseline_config=baseline_config_name,
    )
    pair_level = build_pair_level_summary(enriched)
    config_stats = build_config_stats(runs, enriched)

    paths = {
        "trades_enriched": out_dir / "trades_enriched.csv",
        "summary_by_market_regime": out_dir / "summary_by_market_regime.csv",
        "summary_by_signal_bucket": out_dir / "summary_by_signal_bucket.csv",
        "summary_by_pair_quality": out_dir / "summary_by_pair_quality.csv",
        "summary_by_exit_reason": out_dir / "summary_by_exit_reason.csv",
        "best_vs_baseline_by_segment": out_dir / "best_vs_baseline_by_segment.csv",
        "pair_level_summary": out_dir / "pair_level_summary.csv",
        "regime_pair_interactions": out_dir / "regime_pair_interactions.csv",
        "config_stats": out_dir / "config_stats.csv",
        "config_manifest": out_dir / "config_manifest.csv",
        "metadata": out_dir / "metadata.json",
        "campaign_summary": out_dir / "campaign_summary.txt",
    }

    enriched.to_csv(paths["trades_enriched"], index=False)
    market_summary.to_csv(paths["summary_by_market_regime"], index=False)
    signal_summary.to_csv(paths["summary_by_signal_bucket"], index=False)
    pair_quality_summary.to_csv(paths["summary_by_pair_quality"], index=False)
    exit_summary.to_csv(paths["summary_by_exit_reason"], index=False)
    comparison.to_csv(paths["best_vs_baseline_by_segment"], index=False)
    pair_level.to_csv(paths["pair_level_summary"], index=False)
    interaction_summary.to_csv(paths["regime_pair_interactions"], index=False)
    config_stats.to_csv(paths["config_stats"], index=False)
    pd.DataFrame([config_to_dict(r["config"]) for r in runs]).to_csv(paths["config_manifest"], index=False)

    metadata = {
        "universe": UNIVERSE,
        "start": str(pd.to_datetime(start).date()),
        "end": str(pd.to_datetime(end).date()),
        "best_config_name": best_config_name,
        "baseline_config_name": baseline_config_name,
        "regime_rules": REGIME_RULES_DESCRIPTION,
        "pnl_metric": "pnl uses trade_return_isolated when available, then pnl_spread, then trade_return.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_text = build_campaign_summary(
        enriched=enriched,
        all_segment_summary=all_segment_summary,
        comparison=comparison,
        pair_level=pair_level,
        config_stats=config_stats,
        best_config_name=best_config_name,
        baseline_config_name=baseline_config_name,
        start=start,
        end=end,
    )
    paths["campaign_summary"].write_text(summary_text, encoding="utf-8")
    return paths


def build_config_stats(runs: list[dict[str, Any]], enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        cfg = run["config"]
        res = run["result"]
        stats = dict(res.get("stats", {}))
        trades = enriched[enriched["config_name"] == cfg.name]
        pnl = pd.to_numeric(trades.get("pnl"), errors="coerce")
        rows.append(
            {
                "config_name": cfg.name,
                "role": cfg.role,
                "entry_mode": cfg.entry_mode,
                "z_entry": cfg.z_entry,
                "z_window": cfg.z_window,
                "max_holding_days": cfg.max_holding_days,
                "engine_sharpe": stats.get("Sharpe", np.nan),
                "engine_cagr": stats.get("CAGR", np.nan),
                "engine_max_drawdown": stats.get("Max Drawdown", np.nan),
                "engine_nb_trades": stats.get("Nb Trades", np.nan),
                "diagnostic_nb_trades": int(len(trades)),
                "diagnostic_total_pnl": float(pnl.sum()) if pnl.notna().any() else np.nan,
                "diagnostic_avg_pnl": float(pnl.mean()) if pnl.notna().any() else np.nan,
                "diagnostic_win_rate": float((pnl.dropna() > 0.0).mean()) if pnl.notna().any() else np.nan,
                "lookahead_violations": stats.get("Lookahead violations", np.nan),
                "anomaly_flag": stats.get("Anomaly flag", np.nan),
                "anomaly_reasons": stats.get("Anomaly reasons", ""),
            }
        )
    return pd.DataFrame(rows)


def build_campaign_summary(
    *,
    enriched: pd.DataFrame,
    all_segment_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    pair_level: pd.DataFrame,
    config_stats: pd.DataFrame,
    best_config_name: str,
    baseline_config_name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> str:
    best_segments = all_segment_summary[
        (all_segment_summary["config_name"] == best_config_name)
        & (pd.to_numeric(all_segment_summary["nb_trades"], errors="coerce") >= 3)
    ].copy()
    best_segments["total_pnl"] = pd.to_numeric(best_segments["total_pnl"], errors="coerce")

    top_contributors = best_segments.sort_values("total_pnl", ascending=False).head(5)
    top_destructors = best_segments.sort_values("total_pnl", ascending=True).head(5)

    best_pairs = pair_level[pair_level["config_name"] == best_config_name].copy()
    best_pairs["total_pnl"] = pd.to_numeric(best_pairs["total_pnl"], errors="coerce")
    top_pairs = best_pairs.sort_values("total_pnl", ascending=False).head(5)
    bad_pairs = best_pairs.sort_values("total_pnl", ascending=True).head(5)

    best_stats = _stats_row(config_stats, best_config_name)
    base_stats = _stats_row(config_stats, baseline_config_name)
    diagnosis = _quality_vs_trade_count_diagnosis(best_stats, base_stats, comparison)
    loss_concentration = _loss_concentration_notes(best_segments)
    hypotheses = _next_hypotheses(best_segments, bad_pairs)

    lines = [
        "Sweden edge decomposition campaign",
        "",
        f"Period: {pd.to_datetime(start).date()} -> {pd.to_datetime(end).date()}",
        f"Best config: {best_config_name}",
        f"Baseline: {baseline_config_name}",
        "",
        "Regime rules:",
        f"- {REGIME_RULES_DESCRIPTION}",
        "",
        "Overall comparison:",
        _format_overall_delta(best_stats, base_stats),
        "",
        "Main interpretation:",
        f"- {diagnosis}",
        "",
        "Top 5 contributing segments for best_config:",
        *_format_segment_rows(top_contributors),
        "",
        "Top 5 destructive segments for best_config:",
        *_format_segment_rows(top_destructors),
        "",
        "Loss concentration checks:",
        *[f"- {line}" for line in loss_concentration],
        "",
        "Top contributing pairs:",
        *_format_pair_rows(top_pairs),
        "",
        "Top destructive pairs:",
        *_format_pair_rows(bad_pairs),
        "",
        "Next research hypotheses:",
        *[f"- {line}" for line in hypotheses],
        "",
        "Methodological notes:",
        "- This campaign is diagnostic only and does not alter engine logic.",
        "- Segment buckets are built on the combined trade population for comparability.",
        "- PnL uses trade_return_isolated when available, then pnl_spread, then trade_return.",
        "- MAE/MFE are reconstructed from raw price paths with the trade entry beta.",
        "- Scan features are joined from the scan snapshot actually applied before entry.",
    ]
    return "\n".join(lines) + "\n"


def config_to_dict(config: CampaignConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "label": config.label,
        "role": config.role,
        "z_entry": config.z_entry,
        "z_window": config.z_window,
        "z_exit": config.z_entry / 3.0,
        "z_stop": 2.0 * config.z_entry,
        "max_holding_days": config.max_holding_days,
        "entry_mode": config.entry_mode,
        "zspeed_ewma_span": config.zspeed_ewma_span,
        "zspeed_ewma_cap": config.zspeed_ewma_cap,
        "spread_speed_cap": config.spread_speed_cap,
        "notes": config.notes,
    }


def _best_config_from_history() -> CampaignConfig:
    default = CampaignConfig(
        name="best_config_zewma_s5_c1p3",
        label="zewma_s5_c1.3",
        role="best_config",
        z_entry=1.8,
        z_window=60,
        max_holding_days=30,
        entry_mode="entry_zspeed_ewma_cap",
        zspeed_ewma_span=5,
        zspeed_ewma_cap=1.3,
        notes="Best recent Sweden speed-filter config mapped to current engine API.",
    )
    if not LEGACY_BEST_CONFIG_PATH.exists():
        return default

    try:
        row = pd.read_csv(LEGACY_BEST_CONFIG_PATH).iloc[0]
    except Exception:
        return default

    span = _safe_int(row.get("entry_speed_ewma_span"), default.zspeed_ewma_span)
    cap = _safe_float(row.get("entry_speed_ewma_cap"), default.zspeed_ewma_cap)
    z_entry = _safe_float(row.get("z_entry"), default.z_entry)
    z_window = _safe_int(row.get("z_window"), default.z_window)
    max_hold = _safe_int(row.get("max_holding_days"), default.max_holding_days)
    label = str(row.get("name", default.label))

    return CampaignConfig(
        name=f"best_config_{label.replace('.', 'p')}",
        label=label,
        role="best_config",
        z_entry=float(z_entry),
        z_window=int(z_window),
        max_holding_days=int(max_hold),
        entry_mode="entry_zspeed_ewma_cap",
        zspeed_ewma_span=int(span),
        zspeed_ewma_cap=float(cap),
        notes="Loaded from recent local Sweden weekly speed-filter result and mapped to current API.",
    )


def _covers_period(scans: pd.DataFrame, *, start: str | pd.Timestamp, end: str | pd.Timestamp) -> bool:
    if scans.empty or "scan_date" not in scans.columns:
        return False
    min_dt = pd.to_datetime(scans["scan_date"], errors="coerce").min()
    max_dt = pd.to_datetime(scans["scan_date"], errors="coerce").max()
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    return pd.notna(min_dt) and pd.notna(max_dt) and min_dt <= start_ts and max_dt >= end_ts - BDay(15)


def _safe_float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    return out if np.isfinite(out) else float(fallback)


def _safe_int(value: Any, fallback: int | None) -> int:
    if fallback is None:
        fallback = 0
    try:
        out = int(float(value))
    except Exception:
        return int(fallback)
    return out


def _stats_row(stats: pd.DataFrame, config_name: str) -> pd.Series:
    row = stats[stats["config_name"] == config_name]
    if row.empty:
        return pd.Series(dtype=object)
    return row.iloc[0]


def _format_overall_delta(best: pd.Series, base: pd.Series) -> str:
    if best.empty or base.empty:
        return "- Overall comparison unavailable."
    delta_trades = _safe_float(best.get("diagnostic_nb_trades"), 0.0) - _safe_float(base.get("diagnostic_nb_trades"), 0.0)
    delta_avg = _safe_float(best.get("diagnostic_avg_pnl"), np.nan) - _safe_float(base.get("diagnostic_avg_pnl"), np.nan)
    delta_win = _safe_float(best.get("diagnostic_win_rate"), np.nan) - _safe_float(base.get("diagnostic_win_rate"), np.nan)
    delta_total = _safe_float(best.get("diagnostic_total_pnl"), np.nan) - _safe_float(base.get("diagnostic_total_pnl"), np.nan)
    return (
        "- Delta best-baseline: "
        f"trades={delta_trades:.0f}, avg_pnl/trade={delta_avg:.6f}, "
        f"win_rate={delta_win:.2%}, total_pnl={delta_total:.6f}."
    )


def _quality_vs_trade_count_diagnosis(
    best: pd.Series,
    base: pd.Series,
    comparison: pd.DataFrame,
) -> str:
    if best.empty or base.empty:
        return "Insufficient data to classify the edge source."
    best_trades = _safe_float(best.get("diagnostic_nb_trades"), 0.0)
    base_trades = max(1.0, _safe_float(base.get("diagnostic_nb_trades"), 1.0))
    delta_trades_pct = (best_trades - base_trades) / base_trades
    delta_avg = _safe_float(best.get("diagnostic_avg_pnl"), np.nan) - _safe_float(base.get("diagnostic_avg_pnl"), np.nan)
    delta_win = _safe_float(best.get("diagnostic_win_rate"), np.nan) - _safe_float(base.get("diagnostic_win_rate"), np.nan)

    baseline_only_loss = 0.0
    if not comparison.empty and "segment_presence" in comparison.columns:
        cut = comparison[comparison["segment_presence"] == "baseline_only"].copy()
        if "baseline_total_pnl" in cut.columns:
            baseline_only_loss = float(pd.to_numeric(cut["baseline_total_pnl"], errors="coerce").clip(upper=0.0).sum())

    if np.isfinite(delta_avg) and delta_avg > 0.0 and np.isfinite(delta_win) and delta_win > 0.0:
        if delta_trades_pct < -0.10 and baseline_only_loss < 0.0:
            return (
                "best_config improves average trade quality while also cutting some baseline-only losing segments; "
                "the edge appears mixed between quality improvement and bad-trade reduction."
            )
        return "best_config appears to improve average trade quality within the traded population."
    if delta_trades_pct < -0.10 and baseline_only_loss < 0.0:
        return "best_config appears to win mainly by reducing exposure to losing baseline segments."
    if delta_trades_pct < -0.10:
        return "best_config mostly reduces trade count; quality improvement is not yet clear."
    return "No clear quality or trade-count explanation dominates; inspect segment tables."


def _loss_concentration_notes(best_segments: pd.DataFrame) -> list[str]:
    if best_segments.empty:
        return ["No segment data available."]
    neg = best_segments[pd.to_numeric(best_segments["total_pnl"], errors="coerce") < 0.0].copy()
    if neg.empty:
        return ["No negative segment concentration found for best_config."]

    checks = {
        "stress regimes": neg["segment_value"].astype(str).str.contains("stress", case=False, na=False),
        "strong trends": neg["segment_value"].astype(str).str.contains("trending", case=False, na=False),
        "long half-lives": neg["segment_value"].astype(str).str.contains("long_half_life", case=False, na=False),
        "fast entries": neg["segment_value"].astype(str).str.contains("zspeed.*q5|z_speed.*q5", case=False, regex=True, na=False),
        "unstable beta buckets": neg["segment_value"].astype(str).str.contains("beta_stability.*q3", case=False, regex=True, na=False),
    }
    notes = []
    for label, mask in checks.items():
        if bool(mask.any()):
            total = float(pd.to_numeric(neg.loc[mask, "total_pnl"], errors="coerce").sum())
            notes.append(f"Potential concentration in {label}: total_pnl={total:.6f}.")
    return notes or ["No obvious stress/trend/instability concentration in the top negative segments."]


def _next_hypotheses(best_segments: pd.DataFrame, bad_pairs: pd.DataFrame) -> list[str]:
    text_values = " ".join(best_segments["segment_value"].astype(str).tolist()).lower() if not best_segments.empty else ""
    hypotheses: list[str] = []
    if "stress" in text_values:
        hypotheses.append("Test a diagnostic stress-regime exposure cap or post-stress cooldown before adding any new optimizer.")
    if "trending" in text_values:
        hypotheses.append("Check whether entries against strong 20d market trends have structurally worse MAE/MFE.")
    if "zspeed" in text_values or "speed" in text_values:
        hypotheses.append("Audit high z-speed entry buckets to see if the EWMA cap should be treated as a risk-control candidate.")
    if "long_half_life" in text_values:
        hypotheses.append("Validate whether long half-life pairs should be excluded or sized down in a later robustness study.")
    if not bad_pairs.empty:
        pair = str(bad_pairs.iloc[0].get("pair_id", "top losing pair"))
        hypotheses.append(f"Run a pair-level case study on {pair} before considering any pair blacklist rule.")
    while len(hypotheses) < 3:
        hypotheses.append("Compare best_config and baseline on common segments only to separate quality uplift from trade-count effects.")
    return hypotheses[:5]


def _format_segment_rows(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["- None."]
    out = []
    for r in rows.itertuples(index=False):
        out.append(
            "- "
            f"{r.segment_type}={r.segment_value} | trades={int(r.nb_trades)} | "
            f"total_pnl={float(r.total_pnl):.6f} | avg_pnl={float(r.avg_pnl):.6f} | "
            f"win_rate={float(r.win_rate):.2%}"
        )
    return out


def _format_pair_rows(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["- None."]
    out = []
    for r in rows.itertuples(index=False):
        out.append(
            "- "
            f"{r.pair_id} | trades={int(r.nb_trades)} | total_pnl={float(r.total_pnl):.6f} | "
            f"avg_pnl={float(r.avg_pnl):.6f} | win_rate={float(r.win_rate):.2%}"
        )
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    out_dir = build_output_dir(
        output_root=Path(args.output_root),
        start=args.start,
        end=args.end,
        suffix=args.output_suffix,
    )
    LOGGER.info("Output directory: %s", out_dir)

    scans = load_or_build_scans(start=args.start, end=args.end, rebuild=bool(args.rebuild_scans))
    if scans.empty:
        raise RuntimeError("No Sweden scans available.")
    configs = build_campaign_configs(include_extra=not bool(args.no_extra_config))

    assets = build_universe_assets(scans)
    LOGGER.info("Loading Sweden price panel for %d assets.", len(assets))
    price_panel = load_price_panel(assets, DATA_PATH, start=args.start, end=args.end, buffer_days=520)
    if price_panel.empty:
        raise RuntimeError("No price panel available for Sweden diagnostics.")

    market_features = compute_market_regime_features(price_panel)
    asset_metadata = pd.read_csv(ASSET_REGISTRY_PATH) if ASSET_REGISTRY_PATH.exists() else pd.DataFrame()

    runs: list[dict[str, Any]] = []
    enriched_frames: list[pd.DataFrame] = []
    for config in configs:
        run = run_config(config=config, scans=scans, start=args.start, end=args.end)
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

    best_config_name = next(c.name for c in configs if c.role == "best_config")
    baseline_config_name = next(c.name for c in configs if c.role == "baseline")
    paths = write_outputs(
        out_dir=out_dir,
        enriched=enriched_all,
        runs=runs,
        best_config_name=best_config_name,
        baseline_config_name=baseline_config_name,
        start=args.start,
        end=args.end,
    )

    LOGGER.info("Campaign completed.")
    for name, path in paths.items():
        LOGGER.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
