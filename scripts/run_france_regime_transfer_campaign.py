from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.engine import run_daily_portfolio_engine
from backtesting.global_loop import (
    _get_or_build_global_context,
    _get_or_build_pair_states_for_window,
)
from object.class_file import BatchConfig, StrategyParams
from scripts.run_cross_sectional_robust_research import (
    BASE_DATA_PATH,
    ASSET_REGISTRY_PATH,
    DEFAULT_FEES,
    DEFAULT_TOP_N,
    FamilyConfig,
    build_strategy_params as build_cross_sectional_strategy_params,
    normalize_scans,
)
from scripts.run_sweden_filter_ablation_campaign import (
    FilterCounters,
    FilterThresholds,
    build_market_regime_lookup,
    build_scan_feature_lookup,
    finalize_engine_result,
    make_filtered_ranked_pairs_fn,
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
from utils.inline_scanner import InlineScannerConfig, build_scans_inline


LOGGER = logging.getLogger("france_regime_transfer")

UNIVERSE = "france"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"

REFERENCE_RESULTS_DIR = PROJECT_ROOT / "data" / "experiments" / "robust_cross_sectional_long_2015_2025"
REFERENCE_BEST_BY_COUNTRY = REFERENCE_RESULTS_DIR / "best_by_country_2015_2025.csv"
REFERENCE_SCAN_CACHE = REFERENCE_RESULTS_DIR / "scans" / "france.parquet"

SWEDEN_BETA_DEGRADED_THRESHOLD = 0.221501
SWEDEN_ABS_Z_EXTREME_THRESHOLD = 3.144060
SWEDEN_ZSPEED_EWMA_EXTREME_THRESHOLD = 0.933395

REGIME_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
)
STRUCTURE_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
    "corr_type",
    "beta_stability_bucket",
    "exit_reason_bucket",
)


@dataclass(frozen=True)
class FranceReference:
    source: str
    universe: str
    family: str
    variant: str
    z_entry: float
    z_window: int
    max_hold: int
    signal_space: str
    selection_mode: str
    selection_score_variant: str = "baseline"
    max_positions: int = 1
    notes: str = ""


@dataclass(frozen=True)
class TransferConfig:
    name: str
    label: str
    letter: str
    role: str
    reference: FranceReference
    use_h1_regime_filter: bool = False
    use_h2_entry_filter: bool = False
    use_h3_pair_filter: bool = False
    entry_mode: str = "baseline_entry"
    notes: str = ""


@dataclass(frozen=True)
class PeriodSpec:
    name: str
    label: str
    start: str
    end: str
    kind: str


SPLITS = (
    PeriodSpec("split_1_old", "2018_2020", "2018-01-01", "2020-12-31", "split"),
    PeriodSpec("split_2_mid", "2021_2023", "2021-01-01", "2023-12-31", "split"),
    PeriodSpec("split_3_recent", "2024_2025", "2024-01-01", "2025-12-31", "split"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="France transfer test for the Sweden C/H1 regime filter without retuning."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Full-window start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Full-window end date.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "experiments"),
        help="Root folder for campaign outputs.",
    )
    parser.add_argument("--output-suffix", default=None, help="Optional suffix appended to output directory.")
    parser.add_argument("--rebuild-scans", action="store_true", help="Rebuild France monthly scans.")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke period only.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_france_reference() -> FranceReference:
    if not REFERENCE_BEST_BY_COUNTRY.exists():
        LOGGER.warning("Missing France best-by-country file: %s", REFERENCE_BEST_BY_COUNTRY)
        return fallback_reference()

    best = pd.read_csv(REFERENCE_BEST_BY_COUNTRY)
    row = best[best["universe"].astype(str).str.lower() == UNIVERSE]
    if row.empty:
        LOGGER.warning("No France row in %s", REFERENCE_BEST_BY_COUNTRY)
        return fallback_reference()

    r = row.iloc[0]
    family = str(r["family"])
    variant = str(r.get("variant", ""))
    family_cfg = family_from_name(family)
    max_positions = infer_max_positions(variant)
    notes = (
        "Reference selected from robust_cross_sectional_long_2015_2025/best_by_country_2015_2025.csv. "
        "The variant name ablation_maxpos1 is mapped to max_positions=1; this reproduces the stored "
        "2015-2025 France metrics."
    )
    return FranceReference(
        source=str(REFERENCE_BEST_BY_COUNTRY),
        universe=UNIVERSE,
        family=family,
        variant=variant,
        z_entry=float(r["z_entry"]),
        z_window=int(r["z_window"]),
        max_hold=int(r["max_hold"]),
        signal_space=family_cfg.signal_space,
        selection_mode=family_cfg.selection_mode,
        selection_score_variant=family_cfg.selection_score_variant,
        max_positions=max_positions,
        notes=notes,
    )


def fallback_reference() -> FranceReference:
    family_cfg = family_from_name("raw_composite")
    return FranceReference(
        source="fallback_raw_composite_baseline",
        universe=UNIVERSE,
        family="raw_composite",
        variant="baseline_reference_france",
        z_entry=1.8,
        z_window=100,
        max_hold=25,
        signal_space=family_cfg.signal_space,
        selection_mode=family_cfg.selection_mode,
        selection_score_variant=family_cfg.selection_score_variant,
        max_positions=1,
        notes="Fallback because no explicit France best reference was found.",
    )


def family_from_name(name: str) -> FamilyConfig:
    mapping = {
        "raw_legacy": FamilyConfig(name="raw_legacy", signal_space="raw", selection_mode="legacy"),
        "raw_composite": FamilyConfig(name="raw_composite", signal_space="raw", selection_mode="composite_quality"),
        "idio_legacy": FamilyConfig(name="idio_legacy", signal_space="idio_pca", selection_mode="legacy"),
        "idio_composite": FamilyConfig(name="idio_composite", signal_space="idio_pca", selection_mode="composite_quality"),
        "idio_composite_pair_cap": FamilyConfig(
            name="idio_composite_pair_cap",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            pair_return_cap=0.05,
            trade_return_isolated_cap=0.20,
        ),
        "raw_composite_pair_cap": FamilyConfig(
            name="raw_composite_pair_cap",
            signal_space="raw",
            selection_mode="composite_quality",
            pair_return_cap=0.05,
            trade_return_isolated_cap=0.20,
        ),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported France reference family: {name}")
    return mapping[name]


def infer_max_positions(variant: str) -> int:
    v = str(variant).lower()
    if "maxpos1" in v:
        return 1
    if "maxpos2" in v:
        return 2
    return 5


def build_transfer_configs(reference: FranceReference) -> list[TransferConfig]:
    return [
        TransferConfig(
            name="best_reference_france",
            label="B_fr",
            letter="B_fr",
            role="best_reference_france",
            reference=reference,
            notes="France best reference selected from prior manifests/exports.",
        ),
        TransferConfig(
            name="best_plus_regime_filter_france",
            label="C_fr",
            letter="C_fr",
            role="best_reference_france_plus_sweden_C_regime_filter",
            reference=reference,
            use_h1_regime_filter=True,
            notes=(
                "Sweden C/H1 transferred without retuning: block new entries in stress/stress_trending "
                "when corr_type==medium_corr or beta_std>=0.221501."
            ),
        ),
    ]


def config_to_dict(config: TransferConfig) -> dict[str, Any]:
    ref = config.reference
    return {
        "name": config.name,
        "label": config.label,
        "letter": config.letter,
        "role": config.role,
        "universe": ref.universe,
        "reference_source": ref.source,
        "reference_family": ref.family,
        "reference_variant": ref.variant,
        "z_entry": ref.z_entry,
        "z_exit": ref.z_entry / 3.0,
        "z_stop": 2.0 * ref.z_entry,
        "z_window": ref.z_window,
        "max_holding_days": ref.max_hold,
        "top_n_candidates": DEFAULT_TOP_N,
        "max_positions": ref.max_positions,
        "fees": DEFAULT_FEES,
        "signal_space": ref.signal_space,
        "selection_mode": ref.selection_mode,
        "selection_score_variant": ref.selection_score_variant,
        "entry_mode": config.entry_mode,
        "use_h1_regime_filter": config.use_h1_regime_filter,
        "use_h2_entry_filter": config.use_h2_entry_filter,
        "use_h3_pair_filter": config.use_h3_pair_filter,
        "notes": config.notes,
    }


def build_strategy_params(config: TransferConfig) -> StrategyParams:
    ref = config.reference
    family = family_from_name(ref.family)
    params = build_cross_sectional_strategy_params(
        family,
        z_entry=float(ref.z_entry),
        z_window=int(ref.z_window),
        max_hold=int(ref.max_hold),
    )
    return replace(
        params,
        max_positions=int(ref.max_positions),
        entry_mode=str(config.entry_mode),
        selection_score_variant=str(ref.selection_score_variant),
    )


def load_or_build_france_scans(*, start: str, end: str, rebuild: bool) -> pd.DataFrame:
    if REFERENCE_SCAN_CACHE.exists() and not rebuild:
        scans = normalize_scans(pd.read_parquet(REFERENCE_SCAN_CACHE), UNIVERSE)
        dmin = pd.to_datetime(scans["scan_date"], errors="coerce").min()
        dmax = pd.to_datetime(scans["scan_date"], errors="coerce").max()
        if pd.notna(dmin) and pd.notna(dmax) and dmin <= pd.Timestamp(start) and dmax >= pd.Timestamp(end):
            LOGGER.info("Loading France scan cache: %s", REFERENCE_SCAN_CACHE)
            return scans
        LOGGER.warning("France scan cache does not fully cover requested period: %s", REFERENCE_SCAN_CACHE)

    LOGGER.info("Building France monthly scans from raw data.")
    inline_cfg = InlineScannerConfig(
        raw_data_path=BASE_DATA_PATH,
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
        freq="ME",
        cfg=inline_cfg,
        print_every=20,
    )
    return normalize_scans(scans, UNIVERSE)


def segment_scans(scans: pd.DataFrame, *, start: str | pd.Timestamp, end: str | pd.Timestamp, buffer_bdays: int = 30) -> pd.DataFrame:
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


def run_config(
    *,
    config: TransferConfig,
    base_scans: pd.DataFrame,
    thresholds: FilterThresholds,
    market_features: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    LOGGER.info("Running %s %s -> %s", config.name, start, end)
    params = build_strategy_params(config)
    cfg = BatchConfig(data_path=BASE_DATA_PATH, start_date=pd.Timestamp(start), end_date=pd.Timestamp(end))
    run_scans = segment_scans(base_scans, start=start, end=end)

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

    raw = run_daily_portfolio_engine(
        params=params,
        start=ctx.start,
        end=ctx.end,
        get_ranked_pairs=get_ranked_pairs,
        get_pair_state=get_pair_state,
    )
    if not raw:
        raise RuntimeError(f"No engine result for {config.name}")

    result = finalize_engine_result(raw=raw, ctx=ctx, params=params, config=config)
    result["filter_diagnostics"] = pd.DataFrame([{"config_name": config.name, **counters.as_dict()}])
    return {"config": config, "params": params, "result": result, "scans": run_scans}


def build_trade_level(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_name, group in enriched.groupby("config_name", dropna=False):
        pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
        pnl_valid = pnl.dropna()
        holding = pd.to_numeric(group.get("holding_days"), errors="coerce")
        reasons = group.get("exit_reason_bucket", group.get("exit_reason", pd.Series(index=group.index, dtype=object))).fillna("missing").astype(str)
        wins = pnl_valid > 0
        gross_profit = float(pnl_valid[pnl_valid > 0].sum()) if not pnl_valid.empty else 0.0
        gross_loss = float(pnl_valid[pnl_valid < 0].sum()) if not pnl_valid.empty else 0.0
        avg = float(pnl_valid.mean()) if not pnl_valid.empty else np.nan
        std = float(pnl_valid.std(ddof=1)) if len(pnl_valid) > 1 else np.nan
        rows.append(
            {
                "config_name": str(config_name),
                "nb_trades": int(len(group)),
                "nb_closed_trades": int(pnl.notna().sum()),
                "total_pnl": float(pnl_valid.sum()) if not pnl_valid.empty else np.nan,
                "avg_pnl_per_trade": avg,
                "median_pnl_per_trade": float(pnl_valid.median()) if not pnl_valid.empty else np.nan,
                "win_rate": float(wins.mean()) if not pnl_valid.empty else np.nan,
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
    return pd.DataFrame(rows).sort_values("config_name").reset_index(drop=True)


def build_concentration(enriched: pd.DataFrame) -> pd.DataFrame:
    pair_level = build_pair_level_summary(enriched)
    rows: list[dict[str, Any]] = []
    if pair_level.empty:
        return pd.DataFrame()
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
    return pd.DataFrame(rows).sort_values("config_name").reset_index(drop=True)


def build_slot_utilization(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
                "pct_days_fully_invested": float((n_open >= cfg.reference.max_positions).mean()) if n_open.notna().any() else np.nan,
                "nb_observations": int(len(equity)),
            }
        )
    return pd.DataFrame(rows)


def add_period_columns(df: pd.DataFrame, period: PeriodSpec) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "period_name", period.name)
    out.insert(1, "period_label", period.label)
    out.insert(2, "period_kind", period.kind)
    out.insert(3, "period_start", period.start)
    out.insert(4, "period_end", period.end)
    return out


def run_period(
    *,
    period: PeriodSpec,
    configs: list[TransferConfig],
    scans: pd.DataFrame,
    thresholds: FilterThresholds,
    price_panel: pd.DataFrame,
    market_features: pd.DataFrame,
    asset_metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    enriched_frames: list[pd.DataFrame] = []
    for config in configs:
        run = run_config(
            config=config,
            base_scans=scans,
            thresholds=thresholds,
            market_features=market_features,
            start=period.start,
            end=period.end,
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
            ranking_mode=f"{config.reference.selection_mode}:{config.reference.selection_score_variant}",
            asset_metadata=asset_metadata,
        )
        enriched_frames.append(enriched)
        LOGGER.info("%s %s enriched trades: %d", period.name, config.name, len(enriched))

    enriched_all = pd.concat(enriched_frames, ignore_index=True, sort=False) if enriched_frames else pd.DataFrame()
    if enriched_all.empty:
        raise RuntimeError(f"No enriched trades for period {period.name}")

    portfolio = pd.DataFrame([r["result"]["stats"] for r in runs])
    monthly = build_monthly_output(runs)
    filter_diag = pd.concat(
        [r["result"].get("filter_diagnostics", pd.DataFrame()) for r in runs],
        ignore_index=True,
        sort=False,
    )
    segment_breakdown = summarize_edge_by_segment(enriched_all, STRUCTURE_SEGMENT_COLS)

    variant_vs_reference = pd.DataFrame()
    if not segment_breakdown.empty:
        variant_vs_reference = compare_configs_by_segment(
            segment_breakdown,
            best_config="best_plus_regime_filter_france",
            baseline_config="best_reference_france",
        )

    return {
        "trades_enriched": add_period_columns(enriched_all, period),
        "trade_level": add_period_columns(build_trade_level(enriched_all), period),
        "portfolio_level": add_period_columns(portfolio, period),
        "concentration": add_period_columns(build_concentration(enriched_all), period),
        "exit_behavior": add_period_columns(summarize_edge_by_segment(enriched_all, ("exit_reason_bucket",)), period),
        "regime_exposure": add_period_columns(summarize_edge_by_segment(enriched_all, REGIME_SEGMENT_COLS), period),
        "segment_breakdown": add_period_columns(segment_breakdown, period),
        "pair_level": add_period_columns(build_pair_level_summary(enriched_all), period),
        "monthly_returns": add_period_columns(monthly, period),
        "slot_utilization": add_period_columns(build_slot_utilization(runs), period),
        "filter_diagnostics": add_period_columns(filter_diag, period),
        "variant_vs_reference": add_period_columns(variant_vs_reference, period) if not variant_vs_reference.empty else pd.DataFrame(),
    }


def build_monthly_output(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        cfg = run["config"]
        monthly = run["result"].get("monthly", pd.DataFrame()).copy()
        if monthly.empty:
            continue
        monthly.insert(0, "config_name", cfg.name)
        monthly.insert(1, "variant", cfg.letter)
        frames.append(monthly)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_robustness_scorecard(trade_level: pd.DataFrame, portfolio_level: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    split_trade = trade_level[trade_level["period_kind"] == "split"].copy()
    split_port = portfolio_level[portfolio_level["period_kind"] == "split"].copy()
    split_conc = concentration[concentration["period_kind"] == "split"].copy()
    ref_t = split_trade[split_trade["config_name"] == "best_reference_france"].set_index("period_name")
    ref_p = split_port[split_port["config_name"] == "best_reference_france"].set_index("period_name")
    rows: list[dict[str, Any]] = []
    for config_name in sorted(split_port["config_name"].dropna().unique()):
        p = split_port[split_port["config_name"] == config_name].copy()
        t = split_trade[split_trade["config_name"] == config_name].copy()
        c = split_conc[split_conc["config_name"] == config_name].copy()
        sharpes = pd.to_numeric(p.get("engine_sharpe"), errors="coerce")
        returns = pd.to_numeric(p.get("total_return_engine"), errors="coerce")
        dds = pd.to_numeric(p.get("engine_max_drawdown"), errors="coerce")
        avg_pos = pd.to_numeric(p.get("avg_open_positions"), errors="coerce")
        avg_pnl = pd.to_numeric(t.get("avg_pnl_per_trade"), errors="coerce")
        breadth = pd.to_numeric(c.get("nb_paires_tradees"), errors="coerce")

        sharpe_wins = 0
        pnl_wins = 0
        avg_wins = 0
        lower_dd = 0
        for period_name in sorted(split_port["period_name"].dropna().unique()):
            p_row = p[p["period_name"] == period_name]
            t_row = t[t["period_name"] == period_name]
            if period_name not in ref_p.index or p_row.empty:
                continue
            if _safe_float(p_row.iloc[0].get("engine_sharpe")) > _safe_float(ref_p.loc[period_name].get("engine_sharpe")):
                sharpe_wins += 1
            if abs(_safe_float(p_row.iloc[0].get("engine_max_drawdown"))) < abs(_safe_float(ref_p.loc[period_name].get("engine_max_drawdown"))):
                lower_dd += 1
            if period_name in ref_t.index and not t_row.empty:
                if _safe_float(t_row.iloc[0].get("total_pnl")) > _safe_float(ref_t.loc[period_name].get("total_pnl")):
                    pnl_wins += 1
                if _safe_float(t_row.iloc[0].get("avg_pnl_per_trade")) > _safe_float(ref_t.loc[period_name].get("avg_pnl_per_trade")):
                    avg_wins += 1

        anomaly_count = int(p.get("anomaly_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        rows.append(
            {
                "config_name": config_name,
                "variant": str(p["variant"].iloc[0]) if "variant" in p.columns and not p.empty else "",
                "mean_sharpe_across_splits": float(sharpes.mean()) if sharpes.notna().any() else np.nan,
                "min_sharpe_across_splits": float(sharpes.min()) if sharpes.notna().any() else np.nan,
                "max_sharpe_across_splits": float(sharpes.max()) if sharpes.notna().any() else np.nan,
                "sharpe_std_across_splits": float(sharpes.std(ddof=1)) if sharpes.notna().sum() > 1 else np.nan,
                "mean_total_return_across_splits": float(returns.mean()) if returns.notna().any() else np.nan,
                "mean_avg_pnl_trade_across_splits": float(avg_pnl.mean()) if avg_pnl.notna().any() else np.nan,
                "mean_max_dd_across_splits": float(dds.mean()) if dds.notna().any() else np.nan,
                "mean_avg_positions_across_splits": float(avg_pos.mean()) if avg_pos.notna().any() else np.nan,
                "mean_nb_pairs_traded_across_splits": float(breadth.mean()) if breadth.notna().any() else np.nan,
                "anomaly_count": anomaly_count,
                "splits_outperforming_reference_on_sharpe": sharpe_wins,
                "splits_outperforming_reference_on_total_pnl": pnl_wins,
                "splits_outperforming_reference_on_avg_pnl_trade": avg_wins,
                "splits_with_lower_dd_than_reference": lower_dd,
            }
        )
    out = pd.DataFrame(rows)
    out["robustness_comment"] = out.apply(_robustness_comment, axis=1)
    return out.sort_values("variant").reset_index(drop=True)


def _robustness_comment(row: pd.Series) -> str:
    if str(row.get("config_name")) == "best_reference_france":
        return "France reference."
    sharpe_wins = int(row.get("splits_outperforming_reference_on_sharpe", 0))
    pnl_wins = int(row.get("splits_outperforming_reference_on_total_pnl", 0))
    if sharpe_wins == 3 and pnl_wins >= 2:
        return "Credible transfer candidate across splits."
    if sharpe_wins >= 2:
        return "Partial transfer; inspect weak split."
    return "Transfer not robust versus France reference."


def build_output_dir(*, output_root: Path, start: str, end: str, suffix: str | None, smoke: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"france_regime_transfer_{pd.Timestamp(start).strftime('%Y%m%d')}_{pd.Timestamp(end).strftime('%Y%m%d')}_{stamp}"
    if smoke:
        name = f"{name}_smoke"
    if suffix:
        name = f"{name}_{suffix}"
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_outputs(
    *,
    out_dir: Path,
    frames_by_name: dict[str, list[pd.DataFrame]],
    scorecard: pd.DataFrame,
    configs: list[TransferConfig],
    reference: FranceReference,
    thresholds: FilterThresholds,
    periods: tuple[PeriodSpec, ...],
    full_period: PeriodSpec,
) -> dict[str, Path]:
    combined = {
        name: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for name, frames in frames_by_name.items()
    }

    manifest = pd.DataFrame([config_to_dict(c) for c in configs])
    full_filter = combined["trade_level"]["period_name"] == full_period.name

    paths = {
        "campaign_summary": out_dir / "campaign_summary.txt",
        "conclusion": out_dir / "conclusion.txt",
        "config_manifest": out_dir / "config_manifest.csv",
        "metadata": out_dir / "metadata.json",
        "comparison_trade_level": out_dir / "comparison_trade_level.csv",
        "comparison_portfolio_level": out_dir / "comparison_portfolio_level.csv",
        "comparison_concentration": out_dir / "comparison_concentration.csv",
        "comparison_exit_behavior": out_dir / "comparison_exit_behavior.csv",
        "comparison_regime_exposure": out_dir / "comparison_regime_exposure.csv",
        "split_trade_level": out_dir / "split_trade_level.csv",
        "split_portfolio_level": out_dir / "split_portfolio_level.csv",
        "split_regime_breakdown": out_dir / "split_regime_breakdown.csv",
        "split_monthly_returns": out_dir / "split_monthly_returns.csv",
        "robustness_scorecard": out_dir / "robustness_scorecard.csv",
        "split_slot_utilization": out_dir / "split_slot_utilization.csv",
        "pair_level_summary": out_dir / "pair_level_summary.csv",
        "variant_vs_reference_by_segment": out_dir / "variant_vs_reference_by_segment.csv",
        "filter_diagnostics": out_dir / "filter_diagnostics.csv",
        "trades_enriched": out_dir / "trades_enriched.csv",
        "segment_breakdown": out_dir / "segment_breakdown.csv",
    }

    combined["trade_level"].loc[full_filter].to_csv(paths["comparison_trade_level"], index=False)
    combined["portfolio_level"].loc[combined["portfolio_level"]["period_name"] == full_period.name].to_csv(paths["comparison_portfolio_level"], index=False)
    combined["concentration"].loc[combined["concentration"]["period_name"] == full_period.name].to_csv(paths["comparison_concentration"], index=False)
    combined["exit_behavior"].loc[combined["exit_behavior"]["period_name"] == full_period.name].to_csv(paths["comparison_exit_behavior"], index=False)
    combined["regime_exposure"].loc[combined["regime_exposure"]["period_name"] == full_period.name].to_csv(paths["comparison_regime_exposure"], index=False)
    combined["trade_level"].loc[~full_filter].to_csv(paths["split_trade_level"], index=False)
    combined["portfolio_level"].loc[combined["portfolio_level"]["period_kind"] == "split"].to_csv(paths["split_portfolio_level"], index=False)
    combined["regime_exposure"].loc[combined["regime_exposure"]["period_kind"] == "split"].to_csv(paths["split_regime_breakdown"], index=False)
    combined["monthly_returns"].loc[combined["monthly_returns"]["period_kind"] == "split"].to_csv(paths["split_monthly_returns"], index=False)
    scorecard.to_csv(paths["robustness_scorecard"], index=False)
    combined["slot_utilization"].to_csv(paths["split_slot_utilization"], index=False)
    combined["pair_level"].to_csv(paths["pair_level_summary"], index=False)
    combined["variant_vs_reference"].to_csv(paths["variant_vs_reference_by_segment"], index=False)
    combined["filter_diagnostics"].to_csv(paths["filter_diagnostics"], index=False)
    combined["trades_enriched"].to_csv(paths["trades_enriched"], index=False)
    combined["segment_breakdown"].to_csv(paths["segment_breakdown"], index=False)
    manifest.to_csv(paths["config_manifest"], index=False)

    metadata = {
        "universe": UNIVERSE,
        "full_start": full_period.start,
        "full_end": full_period.end,
        "periods": [full_period.__dict__, *[p.__dict__ for p in periods]],
        "reference": reference.__dict__,
        "thresholds": {
            "beta_stability_degraded_min": thresholds.beta_stability_degraded_min,
            "abs_z_extreme_min_unused": thresholds.abs_z_extreme_min,
            "zspeed_ewma_extreme_min_unused": thresholds.zspeed_ewma_extreme_min,
        },
        "filter_definition": (
            "Block new entries when market_regime in {stress, stress_trending} and "
            "(corr_type == medium_corr or beta_std >= 0.221501)."
        ),
        "regime_rules": REGIME_RULES_DESCRIPTION,
        "method_notes": [
            "No France threshold retuning was performed.",
            "France reference is loaded from previous manifests/exports, not selected by this script.",
            "The ablation_maxpos1 variant is mapped to max_positions=1.",
            "H1 is applied as an entry-time block and does not backfill beyond the current ranked candidate list.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    paths["campaign_summary"].write_text(
        build_campaign_summary(
            trade_level=combined["trade_level"],
            portfolio_level=combined["portfolio_level"],
            concentration=combined["concentration"],
            regime_exposure=combined["regime_exposure"],
            exit_behavior=combined["exit_behavior"],
            scorecard=scorecard,
            reference=reference,
            thresholds=thresholds,
        ),
        encoding="utf-8",
    )
    paths["conclusion"].write_text(
        build_conclusion(
            trade_level=combined["trade_level"],
            portfolio_level=combined["portfolio_level"],
            concentration=combined["concentration"],
            regime_exposure=combined["regime_exposure"],
            exit_behavior=combined["exit_behavior"],
            scorecard=scorecard,
        ),
        encoding="utf-8",
    )
    return paths


def build_campaign_summary(
    *,
    trade_level: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    concentration: pd.DataFrame,
    regime_exposure: pd.DataFrame,
    exit_behavior: pd.DataFrame,
    scorecard: pd.DataFrame,
    reference: FranceReference,
    thresholds: FilterThresholds,
) -> str:
    lines = [
        "France regime transfer campaign",
        "",
        "Reference France:",
        json.dumps(reference.__dict__, indent=2, default=str),
        "",
        "Transferred filter C/H1:",
        f"- stress/stress_trending AND (corr_type == medium_corr OR beta_std >= {thresholds.beta_stability_degraded_min:.6f})",
        "",
        "Regime rules:",
        f"- {REGIME_RULES_DESCRIPTION}",
        "",
        "Full-window trade-level:",
        _compact(trade_level[trade_level["period_kind"] == "full"], ["period_name", "config_name", "nb_trades", "total_pnl", "avg_pnl_per_trade", "win_rate", "nb_tp", "nb_sl", "nb_time"]),
        "",
        "Full-window portfolio-level:",
        _compact(portfolio_level[portfolio_level["period_kind"] == "full"], ["period_name", "config_name", "total_return_engine", "engine_sharpe", "engine_cagr", "engine_max_drawdown", "avg_open_positions", "anomaly_flag"]),
        "",
        "Split scorecard:",
        _compact(scorecard, ["variant", "config_name", "mean_sharpe_across_splits", "min_sharpe_across_splits", "splits_outperforming_reference_on_sharpe", "splits_outperforming_reference_on_total_pnl", "robustness_comment"]),
        "",
        "Full concentration:",
        _compact(concentration[concentration["period_kind"] == "full"], ["config_name", "nb_paires_tradees", "gross_profit", "gross_loss", "top5_share_net_pnl", "top10_share_net_pnl"]),
        "",
        "Full regime exposure:",
        _compact(regime_exposure[regime_exposure["period_kind"] == "full"], ["segment_type", "config_name", "segment_value", "nb_trades", "total_pnl", "avg_pnl", "tp_rate", "sl_rate", "timeout_rate"]),
        "",
        "Full exit behavior:",
        _compact(exit_behavior[exit_behavior["period_kind"] == "full"], ["config_name", "segment_value", "nb_trades", "total_pnl", "avg_pnl", "tp_rate", "sl_rate", "timeout_rate"]),
    ]
    return "\n".join(lines) + "\n"


def build_conclusion(
    *,
    trade_level: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    concentration: pd.DataFrame,
    regime_exposure: pd.DataFrame,
    exit_behavior: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> str:
    full_t = trade_level[trade_level["period_kind"] == "full"].set_index("config_name")
    full_p = portfolio_level[portfolio_level["period_kind"] == "full"].set_index("config_name")
    lines = ["Conclusion", ""]
    if {"best_reference_france", "best_plus_regime_filter_france"}.issubset(full_t.index) and {"best_reference_france", "best_plus_regime_filter_france"}.issubset(full_p.index):
        b_t = full_t.loc["best_reference_france"]
        c_t = full_t.loc["best_plus_regime_filter_france"]
        b_p = full_p.loc["best_reference_france"]
        c_p = full_p.loc["best_plus_regime_filter_france"]
        lines.extend(
            [
                f"Trade-level delta C_fr - B_fr: total_pnl={_safe_float(c_t.total_pnl)-_safe_float(b_t.total_pnl):.6f}, avg_pnl={_safe_float(c_t.avg_pnl_per_trade)-_safe_float(b_t.avg_pnl_per_trade):.6f}, trades={_safe_float(c_t.nb_trades)-_safe_float(b_t.nb_trades):.0f}.",
                f"Portfolio-level delta C_fr - B_fr: return={_safe_float(c_p.total_return_engine)-_safe_float(b_p.total_return_engine):.6f}, sharpe={_safe_float(c_p.engine_sharpe)-_safe_float(b_p.engine_sharpe):.6f}, max_dd={_safe_float(c_p.engine_max_drawdown)-_safe_float(b_p.engine_max_drawdown):.6f}.",
            ]
        )
    c_score = scorecard[scorecard["config_name"] == "best_plus_regime_filter_france"]
    if not c_score.empty:
        r = c_score.iloc[0]
        lines.append(
            "Split stability: "
            f"Sharpe wins={int(r.splits_outperforming_reference_on_sharpe)}/3, "
            f"trade total-pnl wins={int(r.splits_outperforming_reference_on_total_pnl)}/3, "
            f"min Sharpe={_safe_float(r.min_sharpe_across_splits):.3f}."
        )
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "- A credible transfer requires portfolio improvement and temporal stability, not only fewer trades.",
            "- If C_fr mainly reduces trades without improving split-level Sharpe/return, the Sweden mechanism is not confirmed on France.",
        ]
    )
    return "\n".join(lines) + "\n"


def _compact(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "(empty)"
    keep = [c for c in cols if c in df.columns]
    return df[keep].to_string(index=False) if keep else df.to_string(index=False)


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

    reference = load_france_reference()
    configs = build_transfer_configs(reference)
    full_period = PeriodSpec("full", "full", args.start, args.end, "full")
    periods = (PeriodSpec("smoke_2025_q1", "smoke_2025_q1", "2025-01-01", "2025-03-31", "split"),) if args.smoke else SPLITS
    out_dir = build_output_dir(output_root=Path(args.output_root), start=args.start, end=args.end, suffix=args.output_suffix, smoke=bool(args.smoke))

    thresholds = FilterThresholds(
        abs_z_extreme_min=SWEDEN_ABS_Z_EXTREME_THRESHOLD,
        zspeed_ewma_extreme_min=SWEDEN_ZSPEED_EWMA_EXTREME_THRESHOLD,
        beta_stability_degraded_min=SWEDEN_BETA_DEGRADED_THRESHOLD,
        source_dir=REFERENCE_RESULTS_DIR,
    )

    LOGGER.info("Output directory: %s", out_dir)
    LOGGER.info("France reference: %s", reference)
    LOGGER.info("Transferred beta threshold: %.6f", thresholds.beta_stability_degraded_min)

    scans = load_or_build_france_scans(start=args.start, end=args.end, rebuild=bool(args.rebuild_scans))
    if scans.empty:
        raise RuntimeError("No France scans available.")
    assets = build_universe_assets(scans)
    LOGGER.info("Loading France price panel for %d assets.", len(assets))
    price_panel = load_price_panel(assets, BASE_DATA_PATH, start=args.start, end=args.end, buffer_days=520)
    if price_panel.empty:
        raise RuntimeError("No France price panel available.")
    market_features = compute_market_regime_features(price_panel)
    asset_metadata = pd.read_csv(ASSET_REGISTRY_PATH) if ASSET_REGISTRY_PATH.exists() else pd.DataFrame()

    frames_by_name: dict[str, list[pd.DataFrame]] = {
        "trades_enriched": [],
        "trade_level": [],
        "portfolio_level": [],
        "concentration": [],
        "exit_behavior": [],
        "regime_exposure": [],
        "segment_breakdown": [],
        "pair_level": [],
        "monthly_returns": [],
        "slot_utilization": [],
        "filter_diagnostics": [],
        "variant_vs_reference": [],
    }

    for period in (full_period, *periods):
        LOGGER.info("Running period=%s %s -> %s", period.name, period.start, period.end)
        period_frames = run_period(
            period=period,
            configs=configs,
            scans=scans,
            thresholds=thresholds,
            price_panel=price_panel,
            market_features=market_features,
            asset_metadata=asset_metadata,
        )
        for name, frame in period_frames.items():
            frames_by_name[name].append(frame)

    combined_trade = pd.concat(frames_by_name["trade_level"], ignore_index=True, sort=False)
    combined_port = pd.concat(frames_by_name["portfolio_level"], ignore_index=True, sort=False)
    combined_conc = pd.concat(frames_by_name["concentration"], ignore_index=True, sort=False)
    scorecard = build_robustness_scorecard(combined_trade, combined_port, combined_conc)
    paths = write_outputs(
        out_dir=out_dir,
        frames_by_name=frames_by_name,
        scorecard=scorecard,
        configs=configs,
        reference=reference,
        thresholds=thresholds,
        periods=periods,
        full_period=full_period,
    )

    LOGGER.info("France regime transfer campaign completed.")
    for name, path in paths.items():
        LOGGER.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
