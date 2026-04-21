from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from backtesting.engine import run_daily_portfolio_engine
from backtesting.global_loop import (
    _get_or_build_global_context,
    _get_or_build_pair_states_for_window,
)
from object.class_file import BatchConfig, StrategyParams
from scripts.run_cross_sectional_robust_research import (
    ASSET_REGISTRY_PATH,
    BASE_DATA_PATH,
    DEFAULT_FEES,
    DEFAULT_TOP_N,
    FamilyConfig,
    build_strategy_params as build_cross_sectional_strategy_params,
    normalize_scans,
)
from scripts.run_sweden_edge_decomposition_campaign import (
    BASE_FEES as SWEDEN_BASE_FEES,
    BASE_MAX_POSITIONS as SWEDEN_BASE_MAX_POSITIONS,
    BASE_SELECTION_MODE as SWEDEN_BASE_SELECTION_MODE,
    BASE_SELECTION_VARIANT as SWEDEN_BASE_SELECTION_VARIANT,
    BASE_SIGNAL_SPACE as SWEDEN_BASE_SIGNAL_SPACE,
    BASE_TOP_N as SWEDEN_BASE_TOP_N,
    load_or_build_scans as load_or_build_sweden_scans,
)
from scripts.run_sweden_filter_ablation_campaign import (
    FilterCounters,
    FilterThresholds,
    annotate_scan_quality,
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


LOGGER = logging.getLogger("country_research_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RESULTS_DIR = PROJECT_ROOT / "data" / "experiments" / "robust_cross_sectional_long_2015_2025"
BEST_BY_COUNTRY_PATH = REFERENCE_RESULTS_DIR / "best_by_country_2015_2025.csv"
COUNTRY_SUMMARY_PATH = REFERENCE_RESULTS_DIR / "country_summary_all.csv"

DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"
SMOKE_START = "2025-01-01"
SMOKE_END = "2025-03-31"

DEFAULT_BETA_THRESHOLD = 0.221501
DEFAULT_ABS_Z_THRESHOLD = 3.144060
DEFAULT_ZSPEED_EWMA_THRESHOLD = 0.933395

MARKET_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
)
ENTRY_SEGMENT_COLS = (
    "abs_z_entry_quintile",
    "z_speed_1d_quintile",
    "z_speed_ewma_quintile",
    "spread_vol_20d_bucket",
)
PAIR_SEGMENT_COLS = (
    "corr_type",
    "beta_stability_bucket",
    "half_life_type",
    "half_life_6m_bucket",
    "nb_windows_passed_bucket",
    "corr_6m_abs_bucket",
    "recent_corr_drop_bucket",
)
STRUCTURE_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
    "corr_type",
    "beta_stability_bucket",
    "abs_z_entry_quintile",
    "z_speed_ewma_quintile",
    "exit_reason_bucket",
)


@dataclass(frozen=True)
class CountryReference:
    country: str
    reference_name: str
    source_experiment: str
    family: str
    variant: str
    z_entry: float
    z_window: int
    max_holding_days: int
    signal_space: str
    selection_mode: str
    selection_score_variant: str = "baseline"
    max_positions: int = 1
    top_n_candidates: int = DEFAULT_TOP_N
    fees: float = DEFAULT_FEES
    entry_mode: str = "baseline_entry"
    scan_source: str = "cross_sectional_monthly"
    scan_frequency: str = "daily"
    scan_weekday: str = "FRI"
    zspeed_ewma_span: int | None = None
    zspeed_ewma_cap: float | None = None
    pair_return_cap: float | None = None
    trade_return_isolated_cap: float | None = None
    portfolio_vol_target: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class ResearchVariant:
    name: str
    label: str
    letter: str
    role: str
    reference: CountryReference
    use_h1_regime_filter: bool = False
    use_h2_entry_filter: bool = False
    use_h3_pair_filter: bool = False
    h3_block_corr_types: tuple[str, ...] = ()
    h3_block_beta_degraded: bool = False
    h3_half_life_max: float | None = None
    entry_mode: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class PeriodSpec:
    name: str
    label: str
    start: str
    end: str
    kind: str


@dataclass(frozen=True)
class PipelineOptions:
    country: str
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    reference_name: str | None = None
    output_root: Path = PROJECT_ROOT / "data" / "experiments"
    output_suffix: str | None = None
    skip_robustness: bool = False
    smoke: bool = False
    rebuild_scans: bool = False
    max_ablation_variants: int = 5


def default_splits(start: str, end: str, *, smoke: bool = False) -> tuple[PeriodSpec, ...]:
    if smoke:
        return (PeriodSpec("smoke", "smoke", start, end, "split"),)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    requested = (
        PeriodSpec("split_1_old", "2018_2020", "2018-01-01", "2020-12-31", "split"),
        PeriodSpec("split_2_mid", "2021_2023", "2021-01-01", "2023-12-31", "split"),
        PeriodSpec("split_3_recent", "2024_2025", "2024-01-01", "2025-12-31", "split"),
    )
    if start_ts <= pd.Timestamp("2018-01-01") and end_ts >= pd.Timestamp("2025-12-31"):
        return requested

    boundaries = pd.date_range(start_ts, end_ts, periods=4)
    out: list[PeriodSpec] = []
    for i in range(3):
        s = boundaries[i].normalize()
        e = boundaries[i + 1].normalize()
        if i > 0:
            s = s + pd.Timedelta(days=1)
        out.append(PeriodSpec(f"split_{i + 1}", f"split_{i + 1}", s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), "split"))
    return tuple(out)


def select_country_reference(country: str, reference_name: str | None = None) -> CountryReference:
    country_l = str(country).lower().strip()
    ref_name = str(reference_name or "auto").lower().strip()
    if country_l == "sweden" and ref_name in {"auto", "sweden_c", "sweden_regime_c"}:
        return sweden_c_reference()
    if ref_name in {"sweden_c", "sweden_regime_c"}:
        raise ValueError("The sweden_c reference can only be used with --country sweden.")
    return cross_sectional_reference(country_l, reference_name=reference_name)


def sweden_c_reference() -> CountryReference:
    return CountryReference(
        country="sweden",
        reference_name="sweden_best_plus_regime_filter",
        source_experiment=str(PROJECT_ROOT / "data" / "experiments" / "sweden_filter_robustness_20180101_20251231_20260419_095523"),
        family="sweden_zewma",
        variant="best_plus_regime_filter",
        z_entry=1.8,
        z_window=60,
        max_holding_days=30,
        signal_space=SWEDEN_BASE_SIGNAL_SPACE,
        selection_mode=SWEDEN_BASE_SELECTION_MODE,
        selection_score_variant=SWEDEN_BASE_SELECTION_VARIANT,
        max_positions=SWEDEN_BASE_MAX_POSITIONS,
        top_n_candidates=SWEDEN_BASE_TOP_N,
        fees=SWEDEN_BASE_FEES,
        entry_mode="entry_zspeed_ewma_cap",
        scan_source="sweden_weekly",
        scan_frequency="weekly",
        scan_weekday="FRI",
        zspeed_ewma_span=5,
        zspeed_ewma_cap=1.3,
        notes="Local Sweden reference promoted by the prior robustness campaign: best_plus_regime_filter.",
    )


def cross_sectional_reference(country: str, reference_name: str | None = None) -> CountryReference:
    country_l = str(country).lower().strip()
    source = None
    row = pd.Series(dtype=object)
    if BEST_BY_COUNTRY_PATH.exists():
        best = pd.read_csv(BEST_BY_COUNTRY_PATH)
        cand = best[best["universe"].astype(str).str.lower() == country_l]
        if not cand.empty:
            row = cand.iloc[0]
            source = BEST_BY_COUNTRY_PATH

    if row.empty and COUNTRY_SUMMARY_PATH.exists():
        summary = pd.read_csv(COUNTRY_SUMMARY_PATH)
        cand = summary[summary["universe"].astype(str).str.lower() == country_l]
        if not cand.empty:
            row = cand.sort_values("score", ascending=False).iloc[0]
            source = COUNTRY_SUMMARY_PATH

    if row.empty:
        if not country_exists(country_l):
            raise FileNotFoundError(f"No reference and no asset registry country found for country={country_l}")
        family = "raw_composite"
        variant = "fallback_raw_composite"
        z_entry = 1.8
        z_window = 100
        max_hold = 25
        source_s = "fallback_from_asset_registry"
        notes = "Fallback baseline because no prior best reference was found."
    else:
        family = str(row["family"])
        variant = str(row.get("variant", "manifest_best"))
        z_entry = float(row["z_entry"])
        z_window = int(row["z_window"])
        max_hold = int(row.get("max_hold", row.get("max_holding_days", 25)))
        source_s = str(source)
        notes = "Reference selected from existing manifests/exports without new optimization."

    if reference_name and str(reference_name).lower() not in {"auto", "cross_sectional"}:
        LOGGER.warning("reference_name=%s is not an explicit selector yet; using auto reference.", reference_name)

    fam = family_from_name(family, variant)
    return CountryReference(
        country=country_l,
        reference_name=f"{country_l}_{family}_{variant}",
        source_experiment=source_s,
        family=family,
        variant=variant,
        z_entry=z_entry,
        z_window=z_window,
        max_holding_days=max_hold,
        signal_space=fam.signal_space,
        selection_mode=fam.selection_mode,
        selection_score_variant=fam.selection_score_variant,
        max_positions=infer_max_positions(variant),
        top_n_candidates=DEFAULT_TOP_N,
        fees=DEFAULT_FEES,
        pair_return_cap=fam.pair_return_cap,
        trade_return_isolated_cap=fam.trade_return_isolated_cap,
        portfolio_vol_target=fam.portfolio_vol_target,
        notes=notes,
    )


def country_exists(country: str) -> bool:
    if not ASSET_REGISTRY_PATH.exists():
        return False
    reg = pd.read_csv(ASSET_REGISTRY_PATH, usecols=["category_id"])
    return str(country).lower() in set(reg["category_id"].astype(str).str.lower())


def family_from_name(name: str, variant: str = "") -> FamilyConfig:
    base: dict[str, FamilyConfig] = {
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
    if name not in base:
        LOGGER.warning("Unknown family=%s; falling back to raw_composite semantics.", name)
        return base["raw_composite"]
    fam = base[name]
    if "guardrail" in str(variant).lower() and fam.pair_return_cap is None:
        return replace(fam, pair_return_cap=0.05, trade_return_isolated_cap=0.20)
    return fam


def infer_max_positions(variant: str) -> int:
    v = str(variant).lower()
    if "maxpos1" in v:
        return 1
    if "maxpos2" in v:
        return 2
    return 5


def build_strategy_params(reference: CountryReference, variant: ResearchVariant) -> StrategyParams:
    if reference.family == "sweden_zewma":
        kwargs: dict[str, Any] = {
            "z_entry": float(reference.z_entry),
            "z_exit": float(reference.z_entry) / 3.0,
            "z_stop": 2.0 * float(reference.z_entry),
            "z_window": int(reference.z_window),
            "beta_mode": "static",
            "fees": float(reference.fees),
            "top_n_candidates": int(reference.top_n_candidates),
            "max_positions": int(reference.max_positions),
            "max_holding_days": int(reference.max_holding_days),
            "exec_lag_days": 1,
            "scan_frequency": reference.scan_frequency,
            "scan_weekday": reference.scan_weekday,
            "signal_space": reference.signal_space,
            "selection_mode": reference.selection_mode,
            "selection_score_variant": reference.selection_score_variant,
            "eligibility_labels": ("ELIGIBLE",),
            "entry_mode": variant.entry_mode or reference.entry_mode,
        }
        if reference.zspeed_ewma_span is not None:
            kwargs["zspeed_ewma_span"] = int(reference.zspeed_ewma_span)
        if reference.zspeed_ewma_cap is not None:
            kwargs["zspeed_ewma_cap"] = float(reference.zspeed_ewma_cap)
        return StrategyParams(**kwargs)

    fam = family_from_name(reference.family, reference.variant)
    params = build_cross_sectional_strategy_params(
        fam,
        z_entry=float(reference.z_entry),
        z_window=int(reference.z_window),
        max_hold=int(reference.max_holding_days),
    )
    return replace(
        params,
        max_positions=int(reference.max_positions),
        top_n_candidates=int(reference.top_n_candidates),
        fees=float(reference.fees),
        entry_mode=variant.entry_mode or reference.entry_mode,
        selection_score_variant=reference.selection_score_variant,
        pair_return_cap=reference.pair_return_cap,
        trade_return_isolated_cap=reference.trade_return_isolated_cap,
        portfolio_vol_target=reference.portfolio_vol_target,
    )


def load_or_build_country_scans(reference: CountryReference, *, start: str, end: str, rebuild: bool) -> pd.DataFrame:
    country = reference.country
    if reference.scan_source == "sweden_weekly":
        scan_start = "2018-01-05" if pd.Timestamp(start) <= pd.Timestamp("2018-01-05") else start
        return load_or_build_sweden_scans(start=scan_start, end=end, rebuild=rebuild)

    cache = REFERENCE_RESULTS_DIR / "scans" / f"{country}.parquet"
    if cache.exists() and not rebuild:
        scans = normalize_scans(pd.read_parquet(cache), country)
        dmin = pd.to_datetime(scans["scan_date"], errors="coerce").min()
        dmax = pd.to_datetime(scans["scan_date"], errors="coerce").max()
        if pd.notna(dmin) and pd.notna(dmax) and dmin <= pd.Timestamp(start) and dmax >= pd.Timestamp(end):
            LOGGER.info("Loading %s scan cache: %s", country, cache)
            return scans
        LOGGER.warning("Scan cache does not fully cover requested period: %s", cache)

    LOGGER.info("Building %s monthly scans from raw data.", country)
    inline_cfg = InlineScannerConfig(
        raw_data_path=BASE_DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
    )
    scans = build_scans_inline(
        universes=[country],
        start_date=start,
        end_date=end,
        # Use "M" for compatibility with the pandas version pinned in this repo.
        # Newer pandas accepts "ME", but this environment does not.
        freq="M",
        cfg=inline_cfg,
        print_every=20,
    )
    scans = normalize_scans(scans, country)
    if not scans.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        scans.to_parquet(cache, index=False)
        LOGGER.info("Saved %s scan cache: %s", country, cache)
    return scans


def build_country_assets(country: str, scans: pd.DataFrame) -> list[str]:
    assets: set[str] = set()
    if ASSET_REGISTRY_PATH.exists():
        reg = pd.read_csv(ASSET_REGISTRY_PATH)
        if {"category_id", "asset"}.issubset(reg.columns):
            assets.update(
                reg.loc[reg["category_id"].astype(str).str.lower() == str(country).lower(), "asset"]
                .astype(str)
                .str.upper()
                .tolist()
            )
    if not scans.empty:
        assets.update(scans["asset_1"].astype(str).str.upper().tolist())
        assets.update(scans["asset_2"].astype(str).str.upper().tolist())
    return sorted(assets)


def segment_scans(scans: pd.DataFrame, *, start: str | pd.Timestamp, end: str | pd.Timestamp, buffer_bdays: int = 30) -> pd.DataFrame:
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    buffer_start = (start_ts - BDay(int(buffer_bdays))).normalize()
    out = scans.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"], errors="coerce").dt.normalize()
    return out[(out["scan_date"] >= buffer_start) & (out["scan_date"] <= end_ts)].reset_index(drop=True)


def baseline_variant(reference: CountryReference) -> ResearchVariant:
    return ResearchVariant(
        name="reference",
        label="reference",
        letter="REF",
        role="reference",
        reference=reference,
        use_h1_regime_filter=reference.variant == "best_plus_regime_filter",
        notes="Frozen local reference.",
    )


def config_to_dict(variant: ResearchVariant) -> dict[str, Any]:
    ref = variant.reference
    return {
        "name": variant.name,
        "label": variant.label,
        "letter": variant.letter,
        "role": variant.role,
        "country": ref.country,
        "reference_name": ref.reference_name,
        "source_experiment": ref.source_experiment,
        "family": ref.family,
        "reference_variant": ref.variant,
        "z_entry": ref.z_entry,
        "z_exit": ref.z_entry / 3.0,
        "z_stop": 2.0 * ref.z_entry,
        "z_window": ref.z_window,
        "max_holding_days": ref.max_holding_days,
        "top_n_candidates": ref.top_n_candidates,
        "max_positions": ref.max_positions,
        "fees": ref.fees,
        "signal_space": ref.signal_space,
        "selection_mode": ref.selection_mode,
        "selection_score_variant": ref.selection_score_variant,
        "entry_mode": variant.entry_mode or ref.entry_mode,
        "use_h1_regime_filter": variant.use_h1_regime_filter,
        "use_h2_entry_filter": variant.use_h2_entry_filter,
        "use_h3_pair_filter": variant.use_h3_pair_filter,
        "h3_block_corr_types": "|".join(variant.h3_block_corr_types),
        "h3_block_beta_degraded": variant.h3_block_beta_degraded,
        "h3_half_life_max": variant.h3_half_life_max,
        "notes": variant.notes,
    }


def run_variant(
    *,
    variant: ResearchVariant,
    base_scans: pd.DataFrame,
    thresholds: FilterThresholds,
    market_features: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    params = build_strategy_params(variant.reference, variant)
    cfg = BatchConfig(data_path=BASE_DATA_PATH, start_date=pd.Timestamp(start), end_date=pd.Timestamp(end))
    run_scans = segment_scans(base_scans, start=start, end=end)
    h3_diag: dict[str, Any] = {}
    if variant.use_h3_pair_filter:
        run_scans, h3_diag = apply_h3_scan_filter(run_scans, variant, thresholds)

    ctx = _get_or_build_global_context(cfg=cfg, params=params, universes=[variant.reference.country], scans=run_scans)
    if ctx is None:
        raise RuntimeError(f"No global context returned for {variant.name}")
    pair_state_cache = _get_or_build_pair_states_for_window(ctx, int(params.z_window))
    counters = FilterCounters()

    scan_lookup = build_scan_feature_lookup(run_scans, thresholds)
    market_lookup = build_market_regime_lookup(market_features)
    get_ranked_pairs = make_filtered_ranked_pairs_fn(
        config=variant,
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
        raise RuntimeError(f"No engine result for {variant.name}")
    result = finalize_engine_result(raw=raw, ctx=ctx, params=params, config=variant)
    result["filter_diagnostics"] = pd.DataFrame([{"config_name": variant.name, **h3_diag, **counters.as_dict()}])
    return {"config": variant, "params": params, "result": result, "scans": run_scans}


def apply_h3_scan_filter(
    scans: pd.DataFrame,
    variant: ResearchVariant,
    thresholds: FilterThresholds,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotated = annotate_scan_quality(scans, thresholds)
    before_rows = int(len(annotated))
    block = pd.Series(False, index=annotated.index)
    if variant.h3_block_corr_types:
        block |= annotated["_corr_type"].astype(str).isin(list(variant.h3_block_corr_types))
    if variant.h3_block_beta_degraded:
        block |= annotated["_beta_stability_degraded"].fillna(False).astype(bool)
    if variant.h3_half_life_max is not None and "6m_half_life" in annotated.columns:
        half_life = pd.to_numeric(annotated["6m_half_life"], errors="coerce")
        block |= half_life.notna() & (half_life > float(variant.h3_half_life_max))
    if not (variant.h3_block_corr_types or variant.h3_block_beta_degraded or variant.h3_half_life_max is not None):
        block |= annotated["_corr_type"].astype(str).eq("medium_corr") | annotated["_beta_stability_degraded"].fillna(False).astype(bool)

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
        "h3_scan_dates_before": int(pd.to_datetime(annotated["scan_date"], errors="coerce").nunique()),
        "h3_scan_dates_after": int(pd.to_datetime(filtered["scan_date"], errors="coerce").nunique()),
    }
    return filtered.reset_index(drop=True), diag


def enrich_run(
    run: dict[str, Any],
    *,
    price_panel: pd.DataFrame,
    market_features: pd.DataFrame,
    asset_metadata: pd.DataFrame,
) -> pd.DataFrame:
    variant = run["config"]
    ref = variant.reference
    return build_trade_diagnostics(
        trades=run["result"]["trades"],
        config_name=variant.name,
        params=run["params"],
        scans=run["scans"],
        scan_usage=run["result"].get("scan_usage", pd.DataFrame()),
        price_panel=price_panel,
        market_features=market_features,
        ranking_mode=f"{ref.selection_mode}:{ref.selection_score_variant}",
        asset_metadata=asset_metadata,
    )


def run_variants_for_period(
    *,
    period: PeriodSpec,
    variants: list[ResearchVariant],
    scans: pd.DataFrame,
    thresholds: FilterThresholds,
    price_panel: pd.DataFrame,
    market_features: pd.DataFrame,
    asset_metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    enriched_frames: list[pd.DataFrame] = []
    for variant in variants:
        LOGGER.info("Running %s period=%s %s -> %s", variant.name, period.name, period.start, period.end)
        run = run_variant(
            variant=variant,
            base_scans=scans,
            thresholds=thresholds,
            market_features=market_features,
            start=period.start,
            end=period.end,
        )
        runs.append(run)
        enriched = enrich_run(run, price_panel=price_panel, market_features=market_features, asset_metadata=asset_metadata)
        enriched_frames.append(enriched)
        LOGGER.info("%s %s enriched trades: %d", period.name, variant.name, len(enriched))

    enriched_all = pd.concat(enriched_frames, ignore_index=True, sort=False) if enriched_frames else pd.DataFrame()
    if enriched_all.empty:
        raise RuntimeError(f"No enriched trades for period={period.name}")

    out = build_analysis_frames(enriched_all, runs)
    return {name: add_period_columns(frame, period) for name, frame in out.items()}


def build_analysis_frames(enriched: pd.DataFrame, runs: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    segment_breakdown = summarize_edge_by_segment(enriched, STRUCTURE_SEGMENT_COLS)
    variant_vs_ref = pd.DataFrame()
    if not segment_breakdown.empty and enriched["config_name"].nunique() > 1:
        configs = [r["config"].name for r in runs]
        ref_name = configs[0]
        for cfg_name in configs[1:]:
            comp = compare_configs_by_segment(segment_breakdown, best_config=cfg_name, baseline_config=ref_name)
            if not comp.empty:
                comp.insert(0, "variant_config", cfg_name)
                variant_vs_ref = pd.concat([variant_vs_ref, comp], ignore_index=True, sort=False)

    return {
        "trades_enriched": enriched,
        "trade_level": build_trade_level(enriched),
        "portfolio_level": pd.DataFrame([r["result"]["stats"] for r in runs]),
        "concentration": build_concentration(enriched),
        "regime_breakdown": summarize_edge_by_segment(enriched, MARKET_SEGMENT_COLS),
        "entry_quality": summarize_edge_by_segment(enriched, ENTRY_SEGMENT_COLS),
        "pair_quality": summarize_edge_by_segment(enriched, PAIR_SEGMENT_COLS),
        "exit_behavior": summarize_edge_by_segment(enriched, ("exit_reason_bucket",)),
        "pair_level": build_pair_level_summary(enriched),
        "segment_breakdown": segment_breakdown,
        "monthly_returns": build_monthly_output(runs),
        "slot_utilization": build_slot_utilization(runs),
        "filter_diagnostics": pd.concat(
            [r["result"].get("filter_diagnostics", pd.DataFrame()) for r in runs],
            ignore_index=True,
            sort=False,
        ),
        "variant_vs_reference": variant_vs_ref,
    }


def build_trade_level(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_name, group in enriched.groupby("config_name", dropna=False):
        pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
        pnl_valid = pnl.dropna()
        holding = pd.to_numeric(group.get("holding_days"), errors="coerce")
        reasons = group.get("exit_reason_bucket", group.get("exit_reason", pd.Series(index=group.index, dtype=object))).fillna("missing").astype(str)
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


def build_monthly_output(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        variant = run["config"]
        monthly = run["result"].get("monthly", pd.DataFrame()).copy()
        if monthly.empty:
            continue
        monthly.insert(0, "config_name", variant.name)
        monthly.insert(1, "variant", variant.letter)
        frames.append(monthly)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_slot_utilization(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        variant = run["config"]
        equity = run["result"].get("equity", pd.DataFrame()).copy()
        if equity.empty:
            continue
        n_open = pd.to_numeric(equity.get("n_open_positions"), errors="coerce")
        rows.append(
            {
                "config_name": variant.name,
                "variant": variant.letter,
                "avg_open_positions": float(n_open.mean()) if n_open.notna().any() else np.nan,
                "median_open_positions": float(n_open.median()) if n_open.notna().any() else np.nan,
                "max_open_positions": int(n_open.max()) if n_open.notna().any() else 0,
                "pct_days_with_positions": float((n_open > 0).mean()) if n_open.notna().any() else np.nan,
                "pct_days_fully_invested": float((n_open >= variant.reference.max_positions).mean()) if n_open.notna().any() else np.nan,
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


def generate_hypotheses(diagnostic_frames: dict[str, pd.DataFrame], enriched: pd.DataFrame) -> list[dict[str, Any]]:
    trade = diagnostic_frames["trade_level"]
    portfolio = diagnostic_frames["portfolio_level"]
    total_trades = int(pd.to_numeric(trade.get("nb_trades"), errors="coerce").max()) if not trade.empty else 0
    hypotheses: list[dict[str, Any]] = []

    if total_trades < 20:
        hypotheses.append(
            {
                "id": "insufficient_signal",
                "family": "portfolio_translation_note",
                "actionable": False,
                "strength": "weak",
                "rationale": f"Only {total_trades} trades in diagnostic; ablation should stay narrow.",
                "parameters": {},
            }
        )
        return hypotheses

    regime = diagnostic_frames["regime_breakdown"]
    market = regime[regime.get("segment_type").astype(str).eq("market_regime")] if not regime.empty else pd.DataFrame()
    bad_regime = market[
        market.get("segment_value").astype(str).isin(["stress", "stress_trending"])
        & (pd.to_numeric(market.get("total_pnl"), errors="coerce") < 0.0)
    ] if not market.empty else pd.DataFrame()

    pairq = diagnostic_frames["pair_quality"]
    bad_corr_types: list[str] = []
    if not pairq.empty:
        corr = pairq[pairq.get("segment_type").astype(str).eq("corr_type")]
        bad_corr_types = (
            corr[
                (pd.to_numeric(corr.get("nb_trades"), errors="coerce") >= 5)
                & (pd.to_numeric(corr.get("total_pnl"), errors="coerce") < 0.0)
            ]["segment_value"].astype(str).tolist()
        )
    beta_thr = bucket_min(enriched, "beta_stability_bucket", "beta_stability_q3", "beta_stability_score", DEFAULT_BETA_THRESHOLD)
    beta_q3_bad = segment_is_bad(pairq, "beta_stability_bucket", "beta_stability_q3")

    if not bad_regime.empty:
        hypotheses.append(
            {
                "id": "H_regime_fragile_pairs",
                "family": "regime_filter",
                "actionable": True,
                "strength": "medium" if beta_q3_bad or "medium_corr" in bad_corr_types else "weak",
                "rationale": "stress/stress_trending regimes have negative total PnL in the diagnostic.",
                "parameters": {
                    "blocked_regimes": ["stress", "stress_trending"],
                    "fragile_corr_type": "medium_corr",
                    "beta_stability_degraded_min": beta_thr,
                },
            }
        )

    entry = diagnostic_frames["entry_quality"]
    abs_bad = segment_is_bad(entry, "abs_z_entry_quintile", "abs_z_q5")
    zspeed_bad = segment_is_bad(entry, "z_speed_ewma_quintile", "zspeed_ewma_q5")
    abs_thr = bucket_min(enriched, "abs_z_entry_quintile", "abs_z_q5", "abs_z_entry", DEFAULT_ABS_Z_THRESHOLD)
    zspeed_thr = bucket_min(enriched, "z_speed_ewma_quintile", "zspeed_ewma_q5", "z_speed_ewma", DEFAULT_ZSPEED_EWMA_THRESHOLD)
    if abs_bad or zspeed_bad:
        hypotheses.append(
            {
                "id": "H_entry_extreme_speed_or_z",
                "family": "entry_filter",
                "actionable": True,
                "strength": "medium",
                "rationale": "Extreme abs_z or z_speed_ewma bucket is destructive in the diagnostic.",
                "parameters": {
                    "abs_z_entry_min": abs_thr,
                    "zspeed_ewma_min": zspeed_thr,
                    "block_abs_z": bool(abs_bad),
                    "block_zspeed_ewma": bool(zspeed_bad),
                },
            }
        )

    bad_half_life = segment_is_bad(pairq, "half_life_type", "long_half_life") or segment_is_bad(pairq, "half_life_type", "medium_half_life")
    if bad_corr_types or beta_q3_bad or bad_half_life:
        hypotheses.append(
            {
                "id": "H_pair_quality_exclusions",
                "family": "pair_filter",
                "actionable": True,
                "strength": "medium",
                "rationale": "At least one pair-quality segment is destructive in the diagnostic.",
                "parameters": {
                    "block_corr_types": bad_corr_types,
                    "block_beta_degraded": bool(beta_q3_bad),
                    "beta_stability_degraded_min": beta_thr,
                    "half_life_max": 60.0 if bad_half_life else None,
                },
            }
        )

    exitb = diagnostic_frames["exit_behavior"]
    if not exitb.empty:
        bad_exits = exitb[
            exitb.get("segment_value").astype(str).isin(["SL", "TIME"])
            & (pd.to_numeric(exitb.get("total_pnl"), errors="coerce") < 0.0)
        ]
        if not bad_exits.empty:
            hypotheses.append(
                {
                    "id": "H_exit_control_note",
                    "family": "exit_control",
                    "actionable": False,
                    "strength": "note",
                    "rationale": "SL/TIME exits destroy PnL; this is reported but not converted into an exit-rule change by the standard pipeline.",
                    "parameters": {
                        "destructive_exits": bad_exits["segment_value"].astype(str).tolist(),
                    },
                }
            )

    if not portfolio.empty:
        p = portfolio.iloc[0]
        if _safe_float(p.get("engine_sharpe")) < 0.3 or bool(p.get("anomaly_flag", False)):
            hypotheses.append(
                {
                    "id": "H_portfolio_translation_note",
                    "family": "portfolio_translation_note",
                    "actionable": False,
                    "strength": "note",
                    "rationale": "Portfolio-level quality is weak or anomalous; promote no trade-level result without robustness.",
                    "parameters": {
                        "engine_sharpe": _safe_float(p.get("engine_sharpe")),
                        "anomaly_flag": bool(p.get("anomaly_flag", False)),
                    },
                }
            )

    actionables = [h for h in hypotheses if h.get("actionable")]
    notes = [h for h in hypotheses if not h.get("actionable")]
    if not actionables:
        hypotheses.insert(
            0,
            {
                "id": "no_clear_actionable_hypothesis",
                "family": "portfolio_translation_note",
                "actionable": False,
                "strength": "weak",
                "rationale": "No destructive standard segment was strong enough for controlled ablation.",
                "parameters": {},
            },
        )
    return actionables[:3] + notes[:2]


def segment_is_bad(summary: pd.DataFrame, segment_type: str, value: str) -> bool:
    if summary is None or summary.empty:
        return False
    d = summary[
        summary.get("segment_type").astype(str).eq(segment_type)
        & summary.get("segment_value").astype(str).eq(value)
    ]
    if d.empty:
        return False
    row = d.iloc[0]
    return _safe_float(row.get("nb_trades")) >= 5 and _safe_float(row.get("total_pnl")) < 0.0


def bucket_min(enriched: pd.DataFrame, bucket_col: str, bucket_value: str, value_col: str, fallback: float) -> float:
    if enriched.empty or bucket_col not in enriched.columns or value_col not in enriched.columns:
        return float(fallback)
    s = pd.to_numeric(enriched.loc[enriched[bucket_col].astype(str).eq(bucket_value), value_col], errors="coerce").dropna()
    if s.empty:
        return float(fallback)
    out = float(s.min())
    return out if np.isfinite(out) else float(fallback)


def thresholds_from_hypotheses(hypotheses: list[dict[str, Any]]) -> FilterThresholds:
    abs_thr = DEFAULT_ABS_Z_THRESHOLD
    zspeed_thr = DEFAULT_ZSPEED_EWMA_THRESHOLD
    beta_thr = DEFAULT_BETA_THRESHOLD
    for h in hypotheses:
        params = h.get("parameters", {})
        if "abs_z_entry_min" in params and params["abs_z_entry_min"] is not None:
            abs_thr = float(params["abs_z_entry_min"])
        if "zspeed_ewma_min" in params and params["zspeed_ewma_min"] is not None:
            zspeed_thr = float(params["zspeed_ewma_min"])
        if "beta_stability_degraded_min" in params and params["beta_stability_degraded_min"] is not None:
            beta_thr = float(params["beta_stability_degraded_min"])
    return FilterThresholds(abs_thr, zspeed_thr, beta_thr, PROJECT_ROOT)


def build_ablation_variants(reference: CountryReference, hypotheses: list[dict[str, Any]], *, max_variants: int = 5) -> list[ResearchVariant]:
    variants = [baseline_variant(reference)]
    actionable = [h for h in hypotheses if h.get("actionable")]
    has_regime = False
    has_entry = False
    for h in actionable:
        family = str(h.get("family"))
        params = h.get("parameters", {})
        if family == "regime_filter" and not variants[0].use_h1_regime_filter:
            variants.append(
                ResearchVariant(
                    name="reference_plus_regime_filter",
                    label="reference_plus_regime_filter",
                    letter="REGIME",
                    role="regime_filter",
                    reference=reference,
                    use_h1_regime_filter=True,
                    notes=h.get("rationale", ""),
                )
            )
            has_regime = True
        elif family == "entry_filter":
            variants.append(
                ResearchVariant(
                    name="reference_plus_entry_filter",
                    label="reference_plus_entry_filter",
                    letter="ENTRY",
                    role="entry_filter",
                    reference=reference,
                    use_h2_entry_filter=True,
                    notes=h.get("rationale", ""),
                )
            )
            has_entry = True
        elif family == "pair_filter":
            variants.append(
                ResearchVariant(
                    name="reference_plus_pair_filter",
                    label="reference_plus_pair_filter",
                    letter="PAIR",
                    role="pair_filter",
                    reference=reference,
                    use_h3_pair_filter=True,
                    h3_block_corr_types=tuple(params.get("block_corr_types") or ()),
                    h3_block_beta_degraded=bool(params.get("block_beta_degraded", False)),
                    h3_half_life_max=params.get("half_life_max"),
                    notes=h.get("rationale", ""),
                )
            )
    if has_regime and has_entry and len(variants) < max_variants:
        variants.append(
            ResearchVariant(
                name="reference_plus_regime_entry",
                label="reference_plus_regime_entry",
                letter="REGIME_ENTRY",
                role="regime_entry_combo",
                reference=reference,
                use_h1_regime_filter=True,
                use_h2_entry_filter=True,
                notes="Light combo generated because both regime and entry hypotheses were actionable.",
            )
        )
    return variants[: max(1, int(max_variants))]


def build_robustness_scorecard(trade_level: pd.DataFrame, portfolio_level: pd.DataFrame, concentration: pd.DataFrame, reference_name: str = "reference") -> pd.DataFrame:
    if trade_level.empty or portfolio_level.empty:
        return pd.DataFrame()
    ref_t = trade_level[trade_level["config_name"] == reference_name].set_index("period_name")
    ref_p = portfolio_level[portfolio_level["config_name"] == reference_name].set_index("period_name")
    rows: list[dict[str, Any]] = []
    for config_name in sorted(portfolio_level["config_name"].dropna().unique()):
        p = portfolio_level[portfolio_level["config_name"] == config_name].copy()
        t = trade_level[trade_level["config_name"] == config_name].copy()
        c = concentration[concentration["config_name"] == config_name].copy()
        sharpes = pd.to_numeric(p.get("engine_sharpe"), errors="coerce")
        returns = pd.to_numeric(p.get("total_return_engine"), errors="coerce")
        dds = pd.to_numeric(p.get("engine_max_drawdown"), errors="coerce")
        avg_pos = pd.to_numeric(p.get("avg_open_positions"), errors="coerce")
        avg_pnl = pd.to_numeric(t.get("avg_pnl_per_trade"), errors="coerce")
        breadth = pd.to_numeric(c.get("nb_paires_tradees"), errors="coerce")
        sharpe_wins = pnl_wins = avg_wins = lower_dd = 0
        for period_name in sorted(portfolio_level["period_name"].dropna().unique()):
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
    out["robustness_comment"] = out.apply(lambda r: robustness_comment(r, reference_name), axis=1)
    return out.sort_values("config_name").reset_index(drop=True)


def robustness_comment(row: pd.Series, reference_name: str) -> str:
    if str(row.get("config_name")) == reference_name:
        return "Reference."
    sharpe_wins = int(row.get("splits_outperforming_reference_on_sharpe", 0))
    pnl_wins = int(row.get("splits_outperforming_reference_on_total_pnl", 0))
    if sharpe_wins >= 3 and pnl_wins >= 2:
        return "Robust candidate versus reference."
    if sharpe_wins >= 2:
        return "Partially robust; inspect weak split."
    return "Not robust versus reference."


def decide_country_status(
    *,
    ablation_trade: pd.DataFrame,
    ablation_portfolio: pd.DataFrame,
    ablation_concentration: pd.DataFrame,
    robustness_scorecard: pd.DataFrame,
    reference_name: str = "reference",
) -> dict[str, Any]:
    if ablation_trade.empty or ablation_portfolio.empty:
        return {"decision_status": "insufficient_signal", "best_candidate": reference_name, "rationale": "Missing ablation outputs."}
    t = ablation_trade.set_index("config_name")
    p = ablation_portfolio.set_index("config_name")
    if reference_name not in t.index or reference_name not in p.index:
        return {"decision_status": "insufficient_signal", "best_candidate": reference_name, "rationale": "Missing reference row."}
    ref_p = p.loc[reference_name]
    ref_t = t.loc[reference_name]
    candidates = [c for c in p.index if c != reference_name]
    if not candidates:
        return {"decision_status": "insufficient_signal", "best_candidate": reference_name, "rationale": "No actionable ablation variant was generated."}

    rows: list[dict[str, Any]] = []
    for c in candidates:
        rows.append(
            {
                "config_name": c,
                "delta_sharpe": _safe_float(p.loc[c].get("engine_sharpe")) - _safe_float(ref_p.get("engine_sharpe")),
                "delta_return": _safe_float(p.loc[c].get("total_return_engine")) - _safe_float(ref_p.get("total_return_engine")),
                "delta_avg_pnl": _safe_float(t.loc[c].get("avg_pnl_per_trade")) - _safe_float(ref_t.get("avg_pnl_per_trade")),
                "delta_total_pnl": _safe_float(t.loc[c].get("total_pnl")) - _safe_float(ref_t.get("total_pnl")),
                "delta_sl_time": (_safe_float(ref_t.get("nb_sl")) + _safe_float(ref_t.get("nb_time"))) - (_safe_float(t.loc[c].get("nb_sl")) + _safe_float(t.loc[c].get("nb_time"))),
                "delta_tp": _safe_float(t.loc[c].get("nb_tp")) - _safe_float(ref_t.get("nb_tp")),
            }
        )
    cand = pd.DataFrame(rows).sort_values(["delta_sharpe", "delta_return", "delta_avg_pnl"], ascending=False)
    best = cand.iloc[0]
    best_name = str(best["config_name"])
    robust = pd.Series(dtype=object)
    if not robustness_scorecard.empty:
        r = robustness_scorecard[robustness_scorecard["config_name"] == best_name]
        if not r.empty:
            robust = r.iloc[0]
    sharpe_wins = int(robust.get("splits_outperforming_reference_on_sharpe", 0)) if not robust.empty else 0

    if _safe_float(best["delta_sharpe"]) > 0 and _safe_float(best["delta_return"]) > 0 and sharpe_wins >= 2:
        status = "promote" if sharpe_wins >= 3 else "promising_needs_validation"
    elif _safe_float(best["delta_sl_time"]) > 0 and _safe_float(best["delta_sharpe"]) <= 0:
        status = "risk_control_only"
    elif _safe_float(best["delta_sharpe"]) <= 0 and _safe_float(best["delta_return"]) <= 0:
        status = "rejected"
    else:
        status = "promising_needs_validation"
    return {
        "decision_status": status,
        "best_candidate": best_name,
        "best_delta_sharpe": _safe_float(best["delta_sharpe"]),
        "best_delta_return": _safe_float(best["delta_return"]),
        "best_delta_avg_pnl": _safe_float(best["delta_avg_pnl"]),
        "splits_outperforming_reference_on_sharpe": sharpe_wins,
        "rationale": "Decision follows explicit gates on portfolio improvement, trade improvement, robustness and exit-risk reduction.",
    }


def build_output_dir(options: PipelineOptions, start: str, end: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"country_research_{options.country.lower()}_{pd.Timestamp(start).strftime('%Y%m%d')}_{pd.Timestamp(end).strftime('%Y%m%d')}_{stamp}"
    if options.smoke:
        name = f"{name}_smoke"
    if options.output_suffix:
        name = f"{name}_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_country_research_pipeline(options: PipelineOptions) -> Path:
    country = options.country.lower().strip()
    start = SMOKE_START if options.smoke else options.start
    end = SMOKE_END if options.smoke else options.end
    out_dir = build_output_dir(options, start, end)

    reference = select_country_reference(country, options.reference_name)
    reference_path = out_dir / "reference_selection.json"
    reference_path.write_text(json.dumps(asdict(reference), indent=2, default=str), encoding="utf-8")

    LOGGER.info("Country=%s reference=%s", country, reference.reference_name)
    scans = load_or_build_country_scans(reference, start=start, end=end, rebuild=options.rebuild_scans)
    if scans.empty:
        raise RuntimeError(f"No scans available for country={country}")
    assets = build_country_assets(country, scans)
    price_panel = load_price_panel(assets, BASE_DATA_PATH, start=start, end=end, buffer_days=520)
    if price_panel.empty:
        raise RuntimeError(f"No price panel available for country={country}")
    market_features = compute_market_regime_features(price_panel)
    asset_metadata = pd.read_csv(ASSET_REGISTRY_PATH) if ASSET_REGISTRY_PATH.exists() else pd.DataFrame()

    # Phase 1: diagnostic on the frozen reference.
    diagnostic_period = PeriodSpec("diagnostic", "diagnostic", start, end, "diagnostic")
    seed_thresholds = FilterThresholds(DEFAULT_ABS_Z_THRESHOLD, DEFAULT_ZSPEED_EWMA_THRESHOLD, DEFAULT_BETA_THRESHOLD, PROJECT_ROOT)
    ref_variant = baseline_variant(reference)
    diagnostic = run_variants_for_period(
        period=diagnostic_period,
        variants=[ref_variant],
        scans=scans,
        thresholds=seed_thresholds,
        price_panel=price_panel,
        market_features=market_features,
        asset_metadata=asset_metadata,
    )
    write_diagnostic_outputs(out_dir, diagnostic)

    # Phase 2: local hypotheses generated from the diagnostic only.
    hypotheses = generate_hypotheses(diagnostic, diagnostic["trades_enriched"])
    (out_dir / "hypotheses_generated.json").write_text(json.dumps(hypotheses, indent=2, default=str), encoding="utf-8")
    thresholds = thresholds_from_hypotheses(hypotheses)

    # Phase 3: controlled ablation.
    variants = build_ablation_variants(reference, hypotheses, max_variants=options.max_ablation_variants)
    ablation_period = PeriodSpec("ablation", "ablation", start, end, "ablation")
    ablation = run_variants_for_period(
        period=ablation_period,
        variants=variants,
        scans=scans,
        thresholds=thresholds,
        price_panel=price_panel,
        market_features=market_features,
        asset_metadata=asset_metadata,
    )
    write_ablation_outputs(out_dir, ablation, variants)

    # Phase 4: temporal robustness. Keep the same generated variants for comparability.
    robustness_scorecard = pd.DataFrame()
    robustness_combined: dict[str, pd.DataFrame] = {}
    if not options.skip_robustness:
        split_frames: dict[str, list[pd.DataFrame]] = {k: [] for k in ablation.keys()}
        for split in default_splits(start, end, smoke=options.smoke):
            frames = run_variants_for_period(
                period=split,
                variants=variants,
                scans=scans,
                thresholds=thresholds,
                price_panel=price_panel,
                market_features=market_features,
                asset_metadata=asset_metadata,
            )
            for key, frame in frames.items():
                split_frames[key].append(frame)
        robustness_combined = {k: pd.concat(v, ignore_index=True, sort=False) if v else pd.DataFrame() for k, v in split_frames.items()}
        robustness_scorecard = build_robustness_scorecard(
            robustness_combined["trade_level"],
            robustness_combined["portfolio_level"],
            robustness_combined["concentration"],
            reference_name="reference",
        )
        write_robustness_outputs(out_dir, robustness_combined, robustness_scorecard)
    else:
        write_empty_robustness_outputs(out_dir)

    # Phase 5: country decision.
    decision = decide_country_status(
        ablation_trade=ablation["trade_level"],
        ablation_portfolio=ablation["portfolio_level"],
        ablation_concentration=ablation["concentration"],
        robustness_scorecard=robustness_scorecard,
        reference_name="reference",
    )
    scorecard = build_country_scorecard(reference, hypotheses, ablation, robustness_scorecard, decision)
    scorecard.to_csv(out_dir / "country_research_scorecard.csv", index=False)

    write_metadata(out_dir, options, reference, start, end, hypotheses, thresholds, variants)
    write_text_summaries(out_dir, reference, hypotheses, ablation, robustness_scorecard, decision)
    return out_dir


def write_diagnostic_outputs(out_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    mapping = {
        "trade_level": "diagnostic_trade_level.csv",
        "portfolio_level": "diagnostic_portfolio_level.csv",
        "regime_breakdown": "diagnostic_regime_breakdown.csv",
        "entry_quality": "diagnostic_entry_quality.csv",
        "pair_quality": "diagnostic_pair_quality.csv",
        "exit_behavior": "diagnostic_exit_behavior.csv",
        "concentration": "diagnostic_concentration.csv",
        "pair_level": "diagnostic_pair_level_summary.csv",
        "trades_enriched": "trades_enriched.csv",
        "monthly_returns": "monthly_returns.csv",
        "slot_utilization": "slot_utilization.csv",
    }
    for key, filename in mapping.items():
        frames.get(key, pd.DataFrame()).to_csv(out_dir / filename, index=False)


def write_ablation_outputs(out_dir: Path, frames: dict[str, pd.DataFrame], variants: list[ResearchVariant]) -> None:
    pd.DataFrame([config_to_dict(v) for v in variants]).to_csv(out_dir / "ablation_manifest.csv", index=False)
    mapping = {
        "trade_level": "ablation_trade_level.csv",
        "portfolio_level": "ablation_portfolio_level.csv",
        "concentration": "ablation_concentration.csv",
        "exit_behavior": "ablation_exit_behavior.csv",
        "regime_breakdown": "ablation_regime_breakdown.csv",
        "variant_vs_reference": "variant_vs_reference_by_segment.csv",
        "filter_diagnostics": "filter_diagnostics.csv",
    }
    for key, filename in mapping.items():
        frames.get(key, pd.DataFrame()).to_csv(out_dir / filename, index=False)


def write_robustness_outputs(out_dir: Path, frames: dict[str, pd.DataFrame], scorecard: pd.DataFrame) -> None:
    mapping = {
        "trade_level": "robustness_trade_level.csv",
        "portfolio_level": "robustness_portfolio_level.csv",
        "concentration": "robustness_concentration.csv",
        "exit_behavior": "robustness_exit_behavior.csv",
    }
    for key, filename in mapping.items():
        frames.get(key, pd.DataFrame()).to_csv(out_dir / filename, index=False)
    scorecard.to_csv(out_dir / "robustness_scorecard.csv", index=False)


def write_empty_robustness_outputs(out_dir: Path) -> None:
    empty_schemas = {
        "robustness_trade_level.csv": [
            "config_name",
            "nb_trades",
            "total_pnl",
            "avg_pnl_per_trade",
            "median_pnl_per_trade",
            "win_rate",
            "avg_holding_days",
            "nb_tp",
            "nb_sl",
            "nb_time",
            "total_tp_pnl",
            "total_sl_pnl",
            "total_time_pnl",
            "period_name",
            "split_label",
            "period_start",
            "period_end",
        ],
        "robustness_portfolio_level.csv": [
            "config_name",
            "total_return_engine",
            "engine_sharpe",
            "engine_cagr",
            "engine_max_drawdown",
            "engine_volatility",
            "engine_calmar",
            "avg_open_positions",
            "fully_invested_day_pct",
            "anomaly_flag",
            "period_name",
            "split_label",
            "period_start",
            "period_end",
        ],
        "robustness_concentration.csv": [
            "config_name",
            "nb_paires_tradees",
            "nb_paires_positives",
            "nb_paires_negatives",
            "gross_profit",
            "gross_loss",
            "top5_pnl",
            "top10_pnl",
            "bottom10_pnl",
            "top5_share_net_pnl",
            "top10_share_net_pnl",
            "period_name",
            "split_label",
            "period_start",
            "period_end",
        ],
        "robustness_exit_behavior.csv": [
            "config_name",
            "segment_type",
            "segment_value",
            "nb_trades",
            "total_pnl",
            "avg_pnl",
            "period_name",
            "split_label",
            "period_start",
            "period_end",
        ],
        "robustness_scorecard.csv": [
            "config_name",
            "mean_sharpe_across_splits",
            "min_sharpe_across_splits",
            "max_sharpe_across_splits",
            "sharpe_std_across_splits",
            "mean_total_return_across_splits",
            "mean_avg_pnl_trade_across_splits",
            "mean_max_dd_across_splits",
            "mean_avg_positions_across_splits",
            "anomaly_count",
            "splits_outperforming_reference_on_sharpe",
            "splits_outperforming_reference_on_total_pnl",
            "splits_outperforming_reference_on_avg_pnl_trade",
            "splits_with_lower_dd_than_reference",
            "robustness_comment",
        ],
    }
    for filename, columns in empty_schemas.items():
        pd.DataFrame(columns=columns).to_csv(out_dir / filename, index=False)


def build_country_scorecard(
    reference: CountryReference,
    hypotheses: list[dict[str, Any]],
    ablation: dict[str, pd.DataFrame],
    robustness_scorecard: pd.DataFrame,
    decision: dict[str, Any],
) -> pd.DataFrame:
    ref_port = ablation["portfolio_level"][ablation["portfolio_level"]["config_name"] == "reference"]
    best = decision.get("best_candidate", "reference")
    best_port = ablation["portfolio_level"][ablation["portfolio_level"]["config_name"] == best]
    return pd.DataFrame(
        [
            {
                "country": reference.country,
                "reference_name": reference.reference_name,
                "n_hypotheses": len(hypotheses),
                "n_actionable_hypotheses": sum(1 for h in hypotheses if h.get("actionable")),
                "best_candidate": best,
                "decision_status": decision.get("decision_status"),
                "reference_sharpe": _first_float(ref_port, "engine_sharpe"),
                "best_candidate_sharpe": _first_float(best_port, "engine_sharpe"),
                "best_delta_sharpe": decision.get("best_delta_sharpe"),
                "best_delta_return": decision.get("best_delta_return"),
                "best_delta_avg_pnl": decision.get("best_delta_avg_pnl"),
                "splits_outperforming_reference_on_sharpe": decision.get("splits_outperforming_reference_on_sharpe"),
                "robustness_available": not robustness_scorecard.empty,
                "rationale": decision.get("rationale"),
            }
        ]
    )


def write_metadata(
    out_dir: Path,
    options: PipelineOptions,
    reference: CountryReference,
    start: str,
    end: str,
    hypotheses: list[dict[str, Any]],
    thresholds: FilterThresholds,
    variants: list[ResearchVariant],
) -> None:
    meta = {
        "country": reference.country,
        "start": start,
        "end": end,
        "options": {**asdict(options), "output_root": str(options.output_root)},
        "reference": asdict(reference),
        "regime_rules": REGIME_RULES_DESCRIPTION,
        "thresholds": {
            "abs_z_extreme_min": thresholds.abs_z_extreme_min,
            "zspeed_ewma_extreme_min": thresholds.zspeed_ewma_extreme_min,
            "beta_stability_degraded_min": thresholds.beta_stability_degraded_min,
        },
        "hypothesis_count": len(hypotheses),
        "variants": [config_to_dict(v) for v in variants],
        "decision_gates": [
            "promote requires positive portfolio Sharpe/return delta and >=3 split Sharpe wins",
            "promising_needs_validation allows positive portfolio deltas with partial robustness",
            "risk_control_only is used when SL/TIME improves but portfolio does not",
            "rejected is used when portfolio Sharpe and return do not improve",
            "insufficient_signal is used when reference or ablation evidence is missing/too thin",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def write_text_summaries(
    out_dir: Path,
    reference: CountryReference,
    hypotheses: list[dict[str, Any]],
    ablation: dict[str, pd.DataFrame],
    robustness_scorecard: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    summary_lines = [
        f"Country research pipeline: {reference.country}",
        "",
        "Reference:",
        json.dumps(asdict(reference), indent=2, default=str),
        "",
        "Generated hypotheses:",
        json.dumps(hypotheses, indent=2, default=str),
        "",
        "Ablation portfolio-level:",
        compact_table(ablation["portfolio_level"], ["config_name", "total_return_engine", "engine_sharpe", "engine_cagr", "engine_max_drawdown", "avg_open_positions", "anomaly_flag"]),
        "",
        "Ablation trade-level:",
        compact_table(ablation["trade_level"], ["config_name", "nb_trades", "total_pnl", "avg_pnl_per_trade", "win_rate", "nb_tp", "nb_sl", "nb_time"]),
        "",
        "Robustness scorecard:",
        compact_table(robustness_scorecard, ["config_name", "mean_sharpe_across_splits", "min_sharpe_across_splits", "splits_outperforming_reference_on_sharpe", "robustness_comment"]),
    ]
    (out_dir / "campaign_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    conclusion_lines = [
        "Conclusion",
        "",
        f"Decision status: {decision.get('decision_status')}",
        f"Best candidate: {decision.get('best_candidate')}",
        f"Best delta Sharpe: {decision.get('best_delta_sharpe')}",
        f"Best delta return: {decision.get('best_delta_return')}",
        f"Best delta avg pnl/trade: {decision.get('best_delta_avg_pnl')}",
        f"Split Sharpe wins vs reference: {decision.get('splits_outperforming_reference_on_sharpe')}",
        "",
        "Recommendation:",
        str(decision.get("rationale")),
    ]
    (out_dir / "conclusion.txt").write_text("\n".join(conclusion_lines) + "\n", encoding="utf-8")


def compact_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "(empty)"
    keep = [c for c in cols if c in df.columns]
    return df[keep].to_string(index=False) if keep else df.to_string(index=False)


def _first_float(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    return _safe_float(df.iloc[0].get(col))


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default
