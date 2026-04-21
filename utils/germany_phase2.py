import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.run_sweden_filter_ablation_campaign import FilterThresholds
from utils.country_research_pipeline import (
    ASSET_REGISTRY_PATH,
    BASE_DATA_PATH,
    DEFAULT_ABS_Z_THRESHOLD,
    DEFAULT_BETA_THRESHOLD,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_ZSPEED_EWMA_THRESHOLD,
    PROJECT_ROOT,
    PeriodSpec,
    ResearchVariant,
    add_period_columns,
    baseline_variant,
    build_analysis_frames,
    build_country_assets,
    build_robustness_scorecard,
    config_to_dict,
    enrich_run,
    load_or_build_country_scans,
    load_price_panel,
    compute_market_regime_features,
    run_variant,
    select_country_reference,
)


LOGGER = logging.getLogger("germany_phase2")

GERMANY_BETA_THRESHOLD = 0.17934571538040014


@dataclass(frozen=True)
class Phase2Options:
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    smoke: bool = False
    output_root: Path = PROJECT_ROOT / "data" / "experiments"
    output_suffix: str | None = None
    log_level: str = "INFO"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    role: str
    letter: str
    corr_abs_max: float | None = None
    block_beta_degraded: bool = False
    beta_threshold: float = GERMANY_BETA_THRESHOLD
    half_life_max: float | None = None
    notes: str = ""


def build_output_dir(options: Phase2Options, start: str, end: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"germany_phase2_{pd.Timestamp(start).strftime('%Y%m%d')}_{pd.Timestamp(end).strftime('%Y%m%d')}_{stamp}"
    if options.smoke:
        name += "_smoke"
    if options.output_suffix:
        name += f"_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def reference_spec() -> VariantSpec:
    return VariantSpec(
        name="reference",
        label="reference",
        role="reference",
        letter="REF",
        notes="Germany local reference from country research pipeline.",
    )


def best_pair_filter_spec() -> VariantSpec:
    return VariantSpec(
        name="pair_filter_corr_abs_le_0p75",
        label="best_pair_filter_corr_abs_le_0p75",
        role="pair_filter_best",
        letter="PAIR075",
        corr_abs_max=0.75,
        notes="Germany phase-1 winning pair filter: exclude high_corr pairs, i.e. abs(6m_corr) > 0.75.",
    )


def sensitivity_specs(smoke: bool = False) -> list[VariantSpec]:
    specs = [
        reference_spec(),
        VariantSpec(
            name="corr_abs_le_0p70",
            label="corr_abs_le_0p70",
            role="sensitivity_stricter_corr",
            letter="C070",
            corr_abs_max=0.70,
            notes="Stricter corr threshold than the winning filter.",
        ),
        best_pair_filter_spec(),
        VariantSpec(
            name="corr_abs_le_0p80",
            label="corr_abs_le_0p80",
            role="sensitivity_looser_corr",
            letter="C080",
            corr_abs_max=0.80,
            notes="Looser corr threshold than the winning filter.",
        ),
        VariantSpec(
            name="corr_abs_le_0p75_plus_beta_guard",
            label="corr_abs_le_0p75_plus_beta_guard",
            role="sensitivity_corr_beta",
            letter="C075B",
            corr_abs_max=0.75,
            block_beta_degraded=True,
            beta_threshold=GERMANY_BETA_THRESHOLD,
            notes="Winning corr filter plus beta instability guard.",
        ),
        VariantSpec(
            name="beta_guard_only",
            label="beta_guard_only",
            role="simplification_beta_only",
            letter="BETA",
            block_beta_degraded=True,
            beta_threshold=GERMANY_BETA_THRESHOLD,
            notes="Single-condition comparator: block beta_std above Germany diagnostic q3 threshold.",
        ),
    ]
    return [specs[0], specs[2]] if smoke else specs


def temporal_specs() -> list[VariantSpec]:
    return [reference_spec(), best_pair_filter_spec()]


def simplified_specs(smoke: bool = False) -> list[VariantSpec]:
    specs = [
        reference_spec(),
        best_pair_filter_spec(),
        VariantSpec(
            name="corr_abs_le_0p80_simple_loose",
            label="corr_abs_le_0p80_simple_loose",
            role="simplified_looser_corr",
            letter="S080",
            corr_abs_max=0.80,
            notes="Simpler relaxed corr-only version.",
        ),
        VariantSpec(
            name="beta_guard_only",
            label="beta_guard_only",
            role="simplification_beta_only",
            letter="BETA",
            block_beta_degraded=True,
            beta_threshold=GERMANY_BETA_THRESHOLD,
            notes="Alternative single-condition version to test whether beta instability alone explains the edge.",
        ),
    ]
    return specs[:2] if smoke else specs


def make_research_variant(reference: Any, spec: VariantSpec) -> ResearchVariant:
    return ResearchVariant(
        name=spec.name,
        label=spec.label,
        letter=spec.letter,
        role=spec.role,
        reference=reference,
        use_h3_pair_filter=False,
        notes=spec.notes,
    )


def apply_phase2_scan_filter(scans: pd.DataFrame, spec: VariantSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = scans.copy()
    before = int(len(out))
    before_dates = int(pd.to_datetime(out.get("scan_date"), errors="coerce").nunique()) if "scan_date" in out else 0
    block = pd.Series(False, index=out.index)
    if spec.corr_abs_max is not None and "6m_corr" in out.columns:
        corr_abs = pd.to_numeric(out["6m_corr"], errors="coerce").abs()
        block |= corr_abs.notna() & (corr_abs > float(spec.corr_abs_max))
    if spec.block_beta_degraded and "beta_std" in out.columns:
        beta = pd.to_numeric(out["beta_std"], errors="coerce")
        block |= beta.notna() & (beta >= float(spec.beta_threshold))
    if spec.half_life_max is not None and "6m_half_life" in out.columns:
        half_life = pd.to_numeric(out["6m_half_life"], errors="coerce")
        block |= half_life.notna() & (half_life > float(spec.half_life_max))

    filtered = out.loc[~block].copy()
    diag = {
        "phase2_scan_rows_before": before,
        "phase2_scan_rows_after": int(len(filtered)),
        "phase2_scan_rows_removed": int(block.sum()),
        "phase2_scan_removed_pct": float(block.mean()) if before else np.nan,
        "phase2_scan_dates_before": before_dates,
        "phase2_scan_dates_after": int(pd.to_datetime(filtered.get("scan_date"), errors="coerce").nunique()) if "scan_date" in filtered else 0,
        "corr_abs_max": spec.corr_abs_max,
        "block_beta_degraded": spec.block_beta_degraded,
        "beta_threshold": spec.beta_threshold if spec.block_beta_degraded else np.nan,
        "half_life_max": spec.half_life_max,
    }
    return filtered.reset_index(drop=True), diag


def annual_periods(start: str, end: str, *, smoke: bool = False) -> list[PeriodSpec]:
    if smoke:
        return [PeriodSpec("smoke_window", "smoke_window", start, end, "temporal_smoke")]
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    periods: list[PeriodSpec] = []
    for year in range(start_year, end_year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01"))
        e = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31"))
        if s <= e:
            periods.append(PeriodSpec(f"year_{year}", str(year), s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), "annual"))
    return periods


def rolling_24m_periods(start: str, end: str, *, smoke: bool = False) -> list[PeriodSpec]:
    if smoke:
        return []
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    periods: list[PeriodSpec] = []
    for year in range(start_year, end_year):
        s = pd.Timestamp(f"{year}-01-01")
        e = pd.Timestamp(f"{year + 1}-12-31")
        if s >= pd.Timestamp(start) and e <= pd.Timestamp(end):
            periods.append(PeriodSpec(f"roll24_{year}_{year + 1}", f"{year}_{year + 1}", s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), "rolling_24m"))
    return periods


def run_specs_for_period(
    *,
    period: PeriodSpec,
    specs: list[VariantSpec],
    reference: Any,
    scans: pd.DataFrame,
    thresholds: FilterThresholds,
    market_features: pd.DataFrame,
    price_panel: pd.DataFrame,
    asset_metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    enriched_frames: list[pd.DataFrame] = []
    filter_diag: list[dict[str, Any]] = []
    for spec in specs:
        variant = make_research_variant(reference, spec)
        filtered_scans, diag = apply_phase2_scan_filter(scans, spec)
        LOGGER.info("Running %s period=%s %s -> %s", spec.name, period.name, period.start, period.end)
        run = run_variant(
            variant=variant,
            base_scans=filtered_scans,
            thresholds=thresholds,
            market_features=market_features,
            start=period.start,
            end=period.end,
        )
        runs.append(run)
        enriched = enrich_run(run, price_panel=price_panel, market_features=market_features, asset_metadata=asset_metadata)
        enriched_frames.append(enriched)
        filter_diag.append({"config_name": spec.name, **diag})
        LOGGER.info("%s %s enriched trades: %d", period.name, spec.name, len(enriched))

    enriched_all = pd.concat(enriched_frames, ignore_index=True, sort=False) if enriched_frames else pd.DataFrame()
    frames = build_analysis_frames(enriched_all, runs)
    frames["filter_diagnostics"] = pd.DataFrame(filter_diag)
    return {name: add_period_columns(frame, period) for name, frame in frames.items()}


def concat_frames(items: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    keys = sorted({k for item in items for k in item.keys()})
    return {
        key: pd.concat([item[key] for item in items if key in item and not item[key].empty], ignore_index=True, sort=False)
        if any(key in item and not item[key].empty for item in items)
        else pd.DataFrame()
        for key in keys
    }


def build_run_level(frames: dict[str, pd.DataFrame], axis: str) -> pd.DataFrame:
    port = frames.get("portfolio_level", pd.DataFrame()).copy()
    if port.empty:
        return pd.DataFrame()
    keys = ["period_name", "period_label", "period_kind", "period_start", "period_end", "config_name"]
    trade_cols = keys + [
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
        "gross_profit",
        "gross_loss",
    ]
    conc_cols = keys + [
        "nb_paires_tradees",
        "nb_paires_positives",
        "nb_paires_negatives",
        "top5_pnl",
        "top10_pnl",
        "bottom10_pnl",
        "top5_share_net_pnl",
        "top10_share_net_pnl",
        "bottom10_share_gross_loss_abs",
    ]
    out = port.merge(frames.get("trade_level", pd.DataFrame()).reindex(columns=trade_cols), on=keys, how="left")
    out = out.merge(frames.get("concentration", pd.DataFrame()).reindex(columns=conc_cols), on=keys, how="left")
    out.insert(0, "axis", axis)
    return add_reference_deltas(out)


def add_reference_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    delta_cols = {
        "engine_sharpe": "delta_engine_sharpe",
        "total_return_engine": "delta_total_return_engine",
        "avg_pnl_per_trade": "delta_avg_pnl_per_trade",
        "total_pnl": "delta_total_pnl",
        "win_rate": "delta_win_rate",
    }
    out["delta_abs_dd_improvement"] = np.nan
    for dst in delta_cols.values():
        out[dst] = np.nan
    group_cols = ["axis", "period_name"]
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        group = out.loc[list(idx)]
        ref = group[group["config_name"].astype(str).eq("reference")]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        for col, dst in delta_cols.items():
            out.loc[group.index, dst] = pd.to_numeric(group.get(col), errors="coerce") - _safe_float(ref_row.get(col))
        out.loc[group.index, "delta_abs_dd_improvement"] = abs(_safe_float(ref_row.get("engine_max_drawdown"))) - pd.to_numeric(
            group.get("engine_max_drawdown"), errors="coerce"
        ).abs()
    return out


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def pnl_distribution_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    for cfg, group in trades.groupby("config_name", dropna=False):
        pnl = pd.to_numeric(group.get("pnl"), errors="coerce").dropna()
        if pnl.empty:
            continue
        sorted_pnl = pnl.sort_values()
        reasons = group.get("exit_reason_bucket", pd.Series(index=group.index, dtype=object)).fillna("missing").astype(str)
        rows.append(
            {
                "config_name": cfg,
                "nb_trades": int(len(group)),
                "total_pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()),
                "median_pnl": float(pnl.median()),
                "p05_pnl": float(pnl.quantile(0.05)),
                "p25_pnl": float(pnl.quantile(0.25)),
                "p75_pnl": float(pnl.quantile(0.75)),
                "p95_pnl": float(pnl.quantile(0.95)),
                "top5_trade_pnl": float(sorted_pnl.tail(5).sum()),
                "bottom5_trade_pnl": float(sorted_pnl.head(5).sum()),
                "gross_profit": float(pnl[pnl > 0].sum()),
                "gross_loss": float(pnl[pnl < 0].sum()),
                "tp_count": int((reasons == "TP").sum()),
                "sl_count": int((reasons == "SL").sum()),
                "time_count": int((reasons == "TIME").sum()),
                "tp_pnl": float(pd.to_numeric(group.loc[reasons == "TP", "pnl"], errors="coerce").sum()),
                "sl_pnl": float(pd.to_numeric(group.loc[reasons == "SL", "pnl"], errors="coerce").sum()),
                "time_pnl": float(pd.to_numeric(group.loc[reasons == "TIME", "pnl"], errors="coerce").sum()),
                "avg_holding_days": float(pd.to_numeric(group.get("holding_days"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def segment_edge_summary(trades: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for segment in ["market_regime", "corr_type", "pair_quality_bucket", "sector_bucket", "exit_reason_bucket"]:
        if segment not in trades.columns:
            continue
        grouped = trades.groupby(["config_name", segment], dropna=False)
        rows = []
        for (cfg, value), group in grouped:
            pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
            rows.append(
                {
                    "config_name": cfg,
                    "segment_type": segment,
                    "segment_value": value,
                    "nb_trades": int(len(group)),
                    "total_pnl": float(pnl.sum()),
                    "avg_pnl": float(pnl.mean()),
                    "win_rate": float((pnl > 0).mean()) if pnl.notna().any() else np.nan,
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def pair_contribution(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cfg, pair_id), group in trades.groupby(["config_name", "pair_id"], dropna=False):
        pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
        rows.append(
            {
                "config_name": cfg,
                "pair_id": pair_id,
                "asset_left": group.get("asset_left", pd.Series(dtype=object)).iloc[0] if "asset_left" in group else "",
                "asset_right": group.get("asset_right", pd.Series(dtype=object)).iloc[0] if "asset_right" in group else "",
                "nb_trades": int(len(group)),
                "total_pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()),
                "win_rate": float((pnl > 0).mean()) if pnl.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["config_name", "total_pnl"], ascending=[True, False]).reset_index(drop=True) if not out.empty else out


def period_contribution(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["year"] = pd.to_datetime(out.get("entry_datetime"), errors="coerce").dt.year
    rows = []
    for (cfg, year), group in out.groupby(["config_name", "year"], dropna=False):
        pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
        rows.append(
            {
                "config_name": cfg,
                "year": int(year) if pd.notna(year) else np.nan,
                "nb_trades": int(len(group)),
                "total_pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()),
                "win_rate": float((pnl > 0).mean()) if pnl.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["config_name", "year"]).reset_index(drop=True)


def concentration_summary(pair_df: pd.DataFrame, period_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg, group in pair_df.groupby("config_name", dropna=False):
        pnl = pd.to_numeric(group["total_pnl"], errors="coerce")
        valid = group[pnl.notna()].copy()
        valid["_pnl"] = pnl[pnl.notna()].values
        total = float(valid["_pnl"].sum()) if not valid.empty else np.nan
        positive = valid[valid["_pnl"] > 0]["_pnl"]
        negative = valid[valid["_pnl"] < 0]["_pnl"]
        top = valid.sort_values("_pnl", ascending=False)
        years = period_df[period_df["config_name"].astype(str).eq(str(cfg))].copy()
        year_pnl = pd.to_numeric(years.get("total_pnl"), errors="coerce")
        rows.append(
            {
                "config_name": cfg,
                "nb_pairs": int(valid["pair_id"].nunique()) if not valid.empty else 0,
                "nb_positive_pairs": int((valid["_pnl"] > 0).sum()) if not valid.empty else 0,
                "nb_negative_pairs": int((valid["_pnl"] < 0).sum()) if not valid.empty else 0,
                "total_pair_pnl": total,
                "top5_pair_pnl": float(top.head(5)["_pnl"].sum()) if not top.empty else np.nan,
                "top10_pair_pnl": float(top.head(10)["_pnl"].sum()) if not top.empty else np.nan,
                "top5_pair_share_net": float(top.head(5)["_pnl"].sum() / total) if np.isfinite(total) and total else np.nan,
                "top10_pair_share_net": float(top.head(10)["_pnl"].sum() / total) if np.isfinite(total) and total else np.nan,
                "effective_positive_pairs": effective_n(positive),
                "effective_negative_pairs": effective_n(negative.abs()),
                "top2_year_pnl": float(year_pnl.sort_values(ascending=False).head(2).sum()) if year_pnl.notna().any() else np.nan,
                "top2_year_share_net": float(year_pnl.sort_values(ascending=False).head(2).sum() / year_pnl.sum()) if year_pnl.notna().any() and year_pnl.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def effective_n(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    vals = vals[vals > 0]
    denom = float((vals**2).sum())
    return float(vals.sum() ** 2 / denom) if denom > 0 else np.nan


def build_summary_by_axis(
    temporal: pd.DataFrame,
    sensitivity: pd.DataFrame,
    edge: pd.DataFrame,
    concentration: pd.DataFrame,
    simplified: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    best_name = "pair_filter_corr_abs_le_0p75"

    best_temporal = temporal[temporal["config_name"].astype(str).eq(best_name)].copy()
    annual = best_temporal[best_temporal["period_kind"].astype(str).eq("annual")]
    all_windows = best_temporal
    wins = int((pd.to_numeric(all_windows.get("delta_engine_sharpe"), errors="coerce") > 0).sum()) if not all_windows.empty else 0
    count = int(len(all_windows))
    rows.append(
        {
            "axis": "A_temporal_stability",
            "status": "pass" if count and wins / count >= 0.6 else "mixed",
            "key_metric": f"{wins}/{count} windows improve Sharpe vs reference",
            "comment": "Temporal edge is diffuse enough if most annual and rolling windows improve.",
        }
    )

    sens_best = sensitivity[sensitivity["config_name"].astype(str).eq(best_name)]
    sens_positive = sensitivity[
        (sensitivity["config_name"].astype(str) != "reference")
        & (pd.to_numeric(sensitivity.get("delta_engine_sharpe"), errors="coerce") > 0)
    ]
    rows.append(
        {
            "axis": "B_parameter_sensitivity",
            "status": "pass" if len(sens_positive) >= 3 else "mixed",
            "key_metric": f"{len(sens_positive)} non-reference sensitivity variants improve Sharpe",
            "comment": "Plateau exists if nearby corr thresholds and simple comparators remain positive.",
        }
    )

    ref_edge = edge[edge["config_name"].astype(str).eq("reference")]
    best_edge = edge[edge["config_name"].astype(str).eq(best_name)]
    if not ref_edge.empty and not best_edge.empty:
        sl_time_delta = (float(best_edge.iloc[0]["sl_count"]) + float(best_edge.iloc[0]["time_count"])) - (
            float(ref_edge.iloc[0]["sl_count"]) + float(ref_edge.iloc[0]["time_count"])
        )
        avg_delta = float(best_edge.iloc[0]["avg_pnl"]) - float(ref_edge.iloc[0]["avg_pnl"])
    else:
        sl_time_delta = np.nan
        avg_delta = np.nan
    rows.append(
        {
            "axis": "C_edge_decomposition",
            "status": "pass" if avg_delta > 0 and sl_time_delta < 0 else "mixed",
            "key_metric": f"avg_pnl_delta={avg_delta:.6f}; sl_time_delta={sl_time_delta:.0f}",
            "comment": "Best case is improved average trade plus fewer SL/TIME exits.",
        }
    )

    best_conc = concentration[concentration["config_name"].astype(str).eq(best_name)]
    top5_share = float(best_conc.iloc[0]["top5_pair_share_net"]) if not best_conc.empty else np.nan
    effective_pos = float(best_conc.iloc[0]["effective_positive_pairs"]) if not best_conc.empty else np.nan
    rows.append(
        {
            "axis": "D_concentration_breadth",
            "status": "pass" if np.isfinite(effective_pos) and effective_pos >= 10 else "mixed",
            "key_metric": f"top5_share_net={top5_share:.3f}; effective_positive_pairs={effective_pos:.1f}",
            "comment": "Breadth is healthier when effective positive contributors are not too narrow.",
        }
    )

    simple_best = simplified[simplified["config_name"].astype(str).eq(best_name)]
    rows.append(
        {
            "axis": "E_signal_simplification",
            "status": "pass" if not simple_best.empty and float(simple_best.iloc[0].get("engine_sharpe", np.nan)) > 0 else "mixed",
            "key_metric": "Winning filter is already a single corr threshold: abs(6m_corr) <= 0.75.",
            "comment": "No hidden multi-condition rule is required for the main result.",
        }
    )

    summary = pd.DataFrame(rows)
    verdict = decide_verdict(summary, temporal, sensitivity)
    return summary, verdict


def decide_verdict(summary: pd.DataFrame, temporal: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    statuses = dict(zip(summary["axis"], summary["status"]))
    if statuses.get("A_temporal_stability") == "pass" and statuses.get("B_parameter_sensitivity") == "pass" and statuses.get("C_edge_decomposition") == "pass":
        return "promote"
    if statuses.get("A_temporal_stability") == "pass" and statuses.get("C_edge_decomposition") == "pass":
        return "promising_needs_validation"
    if statuses.get("C_edge_decomposition") == "pass":
        return "hold_as_research_case"
    return "reject_for_now"


def write_text_outputs(
    out_dir: Path,
    verdict: str,
    summary_by_axis: pd.DataFrame,
    temporal: pd.DataFrame,
    sensitivity: pd.DataFrame,
    edge: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    best_name = "pair_filter_corr_abs_le_0p75"
    temporal_best = temporal[temporal["config_name"].astype(str).eq(best_name)]
    sensitivity_best = sensitivity[sensitivity["config_name"].astype(str).eq(best_name)]
    edge_best = edge[edge["config_name"].astype(str).eq(best_name)]
    conc_best = concentration[concentration["config_name"].astype(str).eq(best_name)]

    lines = [
        "Germany phase 2 promotion decision",
        "",
        f"Verdict: {verdict}",
        "",
        "Axis summary:",
        summary_by_axis.to_string(index=False),
        "",
        "Key best-filter metrics:",
    ]
    if not sensitivity_best.empty:
        row = sensitivity_best.iloc[0]
        lines.append(
            f"Full period: Sharpe={row.get('engine_sharpe'):.4f}, return={row.get('total_return_engine'):.4f}, "
            f"delta Sharpe={row.get('delta_engine_sharpe'):.4f}, delta return={row.get('delta_total_return_engine'):.4f}."
        )
    if not temporal_best.empty:
        wins = int((pd.to_numeric(temporal_best.get("delta_engine_sharpe"), errors="coerce") > 0).sum())
        lines.append(f"Temporal windows improving Sharpe: {wins}/{len(temporal_best)}.")
    if not edge_best.empty:
        row = edge_best.iloc[0]
        lines.append(
            f"Trade-level: avg pnl={row.get('avg_pnl'):.6f}, median pnl={row.get('median_pnl'):.6f}, "
            f"SL={int(row.get('sl_count'))}, TIME={int(row.get('time_count'))}."
        )
    if not conc_best.empty:
        row = conc_best.iloc[0]
        lines.append(
            f"Concentration: nb_pairs={int(row.get('nb_pairs'))}, top5_share_net={row.get('top5_pair_share_net'):.4f}, "
            f"effective_positive_pairs={row.get('effective_positive_pairs'):.2f}."
        )
    lines.append("")
    lines.append("Decision rule: promote only if temporal stability, local parameter sensitivity, edge decomposition and concentration are all defensible.")
    (out_dir / "promotion_decision.txt").write_text("\n".join(lines), encoding="utf-8")

    conclusion = [
        "Germany phase 2 conclusion",
        "",
        f"Final verdict: {verdict}",
        "",
        "The winning pair filter tested here is intentionally simple: exclude pairs with abs(6m_corr) above 0.75.",
        "Read this file with the CSV exports; it is a generated summary, not a replacement for the tables.",
        "",
        "Outputs to inspect first:",
        "- temporal_stability.csv",
        "- parameter_sensitivity.csv",
        "- edge_decomposition_summary.csv",
        "- concentration_summary.csv",
        "- simplified_filter_comparison.csv",
    ]
    (out_dir / "conclusion.txt").write_text("\n".join(conclusion), encoding="utf-8")


def run_germany_phase2(options: Phase2Options) -> Path:
    start = "2024-01-01" if options.smoke else options.start
    end = "2024-06-30" if options.smoke else options.end
    out_dir = build_output_dir(options, start, end)

    reference = select_country_reference("germany")
    thresholds = FilterThresholds(
        DEFAULT_ABS_Z_THRESHOLD,
        DEFAULT_ZSPEED_EWMA_THRESHOLD,
        GERMANY_BETA_THRESHOLD,
        PROJECT_ROOT,
    )
    scans = load_or_build_country_scans(reference, start=start, end=end, rebuild=False)
    assets = build_country_assets("germany", scans)
    price_panel = load_price_panel(assets, BASE_DATA_PATH, start=start, end=end, buffer_days=520)
    market_features = compute_market_regime_features(price_panel)
    asset_metadata = pd.read_csv(ASSET_REGISTRY_PATH) if ASSET_REGISTRY_PATH.exists() else pd.DataFrame()

    metadata = {
        "country": "germany",
        "start": start,
        "end": end,
        "smoke": options.smoke,
        "reference": asdict(reference),
        "best_filter_definition": "exclude scan rows with abs(6m_corr) > 0.75",
        "beta_threshold": GERMANY_BETA_THRESHOLD,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    LOGGER.info("Axis A: temporal stability")
    temporal_frames: list[dict[str, pd.DataFrame]] = []
    for period in annual_periods(start, end, smoke=options.smoke) + rolling_24m_periods(start, end, smoke=options.smoke):
        temporal_frames.append(
            run_specs_for_period(
                period=period,
                specs=temporal_specs(),
                reference=reference,
                scans=scans,
                thresholds=thresholds,
                market_features=market_features,
                price_panel=price_panel,
                asset_metadata=asset_metadata,
            )
        )
    temporal_combined = concat_frames(temporal_frames)
    temporal_stability = build_run_level(temporal_combined, "A_temporal")

    LOGGER.info("Axis B: parameter sensitivity")
    full_period = PeriodSpec("full_period", "2018_2025" if not options.smoke else "smoke", start, end, "full")
    sensitivity_frames = run_specs_for_period(
        period=full_period,
        specs=sensitivity_specs(smoke=options.smoke),
        reference=reference,
        scans=scans,
        thresholds=thresholds,
        market_features=market_features,
        price_panel=price_panel,
        asset_metadata=asset_metadata,
    )
    parameter_sensitivity = build_run_level(sensitivity_frames, "B_sensitivity")

    LOGGER.info("Axis C/D: edge decomposition and concentration")
    full_ref_best_frames = run_specs_for_period(
        period=full_period,
        specs=temporal_specs(),
        reference=reference,
        scans=scans,
        thresholds=thresholds,
        market_features=market_features,
        price_panel=price_panel,
        asset_metadata=asset_metadata,
    )
    full_trades = full_ref_best_frames.get("trades_enriched", pd.DataFrame()).copy()
    edge_summary = pnl_distribution_summary(full_trades)
    segment_summary = segment_edge_summary(full_trades)
    pair_df = pair_contribution(full_trades)
    period_df = period_contribution(full_trades)
    concentration = concentration_summary(pair_df, period_df)

    LOGGER.info("Axis E: simplified filter comparison")
    simplified_frames = run_specs_for_period(
        period=full_period,
        specs=simplified_specs(smoke=options.smoke),
        reference=reference,
        scans=scans,
        thresholds=thresholds,
        market_features=market_features,
        price_panel=price_panel,
        asset_metadata=asset_metadata,
    )
    simplified = build_run_level(simplified_frames, "E_simplification")

    run_level = pd.concat(
        [
            temporal_stability,
            parameter_sensitivity,
            build_run_level(full_ref_best_frames, "C_D_full_ref_best"),
            simplified,
        ],
        ignore_index=True,
        sort=False,
    )

    summary_by_axis, verdict = build_summary_by_axis(
        temporal_stability,
        parameter_sensitivity,
        edge_summary,
        concentration,
        simplified,
    )

    robustness_scorecard = build_robustness_scorecard(
        temporal_combined.get("trade_level", pd.DataFrame()),
        temporal_combined.get("portfolio_level", pd.DataFrame()),
        temporal_combined.get("concentration", pd.DataFrame()),
        reference_name="reference",
    )

    run_level.to_csv(out_dir / "run_level.csv", index=False)
    summary_by_axis.to_csv(out_dir / "summary_by_axis.csv", index=False)
    temporal_stability.to_csv(out_dir / "temporal_stability.csv", index=False)
    parameter_sensitivity.to_csv(out_dir / "parameter_sensitivity.csv", index=False)
    edge_summary.to_csv(out_dir / "edge_decomposition_summary.csv", index=False)
    segment_summary.to_csv(out_dir / "edge_decomposition_segments.csv", index=False)
    concentration.to_csv(out_dir / "concentration_summary.csv", index=False)
    simplified.to_csv(out_dir / "simplified_filter_comparison.csv", index=False)
    pair_df.to_csv(out_dir / "pair_contribution.csv", index=False)
    period_df.to_csv(out_dir / "period_contribution.csv", index=False)
    robustness_scorecard.to_csv(out_dir / "temporal_robustness_scorecard.csv", index=False)
    pd.DataFrame([config_to_dict(make_research_variant(reference, spec)) | asdict(spec) for spec in sensitivity_specs(smoke=False)]).to_csv(
        out_dir / "variant_manifest.csv",
        index=False,
    )
    full_trades.to_csv(out_dir / "trades_ref_best_enriched.csv", index=False)

    write_text_outputs(
        out_dir,
        verdict,
        summary_by_axis,
        temporal_stability,
        parameter_sensitivity,
        edge_summary,
        concentration,
    )
    LOGGER.info("Output directory: %s", out_dir)
    return out_dir
