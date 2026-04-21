import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_sweden_filter_ablation_campaign import FilterThresholds
from utils.country_research_pipeline import (
    ASSET_REGISTRY_PATH,
    BASE_DATA_PATH,
    DEFAULT_ABS_Z_THRESHOLD,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_ZSPEED_EWMA_THRESHOLD,
    PROJECT_ROOT,
    PeriodSpec,
    add_period_columns,
    build_analysis_frames,
    build_country_assets,
    build_trade_level,
    build_concentration,
    config_to_dict,
    enrich_run,
    load_or_build_country_scans,
    load_price_panel,
    compute_market_regime_features,
    run_variant,
    select_country_reference,
)
from utils.germany_phase2 import (
    GERMANY_BETA_THRESHOLD,
    VariantSpec,
    apply_phase2_scan_filter,
    best_pair_filter_spec,
    make_research_variant,
    reference_spec,
)


LOGGER = logging.getLogger("germany_phase3")


@dataclass(frozen=True)
class Phase3Options:
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    smoke: bool = False
    output_root: Path = PROJECT_ROOT / "data" / "experiments"
    output_suffix: str | None = None
    log_level: str = "INFO"


@dataclass(frozen=True)
class Phase3VariantSpec:
    base: VariantSpec
    bypass_scan_stress_trending: bool = False

    @property
    def name(self) -> str:
        return self.base.name


def fixed_pair_filter() -> Phase3VariantSpec:
    return Phase3VariantSpec(best_pair_filter_spec(), bypass_scan_stress_trending=False)


def reference_filter() -> Phase3VariantSpec:
    return Phase3VariantSpec(reference_spec(), bypass_scan_stress_trending=False)


def mitigation_filter() -> Phase3VariantSpec:
    return Phase3VariantSpec(
        VariantSpec(
            name="pair_filter_corr_abs_le_0p75_bypass_scan_stress_trending",
            label="pair_filter_corr_abs_le_0p75_bypass_scan_stress_trending",
            role="mitigation_scan_regime_bypass",
            letter="PAIR075_BYPASS_ST",
            corr_abs_max=0.75,
            notes=(
                "Mitigation diagnostic: apply corr<=0.75 except on scan dates whose latest known "
                "market regime is stress_trending."
            ),
        ),
        bypass_scan_stress_trending=True,
    )


def build_output_dir(options: Phase3Options, start: str, end: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"germany_phase3_{pd.Timestamp(start).strftime('%Y%m%d')}_{pd.Timestamp(end).strftime('%Y%m%d')}_{stamp}"
    if options.smoke:
        name += "_smoke"
    if options.output_suffix:
        name += f"_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def oos_periods(start: str, end: str, *, smoke: bool = False) -> list[PeriodSpec]:
    if smoke:
        return [PeriodSpec("holdout_2024_h1", "2024_h1", "2024-01-01", "2024-06-30", "oos_holdout_smoke")]
    periods = [
        PeriodSpec("holdout_2024_2025", "final_holdout_2024_2025", "2024-01-01", "2025-12-31", "oos_final_holdout"),
        PeriodSpec("holdout_2023_2025", "late_holdout_2023_2025", "2023-01-01", "2025-12-31", "oos_late_holdout"),
    ]
    for year in range(2021, 2026):
        periods.append(
            PeriodSpec(
                f"wf_test_{year}",
                f"walk_forward_test_{year}",
                f"{year}-01-01",
                f"{year}-12-31",
                "walk_forward_test",
            )
        )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return [p for p in periods if pd.Timestamp(p.start) >= start_ts and pd.Timestamp(p.end) <= end_ts]


def apply_phase3_scan_filter(
    scans: pd.DataFrame,
    spec: Phase3VariantSpec,
    market_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not spec.bypass_scan_stress_trending:
        filtered, diag = apply_phase2_scan_filter(scans, spec.base)
        diag["bypass_scan_stress_trending"] = False
        diag["scan_stress_trending_rows"] = np.nan
        diag["scan_stress_trending_block_bypassed"] = 0
        return filtered, diag

    out = scans.copy()
    before = int(len(out))
    out["_scan_market_regime"] = scan_date_regime(out, market_features)
    corr_abs = pd.to_numeric(out.get("6m_corr"), errors="coerce").abs()
    high_corr = corr_abs.notna() & (corr_abs > float(spec.base.corr_abs_max))
    stress_trending = out["_scan_market_regime"].astype(str).eq("stress_trending")
    block = high_corr & ~stress_trending
    filtered = out.loc[~block].drop(columns=["_scan_market_regime"], errors="ignore").copy()
    diag = {
        "phase2_scan_rows_before": before,
        "phase2_scan_rows_after": int(len(filtered)),
        "phase2_scan_rows_removed": int(block.sum()),
        "phase2_scan_removed_pct": float(block.mean()) if before else np.nan,
        "phase2_scan_dates_before": int(pd.to_datetime(out.get("scan_date"), errors="coerce").nunique()) if "scan_date" in out else 0,
        "phase2_scan_dates_after": int(pd.to_datetime(filtered.get("scan_date"), errors="coerce").nunique()) if "scan_date" in filtered else 0,
        "corr_abs_max": spec.base.corr_abs_max,
        "block_beta_degraded": False,
        "beta_threshold": np.nan,
        "half_life_max": np.nan,
        "bypass_scan_stress_trending": True,
        "scan_stress_trending_rows": int(stress_trending.sum()),
        "scan_stress_trending_block_bypassed": int((high_corr & stress_trending).sum()),
    }
    return filtered.reset_index(drop=True), diag


def scan_date_regime(scans: pd.DataFrame, market_features: pd.DataFrame) -> pd.Series:
    if scans.empty or market_features.empty or "scan_date" not in scans.columns or "datetime" not in market_features.columns:
        return pd.Series("missing", index=scans.index)
    left = scans[["scan_date"]].copy()
    left["_row_id"] = np.arange(len(left))
    left["scan_date"] = pd.to_datetime(left["scan_date"], errors="coerce").dt.normalize()
    right = market_features[["datetime", "market_regime"]].copy()
    right["datetime"] = pd.to_datetime(right["datetime"], errors="coerce").dt.normalize()
    right = right.dropna(subset=["datetime"]).sort_values("datetime")
    merged = pd.merge_asof(
        left.sort_values("scan_date"),
        right,
        left_on="scan_date",
        right_on="datetime",
        direction="backward",
    )
    merged = merged.sort_values("_row_id")
    return merged["market_regime"].fillna("missing").astype(str).reset_index(drop=True)


def run_phase3_specs_for_period(
    *,
    period: PeriodSpec,
    specs: list[Phase3VariantSpec],
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
        variant = make_research_variant(reference, spec.base)
        filtered_scans, diag = apply_phase3_scan_filter(scans, spec, market_features)
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
    out: dict[str, pd.DataFrame] = {}
    for key in keys:
        frames = [item[key] for item in items if key in item and not item[key].empty]
        out[key] = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return out


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
        "gross_profit",
        "gross_loss",
    ]
    conc_cols = keys + ["nb_paires_tradees", "nb_paires_positives", "nb_paires_negatives"]
    out = port.merge(frames.get("trade_level", pd.DataFrame()).reindex(columns=trade_cols), on=keys, how="left")
    out = out.merge(frames.get("concentration", pd.DataFrame()).reindex(columns=conc_cols), on=keys, how="left")
    out.insert(0, "axis", axis)
    return add_reference_deltas(out)


def add_reference_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["engine_sharpe", "total_return_engine", "avg_pnl_per_trade", "total_pnl", "win_rate"]:
        out[f"delta_{col}"] = np.nan
    out["delta_abs_dd_improvement"] = np.nan
    for _, idx in out.groupby(["axis", "period_name"], dropna=False).groups.items():
        group = out.loc[list(idx)]
        ref = group[group["config_name"].astype(str).eq("reference")]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        for col in ["engine_sharpe", "total_return_engine", "avg_pnl_per_trade", "total_pnl", "win_rate"]:
            out.loc[group.index, f"delta_{col}"] = pd.to_numeric(group.get(col), errors="coerce") - safe_float(ref_row.get(col))
        out.loc[group.index, "delta_abs_dd_improvement"] = abs(safe_float(ref_row.get("engine_max_drawdown"))) - pd.to_numeric(
            group.get("engine_max_drawdown"), errors="coerce"
        ).abs()
    return out


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def trade_metrics(group: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
    pnl_valid = pnl.dropna()
    reasons = group.get("exit_reason_bucket", pd.Series(index=group.index, dtype=object)).fillna("missing").astype(str)
    holding = pd.to_numeric(group.get("holding_days"), errors="coerce")
    return {
        "nb_trades": int(len(group)),
        "total_pnl": float(pnl_valid.sum()) if not pnl_valid.empty else np.nan,
        "avg_pnl_per_trade": float(pnl_valid.mean()) if not pnl_valid.empty else np.nan,
        "median_pnl_per_trade": float(pnl_valid.median()) if not pnl_valid.empty else np.nan,
        "win_rate": float((pnl_valid > 0).mean()) if not pnl_valid.empty else np.nan,
        "gross_profit": float(pnl_valid[pnl_valid > 0].sum()) if not pnl_valid.empty else 0.0,
        "gross_loss": float(pnl_valid[pnl_valid < 0].sum()) if not pnl_valid.empty else 0.0,
        "nb_tp": int((reasons == "TP").sum()),
        "nb_sl": int((reasons == "SL").sum()),
        "nb_time": int((reasons == "TIME").sum()),
        "avg_holding_days": float(holding.mean()) if holding.notna().any() else np.nan,
    }


def stress_trending_tables(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stress = trades[trades.get("market_regime", pd.Series(index=trades.index, dtype=object)).astype(str).eq("stress_trending")].copy()
    summary_rows = []
    for cfg, group in stress.groupby("config_name", dropna=False):
        summary_rows.append({"config_name": cfg, **trade_metrics(group)})
    summary = pd.DataFrame(summary_rows)

    bucket_rows = []
    for (cfg, corr_type), group in stress.groupby(["config_name", "corr_type"], dropna=False):
        bucket_rows.append({"config_name": cfg, "corr_type": corr_type, **trade_metrics(group)})

    exit_rows = []
    for (cfg, reason), group in stress.groupby(["config_name", "exit_reason_bucket"], dropna=False):
        exit_rows.append({"config_name": cfg, "exit_reason_bucket": reason, **trade_metrics(group)})

    pair_rows = []
    for (cfg, pair_id), group in stress.groupby(["config_name", "pair_id"], dropna=False):
        row = {
            "config_name": cfg,
            "pair_id": pair_id,
            "asset_left": group.get("asset_left", pd.Series(dtype=object)).iloc[0] if "asset_left" in group else "",
            "asset_right": group.get("asset_right", pd.Series(dtype=object)).iloc[0] if "asset_right" in group else "",
            **trade_metrics(group),
        }
        pair_rows.append(row)

    period_rows = []
    tmp = stress.copy()
    tmp["entry_year"] = pd.to_datetime(tmp.get("entry_datetime"), errors="coerce").dt.year
    for (cfg, year), group in tmp.groupby(["config_name", "entry_year"], dropna=False):
        period_rows.append({"config_name": cfg, "entry_year": year, **trade_metrics(group)})

    return {
        "stress_trending_summary": summary.sort_values("config_name").reset_index(drop=True) if not summary.empty else summary,
        "stress_trending_corr_bucket": pd.DataFrame(bucket_rows).sort_values(["config_name", "total_pnl"], ascending=[True, False]).reset_index(drop=True)
        if bucket_rows
        else pd.DataFrame(),
        "stress_trending_exit_breakdown": pd.DataFrame(exit_rows).sort_values(["config_name", "exit_reason_bucket"]).reset_index(drop=True)
        if exit_rows
        else pd.DataFrame(),
        "stress_trending_pair_contribution": pd.DataFrame(pair_rows).sort_values(["config_name", "total_pnl"], ascending=[True, False]).reset_index(drop=True)
        if pair_rows
        else pd.DataFrame(),
        "stress_trending_period_contribution": pd.DataFrame(period_rows).sort_values(["config_name", "entry_year"]).reset_index(drop=True)
        if period_rows
        else pd.DataFrame(),
        "oos_trade_enrichment": trades,
    }


def build_oos_summary(oos_details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if oos_details.empty:
        return pd.DataFrame()
    for cfg, group in oos_details.groupby("config_name", dropna=False):
        wins = int((pd.to_numeric(group.get("delta_engine_sharpe"), errors="coerce") > 0).sum()) if cfg != "reference" else 0
        return_wins = int((pd.to_numeric(group.get("delta_total_return_engine"), errors="coerce") > 0).sum()) if cfg != "reference" else 0
        dd_wins = int((pd.to_numeric(group.get("delta_abs_dd_improvement"), errors="coerce") > 0).sum()) if cfg != "reference" else 0
        rows.append(
            {
                "config_name": cfg,
                "nb_windows": int(len(group)),
                "windows_sharpe_improved": wins,
                "windows_return_improved": return_wins,
                "windows_dd_improved": dd_wins,
                "mean_sharpe": float(pd.to_numeric(group.get("engine_sharpe"), errors="coerce").mean()),
                "min_sharpe": float(pd.to_numeric(group.get("engine_sharpe"), errors="coerce").min()),
                "mean_return": float(pd.to_numeric(group.get("total_return_engine"), errors="coerce").mean()),
                "mean_avg_pnl_per_trade": float(pd.to_numeric(group.get("avg_pnl_per_trade"), errors="coerce").mean()),
                "total_trades_across_windows": int(pd.to_numeric(group.get("nb_trades"), errors="coerce").sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("config_name").reset_index(drop=True)


def decide_phase3(
    oos_summary: pd.DataFrame,
    oos_details: pd.DataFrame,
    stress_summary: pd.DataFrame,
    mitigation: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    best = "pair_filter_corr_abs_le_0p75"
    best_oos = oos_summary[oos_summary["config_name"].astype(str).eq(best)]
    if best_oos.empty:
        return "reject_for_now", {"reason": "Missing OOS fixed-filter row."}
    best_row = best_oos.iloc[0]
    final = oos_details[
        oos_details["period_name"].astype(str).eq("holdout_2024_2025")
        & oos_details["config_name"].astype(str).eq(best)
    ]
    final_pass = (
        not final.empty
        and safe_float(final.iloc[0].get("delta_engine_sharpe")) > 0
        and safe_float(final.iloc[0].get("delta_total_return_engine")) > 0
    )
    majority_pass = safe_float(best_row.get("windows_sharpe_improved")) >= 4

    if stress_summary.empty or "config_name" not in stress_summary.columns:
        st_ref = pd.DataFrame()
        st_best = pd.DataFrame()
    else:
        st_ref = stress_summary[stress_summary["config_name"].astype(str).eq("reference")]
        st_best = stress_summary[stress_summary["config_name"].astype(str).eq(best)]
    stress_delta = np.nan
    if not st_ref.empty and not st_best.empty:
        stress_delta = safe_float(st_best.iloc[0].get("total_pnl")) - safe_float(st_ref.iloc[0].get("total_pnl"))
    stress_risk_material = np.isfinite(stress_delta) and stress_delta < -0.5

    mit_best = mitigation[mitigation["config_name"].astype(str).eq("pair_filter_corr_abs_le_0p75_bypass_scan_stress_trending")]
    mitigation_helpful = False
    if not mit_best.empty:
        mitigation_helpful = safe_float(mit_best.iloc[0].get("delta_engine_sharpe")) > safe_float(
            mitigation[mitigation["config_name"].astype(str).eq(best)].iloc[0].get("delta_engine_sharpe")
        )

    details = {
        "final_holdout_pass": bool(final_pass),
        "majority_oos_pass": bool(majority_pass),
        "stress_delta_vs_reference": stress_delta,
        "stress_risk_material": bool(stress_risk_material),
        "mitigation_helpful_vs_pair_filter": bool(mitigation_helpful),
    }
    if final_pass and majority_pass and not stress_risk_material:
        verdict = "promote"
    elif final_pass and majority_pass and stress_risk_material:
        verdict = "promising_needs_validation"
    elif final_pass or majority_pass:
        verdict = "hold_as_research_case"
    else:
        verdict = "reject_for_now"
    return verdict, details


def write_text_outputs(out_dir: Path, verdict: str, decision_details: dict[str, Any]) -> None:
    lines = [
        "Germany phase 3 promotion decision",
        "",
        f"Verdict: {verdict}",
        "",
        "Decision details:",
        json.dumps(decision_details, indent=2, default=str),
        "",
        "Methodological note:",
        "The fixed corr<=0.75 rule is tested without retuning. OOS here is a chronological validation protocol inside the available dataset, not an external unseen dataset.",
    ]
    (out_dir / "promotion_decision.txt").write_text("\n".join(lines), encoding="utf-8")
    conclusion = [
        "Germany phase 3 conclusion",
        "",
        f"Final verdict: {verdict}",
        "",
        "Read oos_validation_summary.csv, oos_window_details.csv, stress_trending_summary.csv and mitigation_comparison.csv for the numeric basis.",
    ]
    (out_dir / "conclusion.txt").write_text("\n".join(conclusion), encoding="utf-8")


def run_germany_phase3(options: Phase3Options) -> Path:
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

    meta = {
        "country": "germany",
        "start": start,
        "end": end,
        "smoke": options.smoke,
        "reference": asdict(reference),
        "fixed_filter": "exclude scan rows with abs(6m_corr) > 0.75",
        "mitigation_tested": "bypass corr filter on scan dates mapped to stress_trending using backward-looking market regime",
        "oos_note": "chronological validation with fixed rule, no retuning; not external unseen data",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    LOGGER.info("Axis 1: OOS validation")
    oos_frames = []
    for period in oos_periods(start, end, smoke=options.smoke):
        oos_frames.append(
            run_phase3_specs_for_period(
                period=period,
                specs=[reference_filter(), fixed_pair_filter()],
                reference=reference,
                scans=scans,
                thresholds=thresholds,
                market_features=market_features,
                price_panel=price_panel,
                asset_metadata=asset_metadata,
            )
        )
    oos_combined = concat_frames(oos_frames)
    oos_details = build_run_level(oos_combined, "axis1_oos")
    oos_summary = build_oos_summary(oos_details)

    LOGGER.info("Axis 2: stress_trending diagnostic")
    full_period = PeriodSpec("full_period", "2018_2025" if not options.smoke else "smoke", start, end, "full")
    full_frames = run_phase3_specs_for_period(
        period=full_period,
        specs=[reference_filter(), fixed_pair_filter()],
        reference=reference,
        scans=scans,
        thresholds=thresholds,
        market_features=market_features,
        price_panel=price_panel,
        asset_metadata=asset_metadata,
    )
    stress_tables = stress_trending_tables(full_frames.get("trades_enriched", pd.DataFrame()))

    LOGGER.info("Mitigation diagnostic")
    mitigation_frames = run_phase3_specs_for_period(
        period=full_period,
        specs=[reference_filter(), fixed_pair_filter(), mitigation_filter()],
        reference=reference,
        scans=scans,
        thresholds=thresholds,
        market_features=market_features,
        price_panel=price_panel,
        asset_metadata=asset_metadata,
    )
    mitigation_comparison = build_run_level(mitigation_frames, "mitigation_full")
    mitigation_stress = stress_trending_tables(mitigation_frames.get("trades_enriched", pd.DataFrame()))["stress_trending_summary"]
    if not mitigation_stress.empty:
        mitigation_stress.insert(0, "axis", "mitigation_stress_trending")

    run_level = pd.concat([oos_details, build_run_level(full_frames, "axis2_full"), mitigation_comparison], ignore_index=True, sort=False)
    verdict, decision_details = decide_phase3(
        oos_summary,
        oos_details,
        stress_tables["stress_trending_summary"],
        mitigation_comparison,
    )

    run_level.to_csv(out_dir / "run_level.csv", index=False)
    oos_summary.to_csv(out_dir / "oos_validation_summary.csv", index=False)
    oos_details.to_csv(out_dir / "oos_window_details.csv", index=False)
    stress_tables["stress_trending_summary"].to_csv(out_dir / "stress_trending_summary.csv", index=False)
    stress_tables["stress_trending_corr_bucket"].to_csv(out_dir / "stress_trending_corr_bucket.csv", index=False)
    stress_tables["stress_trending_pair_contribution"].to_csv(out_dir / "stress_trending_pair_contribution.csv", index=False)
    stress_tables["stress_trending_exit_breakdown"].to_csv(out_dir / "stress_trending_exit_breakdown.csv", index=False)
    stress_tables["stress_trending_period_contribution"].to_csv(out_dir / "stress_trending_period_contribution.csv", index=False)
    mitigation_comparison.to_csv(out_dir / "mitigation_comparison.csv", index=False)
    mitigation_stress.to_csv(out_dir / "mitigation_stress_trending_summary.csv", index=False)
    oos_combined.get("trades_enriched", pd.DataFrame()).to_csv(out_dir / "oos_trade_enrichment.csv", index=False)
    pd.DataFrame(
        [
            {"name": s.name, **asdict(s.base), "bypass_scan_stress_trending": s.bypass_scan_stress_trending}
            for s in [reference_filter(), fixed_pair_filter(), mitigation_filter()]
        ]
    ).to_csv(out_dir / "variant_manifest.csv", index=False)

    decision_payload = {"verdict": verdict, **decision_details}
    (out_dir / "decision_details.json").write_text(json.dumps(decision_payload, indent=2, default=str), encoding="utf-8")
    write_text_outputs(out_dir, verdict, decision_details)
    LOGGER.info("Output directory: %s", out_dir)
    return out_dir
