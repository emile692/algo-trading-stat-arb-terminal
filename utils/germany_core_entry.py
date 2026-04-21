from __future__ import annotations

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
    build_country_assets,
    compute_market_regime_features,
    load_or_build_country_scans,
    load_price_panel,
    select_country_reference,
)
from utils.germany_phase2 import GERMANY_BETA_THRESHOLD
from utils.germany_phase3 import (
    Phase3VariantSpec,
    build_oos_summary,
    build_run_level,
    concat_frames,
    fixed_pair_filter,
    mitigation_filter,
    oos_periods,
    reference_filter,
    run_phase3_specs_for_period,
    safe_float,
    stress_trending_tables,
)
from utils.multibook_portfolio import (
    MultibookOptions,
    align_book_returns,
    default_books,
    drawdown_contribution,
    load_book_monthly,
    portfolio_returns,
    return_metrics,
    weight_schemes_for_combo,
)


LOGGER = logging.getLogger("germany_core_entry")
EXPERIMENTS_ROOT = PROJECT_ROOT / "data" / "experiments"
GERMANY_SIMPLE = "pair_filter_corr_abs_le_0p75"
GERMANY_MITIGATION = "pair_filter_corr_abs_le_0p75_bypass_scan_stress_trending"


@dataclass(frozen=True)
class GermanyCoreEntryOptions:
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    output_root: Path = EXPERIMENTS_ROOT
    output_suffix: str | None = None
    smoke: bool = False


def build_output_dir(options: GermanyCoreEntryOptions, start: str, end: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"germany_core_entry_{pd.Timestamp(start).strftime('%Y%m%d')}_{pd.Timestamp(end).strftime('%Y%m%d')}_{stamp}"
    if options.smoke:
        name += "_smoke"
    if options.output_suffix:
        name += f"_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def final_specs() -> list[Phase3VariantSpec]:
    return [reference_filter(), fixed_pair_filter(), mitigation_filter()]


def version_manifest(specs: list[Phase3VariantSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.name == "reference":
            rule = "Germany local reference, no pair filter."
            status = "baseline_comparator"
        elif spec.name == GERMANY_SIMPLE:
            rule = "Exclude scan rows with abs(6m_corr) > 0.75."
            status = "current_simple_candidate"
        else:
            rule = "Exclude scan rows with abs(6m_corr) > 0.75 except when the latest known scan-date regime is stress_trending."
            status = "mitigation_candidate"
        rows.append(
            {
                "config_name": spec.name,
                "variant": spec.base.letter,
                "role": spec.base.role,
                "rule": rule,
                "bypass_scan_stress_trending": spec.bypass_scan_stress_trending,
                "corr_abs_max": spec.base.corr_abs_max,
                "status": status,
                "notes": spec.base.notes,
            }
        )
    return pd.DataFrame(rows)


def add_final_oos_flags(oos_details: pd.DataFrame) -> pd.DataFrame:
    if oos_details.empty:
        return oos_details
    out = oos_details.copy()
    out["beats_reference_sharpe"] = pd.to_numeric(out.get("delta_engine_sharpe"), errors="coerce") > 0
    out["beats_reference_return"] = pd.to_numeric(out.get("delta_total_return_engine"), errors="coerce") > 0
    out["lower_dd_than_reference"] = pd.to_numeric(out.get("delta_abs_dd_improvement"), errors="coerce") > 0

    keys = ["period_name", "period_label", "period_kind", "period_start", "period_end"]
    simple = out[out["config_name"].astype(str).eq(GERMANY_SIMPLE)]
    simple_cols = keys + ["engine_sharpe", "total_return_engine", "engine_max_drawdown"]
    simple = simple.reindex(columns=simple_cols).rename(
        columns={
            "engine_sharpe": "_simple_engine_sharpe",
            "total_return_engine": "_simple_total_return_engine",
            "engine_max_drawdown": "_simple_engine_max_drawdown",
        }
    )
    out = out.merge(simple, on=keys, how="left")
    out["delta_vs_simple_engine_sharpe"] = pd.to_numeric(out.get("engine_sharpe"), errors="coerce") - pd.to_numeric(
        out.get("_simple_engine_sharpe"), errors="coerce"
    )
    out["delta_vs_simple_total_return"] = pd.to_numeric(out.get("total_return_engine"), errors="coerce") - pd.to_numeric(
        out.get("_simple_total_return_engine"), errors="coerce"
    )
    out["delta_vs_simple_abs_dd_improvement"] = pd.to_numeric(out.get("engine_max_drawdown"), errors="coerce").abs().rsub(
        pd.to_numeric(out.get("_simple_engine_max_drawdown"), errors="coerce").abs()
    )
    return out.drop(columns=[c for c in out.columns if c.startswith("_simple_")])


def normalize_germany_monthly(monthly: pd.DataFrame, config_name: str, book_name: str = "germany") -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    out = monthly.copy()
    if "trade_month" not in out.columns or "month_return" not in out.columns:
        return pd.DataFrame()
    out = out[out["config_name"].astype(str).eq(config_name)].copy()
    if out.empty:
        return pd.DataFrame()
    out["trade_month"] = pd.to_datetime(out["trade_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["month_return"] = pd.to_numeric(out["month_return"], errors="coerce")
    out["book"] = book_name
    out["country"] = "germany"
    return out[["trade_month", "month_return", "config_name", "book", "country"]].dropna(
        subset=["trade_month", "month_return"]
    )


def period_filter_monthly(monthly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    out = monthly.copy()
    out["trade_month"] = pd.to_datetime(out["trade_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    start_m = pd.Timestamp(start).to_period("M").to_timestamp()
    end_m = pd.Timestamp(end).to_period("M").to_timestamp()
    return out[(out["trade_month"] >= start_m) & (out["trade_month"] <= end_m)].copy()


def load_core_monthly(options: GermanyCoreEntryOptions, out_dir: Path, start: str, end: str) -> pd.DataFrame:
    books = [book for book in default_books() if book.book in {"france", "sweden", "netherlands"}]
    mb_options = MultibookOptions(start=options.start, end=options.end, output_root=options.output_root, smoke=options.smoke)
    frames = [load_book_monthly(book, out_dir, mb_options) for book in books]
    monthly = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return period_filter_monthly(monthly, start, end)


def portfolio_impact_outputs(
    *,
    core_monthly: pd.DataFrame,
    germany_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = [
        ("core_clean", None, core_monthly),
        (
            "core_plus_germany_simple",
            GERMANY_SIMPLE,
            pd.concat(
                [core_monthly, normalize_germany_monthly(germany_monthly, GERMANY_SIMPLE)],
                ignore_index=True,
                sort=False,
            ),
        ),
        (
            "core_plus_germany_mitigation",
            GERMANY_MITIGATION,
            pd.concat(
                [core_monthly, normalize_germany_monthly(germany_monthly, GERMANY_MITIGATION)],
                ignore_index=True,
                sort=False,
            ),
        ),
    ]
    perf_rows: list[dict[str, Any]] = []
    dd_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    for portfolio_name, germany_config, monthly in scenarios:
        returns = align_book_returns(monthly)
        combo = tuple(returns.columns)
        if len(combo) < 2:
            continue
        for scheme, weights in weight_schemes_for_combo(returns, combo).items():
            pr = portfolio_returns(returns, weights)
            metrics = return_metrics(pr)
            row = {
                "portfolio_name": portfolio_name,
                "germany_config": germany_config or "",
                "weight_scheme": scheme,
                **{f"weight_{book}": weights.get(book, 0.0) for book in ["france", "sweden", "netherlands", "germany"]},
                **metrics,
            }
            perf_rows.append(row)
            contrib = drawdown_contribution(returns, weights, metrics["dd_start"], metrics["dd_end"])
            for book, value in contrib.items():
                dd_rows.append(
                    {
                        "portfolio_name": portfolio_name,
                        "germany_config": germany_config or "",
                        "weight_scheme": scheme,
                        "book": book,
                        "weighted_return_during_max_dd": value,
                        "weight": weights.get(book, 0.0),
                        "max_drawdown": metrics["max_drawdown"],
                        "dd_start": metrics["dd_start"],
                        "dd_end": metrics["dd_end"],
                    }
                )
            monthly_rows.append(
                pd.DataFrame(
                    {
                        "trade_month": pr.index,
                        "portfolio_name": portfolio_name,
                        "germany_config": germany_config or "",
                        "weight_scheme": scheme,
                        "portfolio_month_return": pr.values,
                    }
                )
            )

    perf = pd.DataFrame(perf_rows)
    if not perf.empty:
        for scheme in perf["weight_scheme"].dropna().unique():
            core = perf[(perf["portfolio_name"] == "core_clean") & (perf["weight_scheme"] == scheme)]
            if core.empty:
                continue
            core_row = core.iloc[0]
            mask = perf["weight_scheme"].eq(scheme)
            perf.loc[mask, "delta_sharpe_vs_core"] = pd.to_numeric(perf.loc[mask, "sharpe"], errors="coerce") - safe_float(
                core_row.get("sharpe")
            )
            perf.loc[mask, "delta_return_vs_core"] = pd.to_numeric(
                perf.loc[mask, "total_return"], errors="coerce"
            ) - safe_float(core_row.get("total_return"))
            perf.loc[mask, "delta_abs_dd_improvement_vs_core"] = abs(safe_float(core_row.get("max_drawdown"))) - pd.to_numeric(
                perf.loc[mask, "max_drawdown"], errors="coerce"
            ).abs()
    return (
        perf,
        pd.DataFrame(dd_rows),
        pd.concat(monthly_rows, ignore_index=True, sort=False) if monthly_rows else pd.DataFrame(),
    )


def extract_row(df: pd.DataFrame, *, config: str, period: str | None = None) -> pd.Series | None:
    if df.empty or "config_name" not in df.columns:
        return None
    mask = df["config_name"].astype(str).eq(config)
    if period is not None and "period_name" in df.columns:
        mask &= df["period_name"].astype(str).eq(period)
    sub = df[mask]
    return None if sub.empty else sub.iloc[0]


def portfolio_row(portfolio_impact: pd.DataFrame, portfolio_name: str, scheme: str = "inverse_vol") -> pd.Series | None:
    if portfolio_impact.empty:
        return None
    sub = portfolio_impact[
        portfolio_impact["portfolio_name"].astype(str).eq(portfolio_name)
        & portfolio_impact["weight_scheme"].astype(str).eq(scheme)
    ]
    return None if sub.empty else sub.iloc[0]


def build_core_decision(
    *,
    oos_details: pd.DataFrame,
    oos_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    portfolio_impact: pd.DataFrame,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    details: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for cfg in [GERMANY_SIMPLE, GERMANY_MITIGATION]:
        summary = extract_row(oos_summary, config=cfg)
        final = extract_row(oos_details, config=cfg, period="holdout_2024_2025")
        y2025 = extract_row(oos_details, config=cfg, period="wf_test_2025")
        stress = extract_row(stress_summary, config=cfg)
        port_name = "core_plus_germany_simple" if cfg == GERMANY_SIMPLE else "core_plus_germany_mitigation"
        port = portfolio_row(portfolio_impact, port_name)
        row = {
            "config_name": cfg,
            "oos_windows": safe_float(summary.get("nb_windows")) if summary is not None else np.nan,
            "oos_sharpe_wins": safe_float(summary.get("windows_sharpe_improved")) if summary is not None else np.nan,
            "oos_return_wins": safe_float(summary.get("windows_return_improved")) if summary is not None else np.nan,
            "oos_dd_wins": safe_float(summary.get("windows_dd_improved")) if summary is not None else np.nan,
            "mean_oos_sharpe": safe_float(summary.get("mean_sharpe")) if summary is not None else np.nan,
            "min_oos_sharpe": safe_float(summary.get("min_sharpe")) if summary is not None else np.nan,
            "final_holdout_delta_sharpe": safe_float(final.get("delta_engine_sharpe")) if final is not None else np.nan,
            "final_holdout_delta_return": safe_float(final.get("delta_total_return_engine")) if final is not None else np.nan,
            "wf_2025_delta_sharpe": safe_float(y2025.get("delta_engine_sharpe")) if y2025 is not None else np.nan,
            "wf_2025_delta_return": safe_float(y2025.get("delta_total_return_engine")) if y2025 is not None else np.nan,
            "stress_trending_total_pnl": safe_float(stress.get("total_pnl")) if stress is not None else np.nan,
            "stress_trending_avg_pnl": safe_float(stress.get("avg_pnl_per_trade")) if stress is not None else np.nan,
            "portfolio_delta_sharpe_vs_core": safe_float(port.get("delta_sharpe_vs_core")) if port is not None else np.nan,
            "portfolio_delta_return_vs_core": safe_float(port.get("delta_return_vs_core")) if port is not None else np.nan,
            "portfolio_delta_abs_dd_improvement_vs_core": safe_float(port.get("delta_abs_dd_improvement_vs_core"))
            if port is not None
            else np.nan,
        }
        row["passes_final_holdout"] = row["final_holdout_delta_sharpe"] > 0 and row["final_holdout_delta_return"] > 0
        row["passes_2025"] = row["wf_2025_delta_sharpe"] > 0 and row["wf_2025_delta_return"] > 0
        row["passes_oos_majority"] = row["oos_sharpe_wins"] >= 5 and row["oos_return_wins"] >= 5
        row["stress_trending_non_destructive"] = row["stress_trending_total_pnl"] >= 0
        row["portfolio_improves_core"] = row["portfolio_delta_sharpe_vs_core"] > 0 and row["portfolio_delta_return_vs_core"] > 0
        row["portfolio_dd_not_materially_worse"] = row["portfolio_delta_abs_dd_improvement_vs_core"] >= -0.03
        rows.append(row)

    decision_df = pd.DataFrame(rows)
    simple = decision_df[decision_df["config_name"].eq(GERMANY_SIMPLE)].iloc[0]
    mitigation = decision_df[decision_df["config_name"].eq(GERMANY_MITIGATION)].iloc[0]
    mitigation_better_than_simple = (
        safe_float(mitigation.get("mean_oos_sharpe")) > safe_float(simple.get("mean_oos_sharpe"))
        and safe_float(mitigation.get("stress_trending_total_pnl")) > safe_float(simple.get("stress_trending_total_pnl"))
        and safe_float(mitigation.get("portfolio_delta_sharpe_vs_core")) >= safe_float(simple.get("portfolio_delta_sharpe_vs_core"))
    )
    simple_ready = all(
        bool(simple.get(key))
        for key in [
            "passes_final_holdout",
            "passes_2025",
            "passes_oos_majority",
            "stress_trending_non_destructive",
            "portfolio_improves_core",
            "portfolio_dd_not_materially_worse",
        ]
    )
    mitigation_ready = all(
        bool(mitigation.get(key))
        for key in [
            "passes_final_holdout",
            "passes_2025",
            "passes_oos_majority",
            "stress_trending_non_destructive",
            "portfolio_improves_core",
            "portfolio_dd_not_materially_worse",
        ]
    ) and mitigation_better_than_simple

    if mitigation_ready:
        verdict = "enter_core_with_mitigated_version"
        selected = GERMANY_MITIGATION
    elif simple_ready:
        verdict = "enter_core_with_simple_version"
        selected = GERMANY_SIMPLE
    elif bool(simple.get("passes_final_holdout")) or bool(mitigation.get("passes_final_holdout")):
        verdict = "remain_needs_validation"
        selected = ""
    else:
        verdict = "do_not_enter_core"
        selected = ""

    details.update(
        {
            "verdict": verdict,
            "selected_config": selected,
            "mitigation_better_than_simple": bool(mitigation_better_than_simple),
            "decision_rules": {
                "simple_ready": bool(simple_ready),
                "mitigation_ready": bool(mitigation_ready),
                "oos_majority_threshold": ">=5 Sharpe wins and >=5 return wins across fixed OOS windows",
                "portfolio_dd_tolerance": "max drawdown may not worsen by more than 3 percentage points versus core_clean inverse_vol",
            },
        }
    )
    decision_df["final_verdict"] = verdict
    decision_df["selected_config"] = selected
    return decision_df, verdict, details


def write_text_outputs(out_dir: Path, verdict: str, details: dict[str, Any], decision_df: pd.DataFrame) -> None:
    lines = [
        "Germany core-entry final decision",
        "",
        f"Verdict: {verdict}",
        f"Selected config: {details.get('selected_config') or 'none'}",
        "",
        "Decision details:",
        json.dumps(details, indent=2, default=str),
        "",
        "Methodological note:",
        "Only three fixed Germany versions were compared: reference, corr<=0.75, and scan-time stress_trending bypass.",
        "No retuning, no new filter family, no core engine modification.",
    ]
    (out_dir / "promotion_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = ["Germany core-entry conclusion", "", f"Final decision: {verdict}", ""]
    if not decision_df.empty:
        for _, row in decision_df.iterrows():
            summary_lines.append(
                f"- {row['config_name']}: OOS Sharpe wins={row['oos_sharpe_wins']}/{row['oos_windows']}, "
                f"2025 delta Sharpe={row['wf_2025_delta_sharpe']:.3f}, "
                f"stress_trending PnL={row['stress_trending_total_pnl']:.3f}, "
                f"portfolio delta Sharpe={row['portfolio_delta_sharpe_vs_core']:.3f}"
            )
    summary_lines.extend(
        [
            "",
            "Read germany_oos_final_validation.csv, germany_stress_trending_comparison.csv and germany_portfolio_impact.csv for the numeric basis.",
        ]
    )
    (out_dir / "conclusion.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def run_germany_core_entry(options: GermanyCoreEntryOptions) -> Path:
    start = "2024-01-01" if options.smoke else options.start
    end = "2024-06-30" if options.smoke else options.end
    out_dir = build_output_dir(options, start, end)
    specs = final_specs()

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
        "versions": version_manifest(specs).to_dict(orient="records"),
        "portfolio_core": ["france", "sweden", "netherlands"],
        "methodology": "Final fixed-version Germany core-entry test. No retuning; mitigation is scan-time stress_trending bypass only.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    version_manifest(specs).to_csv(out_dir / "germany_final_versions.csv", index=False)

    LOGGER.info("Running full-period Germany versions")
    full_period = PeriodSpec("full_period", "2018_2025" if not options.smoke else "smoke", start, end, "full")
    full_frames = run_phase3_specs_for_period(
        period=full_period,
        specs=specs,
        reference=reference,
        scans=scans,
        thresholds=thresholds,
        market_features=market_features,
        price_panel=price_panel,
        asset_metadata=asset_metadata,
    )
    full_run_level = build_run_level(full_frames, "full_period_final")
    stress_tables = stress_trending_tables(full_frames.get("trades_enriched", pd.DataFrame()))

    LOGGER.info("Running fixed OOS windows")
    oos_frames = []
    for period in oos_periods(start, end, smoke=options.smoke):
        oos_frames.append(
            run_phase3_specs_for_period(
                period=period,
                specs=specs,
                reference=reference,
                scans=scans,
                thresholds=thresholds,
                market_features=market_features,
                price_panel=price_panel,
                asset_metadata=asset_metadata,
            )
        )
    oos_combined = concat_frames(oos_frames)
    oos_details = add_final_oos_flags(build_run_level(oos_combined, "final_oos"))
    oos_summary = build_oos_summary(oos_details)

    LOGGER.info("Building portfolio impact versus France+Sweden+Netherlands core")
    core_monthly = load_core_monthly(options, out_dir, start, end)
    germany_monthly = full_frames.get("monthly_returns", pd.DataFrame()).copy()
    portfolio_impact, dd_contribution, core_vs_monthly = portfolio_impact_outputs(
        core_monthly=core_monthly,
        germany_monthly=germany_monthly,
    )

    decision_df, verdict, details = build_core_decision(
        oos_details=oos_details,
        oos_summary=oos_summary,
        stress_summary=stress_tables["stress_trending_summary"],
        portfolio_impact=portfolio_impact,
    )

    run_level = pd.concat([full_run_level, oos_details], ignore_index=True, sort=False)
    run_level.to_csv(out_dir / "run_level.csv", index=False)
    oos_details.to_csv(out_dir / "germany_oos_final_validation.csv", index=False)
    oos_summary.to_csv(out_dir / "germany_oos_summary.csv", index=False)
    stress_tables["stress_trending_summary"].to_csv(out_dir / "germany_stress_trending_comparison.csv", index=False)
    stress_tables["stress_trending_corr_bucket"].to_csv(out_dir / "germany_stress_trending_corr_bucket.csv", index=False)
    stress_tables["stress_trending_pair_contribution"].to_csv(out_dir / "germany_stress_trending_pair_contribution.csv", index=False)
    stress_tables["stress_trending_exit_breakdown"].to_csv(out_dir / "germany_stress_trending_exit_breakdown.csv", index=False)
    full_frames.get("exit_behavior", pd.DataFrame()).to_csv(out_dir / "germany_exit_breakdown_final.csv", index=False)
    portfolio_impact.to_csv(out_dir / "germany_portfolio_impact.csv", index=False)
    dd_contribution.to_csv(out_dir / "germany_drawdown_contribution.csv", index=False)
    decision_df.to_csv(out_dir / "germany_core_decision.csv", index=False)
    germany_monthly.to_csv(out_dir / "germany_monthly_returns_final.csv", index=False)
    core_vs_monthly.to_csv(out_dir / "core_vs_core_plus_germany_monthly_returns.csv", index=False)
    full_frames.get("filter_diagnostics", pd.DataFrame()).to_csv(out_dir / "germany_filter_diagnostics_final.csv", index=False)

    details["output_dir"] = str(out_dir)
    (out_dir / "decision_details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    write_text_outputs(out_dir, verdict, details, decision_df)
    LOGGER.info("Germany core-entry campaign complete: %s", out_dir)
    return out_dir
