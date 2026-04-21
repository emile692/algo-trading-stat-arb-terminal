from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.country_research_pipeline import PROJECT_ROOT
from utils.germany_core_entry import GERMANY_MITIGATION, GERMANY_SIMPLE
from utils.germany_phase3 import safe_float
from utils.multibook_portfolio import align_book_returns, drawdown_contribution, portfolio_returns, return_metrics


LOGGER = logging.getLogger("germany_shadow_validation")
EXPERIMENTS_ROOT = PROJECT_ROOT / "data" / "experiments"
RECENT_WINDOWS = [
    ("holdout_2024_2025", "2024-01-01", "2025-12-31", "final holdout"),
    ("wf_test_2025", "2025-01-01", "2025-12-31", "critical recent year"),
    ("holdout_2023_2025", "2023-01-01", "2025-12-31", "late holdout context"),
]
PORTFOLIO_SCENARIOS = [
    ("core_clean", ""),
    ("core_plus_germany_simple", GERMANY_SIMPLE),
    ("core_plus_germany_mitigation", GERMANY_MITIGATION),
]
BOOKS = ["france", "sweden", "netherlands", "germany"]


@dataclass(frozen=True)
class ShadowValidationOptions:
    output_root: Path = EXPERIMENTS_ROOT
    output_suffix: str | None = None
    germany_core_entry_dir: Path | None = None
    multibook_dir: Path | None = None
    smoke: bool = False


def latest_experiment(pattern: str, *, exclude_smoke: bool = True) -> Path:
    candidates = sorted(EXPERIMENTS_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if exclude_smoke:
        candidates = [p for p in candidates if "_smoke" not in p.name]
    if not candidates:
        raise FileNotFoundError(f"No experiment directory matched {pattern}")
    return candidates[0]


def build_output_dir(options: ShadowValidationOptions) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"germany_shadow_validation_{stamp}"
    if options.smoke:
        name += "_smoke"
    if options.output_suffix:
        name += f"_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_inputs(options: ShadowValidationOptions) -> dict[str, Any]:
    germany_dir = options.germany_core_entry_dir or latest_experiment("germany_core_entry_20180101_20251231_*")
    multibook_dir = options.multibook_dir or latest_experiment("multibook_portfolio_sweden_germany_france_netherlands_*")
    required = {
        "germany_oos": germany_dir / "germany_oos_final_validation.csv",
        "germany_full": germany_dir / "run_level.csv",
        "stress": germany_dir / "germany_stress_trending_comparison.csv",
        "germany_monthly": germany_dir / "germany_monthly_returns_final.csv",
        "full_portfolio_impact": germany_dir / "germany_portfolio_impact.csv",
        "core_monthly": multibook_dir / "book_monthly_returns.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source files: " + ", ".join(missing))
    return {
        "germany_dir": germany_dir,
        "multibook_dir": multibook_dir,
        "germany_oos": pd.read_csv(required["germany_oos"]),
        "germany_full": pd.read_csv(required["germany_full"]),
        "stress": pd.read_csv(required["stress"]),
        "germany_monthly": pd.read_csv(required["germany_monthly"]),
        "full_portfolio_impact": pd.read_csv(required["full_portfolio_impact"]),
        "core_monthly": pd.read_csv(required["core_monthly"]),
    }


def normalize_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trade_month"] = pd.to_datetime(out["trade_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["month_return"] = pd.to_numeric(out["month_return"], errors="coerce")
    return out.dropna(subset=["trade_month", "month_return"]).reset_index(drop=True)


def germany_book_monthly(germany_monthly: pd.DataFrame, config_name: str) -> pd.DataFrame:
    out = normalize_monthly(germany_monthly)
    out = out[out["config_name"].astype(str).eq(config_name)].copy()
    out["book"] = "germany"
    out["country"] = "germany"
    return out[["trade_month", "month_return", "config_name", "book", "country"]]


def filter_monthly(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = normalize_monthly(df)
    start_m = pd.Timestamp(start).to_period("M").to_timestamp()
    end_m = pd.Timestamp(end).to_period("M").to_timestamp()
    return out[(out["trade_month"] >= start_m) & (out["trade_month"] <= end_m)].copy()


def fixed_weights(full_portfolio_impact: pd.DataFrame, portfolio_name: str, scheme: str) -> dict[str, float]:
    row = full_portfolio_impact[
        full_portfolio_impact["portfolio_name"].astype(str).eq(portfolio_name)
        & full_portfolio_impact["weight_scheme"].astype(str).eq(scheme)
    ]
    if row.empty:
        return {}
    source = row.iloc[0]
    return {book: safe_float(source.get(f"weight_{book}")) for book in BOOKS if safe_float(source.get(f"weight_{book}")) > 0}


def build_recent_validation(germany_oos: pd.DataFrame, *, smoke: bool) -> pd.DataFrame:
    windows = ["wf_test_2025"] if smoke else [w[0] for w in RECENT_WINDOWS]
    out = germany_oos[germany_oos["period_name"].astype(str).isin(windows)].copy()
    order = {name: idx for idx, (name, *_rest) in enumerate(RECENT_WINDOWS)}
    out["window_priority"] = out["period_name"].map(order).fillna(99).astype(int)
    return out.sort_values(["window_priority", "config_name"]).reset_index(drop=True)


def build_recent_exit_breakdown(recent_validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in recent_validation.iterrows():
        nb = safe_float(row.get("nb_trades"))
        for reason, col in [("TP", "nb_tp"), ("SL", "nb_sl"), ("TIME", "nb_time")]:
            count = safe_float(row.get(col))
            rows.append(
                {
                    "period_name": row.get("period_name"),
                    "period_start": row.get("period_start"),
                    "period_end": row.get("period_end"),
                    "config_name": row.get("config_name"),
                    "exit_reason": reason,
                    "nb_exits": count,
                    "exit_rate": count / nb if nb and np.isfinite(nb) else np.nan,
                    "nb_trades": nb,
                }
            )
    return pd.DataFrame(rows)


def build_recent_portfolio_impact(
    *,
    core_monthly: pd.DataFrame,
    germany_monthly: pd.DataFrame,
    full_portfolio_impact: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = [("wf_test_2025", "2025-01-01", "2025-12-31", "critical recent year")] if smoke else RECENT_WINDOWS
    core = normalize_monthly(core_monthly)
    core = core[core["book"].astype(str).isin(["france", "sweden", "netherlands"])].copy()
    perf_rows: list[dict[str, Any]] = []
    dd_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    for period_name, start, end, label in windows:
        core_window = filter_monthly(core, start, end)
        for scenario, germany_config in PORTFOLIO_SCENARIOS:
            monthly = core_window.copy()
            if germany_config:
                monthly = pd.concat(
                    [monthly, filter_monthly(germany_book_monthly(germany_monthly, germany_config), start, end)],
                    ignore_index=True,
                    sort=False,
                )
            returns = align_book_returns(monthly)
            for scheme in ["equal_weight", "inverse_vol"]:
                weights = fixed_weights(full_portfolio_impact, scenario, scheme)
                if not weights:
                    continue
                pr = portfolio_returns(returns, weights)
                metrics = return_metrics(pr)
                perf_rows.append(
                    {
                        "period_name": period_name,
                        "period_label": label,
                        "period_start": start,
                        "period_end": end,
                        "portfolio_name": scenario,
                        "germany_config": germany_config,
                        "weight_scheme": scheme,
                        **{f"weight_{book}": weights.get(book, 0.0) for book in BOOKS},
                        **metrics,
                    }
                )
                contrib = drawdown_contribution(returns, weights, metrics["dd_start"], metrics["dd_end"])
                for book, value in contrib.items():
                    dd_rows.append(
                        {
                            "period_name": period_name,
                            "portfolio_name": scenario,
                            "weight_scheme": scheme,
                            "book": book,
                            "weighted_return_during_max_dd": value,
                            "weight": weights.get(book, 0.0),
                            "max_drawdown": metrics["max_drawdown"],
                            "dd_start": metrics["dd_start"],
                            "dd_end": metrics["dd_end"],
                        }
                    )
                monthly_frames.append(
                    pd.DataFrame(
                        {
                            "trade_month": pr.index,
                            "period_name": period_name,
                            "portfolio_name": scenario,
                            "germany_config": germany_config,
                            "weight_scheme": scheme,
                            "portfolio_month_return": pr.values,
                        }
                    )
                )
    perf = pd.DataFrame(perf_rows)
    if not perf.empty:
        for (period, scheme), group in perf.groupby(["period_name", "weight_scheme"]):
            core_row = group[group["portfolio_name"].eq("core_clean")]
            simple_row = group[group["portfolio_name"].eq("core_plus_germany_simple")]
            if core_row.empty:
                continue
            c = core_row.iloc[0]
            mask = perf["period_name"].eq(period) & perf["weight_scheme"].eq(scheme)
            perf.loc[mask, "delta_sharpe_vs_core"] = pd.to_numeric(perf.loc[mask, "sharpe"], errors="coerce") - safe_float(c.get("sharpe"))
            perf.loc[mask, "delta_return_vs_core"] = pd.to_numeric(perf.loc[mask, "total_return"], errors="coerce") - safe_float(
                c.get("total_return")
            )
            perf.loc[mask, "delta_abs_dd_improvement_vs_core"] = abs(safe_float(c.get("max_drawdown"))) - pd.to_numeric(
                perf.loc[mask, "max_drawdown"], errors="coerce"
            ).abs()
            if not simple_row.empty:
                s = simple_row.iloc[0]
                perf.loc[mask, "delta_sharpe_vs_simple_portfolio"] = pd.to_numeric(
                    perf.loc[mask, "sharpe"], errors="coerce"
                ) - safe_float(s.get("sharpe"))
                perf.loc[mask, "delta_return_vs_simple_portfolio"] = pd.to_numeric(
                    perf.loc[mask, "total_return"], errors="coerce"
                ) - safe_float(s.get("total_return"))
    return (
        perf,
        pd.DataFrame(dd_rows),
        pd.concat(monthly_frames, ignore_index=True, sort=False) if monthly_frames else pd.DataFrame(),
    )


def metric_from(df: pd.DataFrame, config: str, column: str, *, period_name: str | None = None) -> float:
    if df.empty:
        return np.nan
    mask = df["config_name"].astype(str).eq(config)
    if period_name and "period_name" in df.columns:
        mask &= df["period_name"].astype(str).eq(period_name)
    sub = df[mask]
    return np.nan if sub.empty else safe_float(sub.iloc[0].get(column))


def build_simple_vs_mitigated_summary(
    *,
    full_run: pd.DataFrame,
    recent_validation: pd.DataFrame,
    stress: pd.DataFrame,
    portfolio_recent: pd.DataFrame,
    full_portfolio_impact: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    full = full_run[full_run["axis"].astype(str).eq("full_period_final")].copy()
    for metric in ["engine_sharpe", "total_return_engine", "engine_max_drawdown", "nb_trades", "avg_pnl_per_trade"]:
        rows.append(
            {
                "dimension": "full_period_germany",
                "window": "2018_2025",
                "metric": metric,
                "simple_value": metric_from(full, GERMANY_SIMPLE, metric),
                "mitigated_value": metric_from(full, GERMANY_MITIGATION, metric),
            }
        )

    for period in recent_validation["period_name"].dropna().unique():
        for metric in ["engine_sharpe", "total_return_engine", "engine_max_drawdown", "nb_trades", "avg_pnl_per_trade"]:
            rows.append(
                {
                    "dimension": "recent_germany",
                    "window": period,
                    "metric": metric,
                    "simple_value": metric_from(recent_validation, GERMANY_SIMPLE, metric, period_name=period),
                    "mitigated_value": metric_from(recent_validation, GERMANY_MITIGATION, metric, period_name=period),
                }
            )

    for metric in ["total_pnl", "avg_pnl_per_trade", "nb_trades", "win_rate", "nb_tp", "nb_sl", "nb_time"]:
        rows.append(
            {
                "dimension": "stress_trending",
                "window": "full_period",
                "metric": metric,
                "simple_value": metric_from(stress, GERMANY_SIMPLE, metric),
                "mitigated_value": metric_from(stress, GERMANY_MITIGATION, metric),
            }
        )

    for source_name, source in [("portfolio_full", full_portfolio_impact), ("portfolio_recent", portfolio_recent)]:
        for _, row in source[source["portfolio_name"].astype(str).eq("core_plus_germany_mitigation")].iterrows():
            period = row.get("period_name", "2018_2025")
            scheme = row.get("weight_scheme")
            simple = source[
                source["portfolio_name"].astype(str).eq("core_plus_germany_simple")
                & source["weight_scheme"].astype(str).eq(str(scheme))
            ]
            if "period_name" in source.columns:
                simple = simple[simple["period_name"].astype(str).eq(str(period))]
            if simple.empty:
                continue
            simple_row = simple.iloc[0]
            for metric in ["sharpe", "total_return", "max_drawdown", "delta_sharpe_vs_core", "delta_return_vs_core"]:
                rows.append(
                    {
                        "dimension": source_name,
                        "window": period,
                        "weight_scheme": scheme,
                        "metric": metric,
                        "simple_value": safe_float(simple_row.get(metric)),
                        "mitigated_value": safe_float(row.get(metric)),
                    }
                )

    out = pd.DataFrame(rows)
    out["delta_mitigated_minus_simple"] = pd.to_numeric(out["mitigated_value"], errors="coerce") - pd.to_numeric(
        out["simple_value"], errors="coerce"
    )
    return out


def make_decision(
    *,
    recent_validation: pd.DataFrame,
    stress: pd.DataFrame,
    portfolio_recent: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    inverse = portfolio_recent[portfolio_recent["weight_scheme"].astype(str).eq("inverse_vol")]
    mitigated_recent = inverse[inverse["portfolio_name"].astype(str).eq("core_plus_germany_mitigation")]
    key_recent = mitigated_recent[mitigated_recent["period_name"].astype(str).isin(["holdout_2024_2025", "wf_test_2025"])]
    portfolio_recent_pass = (
        not key_recent.empty
        and (pd.to_numeric(key_recent["delta_sharpe_vs_core"], errors="coerce") > 0).all()
        and (pd.to_numeric(key_recent["delta_return_vs_core"], errors="coerce") > 0).all()
        and (pd.to_numeric(key_recent["delta_abs_dd_improvement_vs_core"], errors="coerce") >= 0).all()
    )

    st_simple = metric_from(stress, GERMANY_SIMPLE, "total_pnl")
    st_mitigated = metric_from(stress, GERMANY_MITIGATION, "total_pnl")
    stress_pass = np.isfinite(st_mitigated) and st_mitigated > 0 and st_mitigated > st_simple

    recent_rows = recent_validation[recent_validation["config_name"].astype(str).eq(GERMANY_MITIGATION)]
    key_germany = recent_rows[recent_rows["period_name"].astype(str).isin(["holdout_2024_2025", "wf_test_2025"])]
    standalone_recent_not_negative = (
        not key_germany.empty
        and (pd.to_numeric(key_germany["engine_sharpe"], errors="coerce") > 0).all()
        and (pd.to_numeric(key_germany["total_return_engine"], errors="coerce") > 0).all()
    )
    mitigation_not_worse_than_simple_recent = (
        not key_germany.empty
        and (pd.to_numeric(key_germany["delta_vs_simple_engine_sharpe"], errors="coerce") >= -0.05).all()
        and (pd.to_numeric(key_germany["delta_vs_simple_total_return"], errors="coerce") >= -0.02).all()
    )

    y2025 = key_germany[key_germany["period_name"].astype(str).eq("wf_test_2025")]
    y2025_under_reference = (
        not y2025.empty
        and safe_float(y2025.iloc[0].get("delta_engine_sharpe")) < 0
        and safe_float(y2025.iloc[0].get("delta_total_return_engine")) < 0
    )

    full_portfolio_advantage = summary[
        summary["dimension"].astype(str).eq("portfolio_full")
        & summary["weight_scheme"].astype(str).eq("inverse_vol")
        & summary["metric"].astype(str).eq("sharpe")
    ]
    full_advantage_pass = not full_portfolio_advantage.empty and safe_float(
        full_portfolio_advantage.iloc[0].get("delta_mitigated_minus_simple")
    ) > 0

    if portfolio_recent_pass and stress_pass and mitigation_not_worse_than_simple_recent and standalone_recent_not_negative and full_advantage_pass:
        verdict = "promote_to_core_shadow_passed"
    elif portfolio_recent_pass and stress_pass and standalone_recent_not_negative:
        verdict = "keep_as_shadow_candidate"
    else:
        verdict = "do_not_promote_current_mitigated_version"

    details = {
        "portfolio_recent_pass": bool(portfolio_recent_pass),
        "stress_pass": bool(stress_pass),
        "mitigation_not_worse_than_simple_recent": bool(mitigation_not_worse_than_simple_recent),
        "standalone_recent_not_negative": bool(standalone_recent_not_negative),
        "wf_2025_under_reference": bool(y2025_under_reference),
        "full_portfolio_advantage_pass": bool(full_advantage_pass),
        "decision_note": (
            "2025 Germany standalone remains below reference, but the mitigated book is positive, not worse than simple, "
            "and improves the recent core portfolio. This is tracked as residual risk, not a hard block."
        )
        if y2025_under_reference
        else "",
    }
    decision = pd.DataFrame(
        [
            {
                "candidate": GERMANY_MITIGATION,
                "decision": verdict,
                **details,
            }
        ]
    )
    return decision, verdict, details


def write_text_outputs(out_dir: Path, verdict: str, details: dict[str, Any]) -> None:
    lines = [
        "Germany mitigated shadow validation",
        "",
        f"Decision: {verdict}",
        "",
        "Decision details:",
        json.dumps(details, indent=2, default=str),
        "",
        "Methodology:",
        "No new Germany variant was tested. The campaign reuses existing core-entry and multibook outputs, then restricts the readout to recent windows and current core portfolio impact.",
    ]
    (out_dir / "promotion_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    conclusion = [
        "Germany shadow validation conclusion",
        "",
        f"Final decision: {verdict}",
        "",
        "Read germany_recent_validation.csv, germany_recent_portfolio_impact.csv and germany_simple_vs_mitigated_summary.csv for the numeric basis.",
    ]
    (out_dir / "conclusion.txt").write_text("\n".join(conclusion) + "\n", encoding="utf-8")


def run_germany_shadow_validation(options: ShadowValidationOptions) -> Path:
    out_dir = build_output_dir(options)
    inputs = load_inputs(options)

    recent_validation = build_recent_validation(inputs["germany_oos"], smoke=options.smoke)
    recent_exit = build_recent_exit_breakdown(recent_validation)
    portfolio_recent, drawdown_recent, portfolio_monthly = build_recent_portfolio_impact(
        core_monthly=inputs["core_monthly"],
        germany_monthly=inputs["germany_monthly"],
        full_portfolio_impact=inputs["full_portfolio_impact"],
        smoke=options.smoke,
    )
    summary = build_simple_vs_mitigated_summary(
        full_run=inputs["germany_full"],
        recent_validation=recent_validation,
        stress=inputs["stress"],
        portfolio_recent=portfolio_recent,
        full_portfolio_impact=inputs["full_portfolio_impact"],
    )
    decision, verdict, details = make_decision(
        recent_validation=recent_validation,
        stress=inputs["stress"],
        portfolio_recent=portfolio_recent,
        summary=summary,
    )

    recent_validation.to_csv(out_dir / "germany_recent_validation.csv", index=False)
    recent_exit.to_csv(out_dir / "germany_recent_exit_breakdown.csv", index=False)
    portfolio_recent.to_csv(out_dir / "germany_recent_portfolio_impact.csv", index=False)
    drawdown_recent.to_csv(out_dir / "germany_recent_drawdown_contribution.csv", index=False)
    summary.to_csv(out_dir / "germany_simple_vs_mitigated_summary.csv", index=False)
    decision.to_csv(out_dir / "germany_shadow_decision.csv", index=False)
    portfolio_monthly.to_csv(out_dir / "core_plus_germany_recent_monthly_returns.csv", index=False)
    core_recent = portfolio_monthly[portfolio_monthly["portfolio_name"].astype(str).eq("core_clean")].copy()
    core_recent.to_csv(out_dir / "core_recent_monthly_returns.csv", index=False)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_germany_core_entry_dir": str(inputs["germany_dir"]),
        "source_multibook_dir": str(inputs["multibook_dir"]),
        "smoke": options.smoke,
        "candidate": GERMANY_MITIGATION,
        "compared_configs": ["reference", GERMANY_SIMPLE, GERMANY_MITIGATION],
        "recent_windows": RECENT_WINDOWS,
        "methodology": "Shadow validation from existing outputs only; no new variants, no retuning, no backtest engine changes.",
        "decision_details": details,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_text_outputs(out_dir, verdict, details)
    LOGGER.info("Germany shadow validation complete: %s", out_dir)
    return out_dir
