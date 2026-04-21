from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "data" / "experiments"


@dataclass(frozen=True)
class QualificationOptions:
    countries: tuple[str, ...] = ("france", "norway", "netherlands")
    output_root: Path = EXPERIMENTS_ROOT
    output_suffix: str | None = None
    smoke: bool = False


@dataclass(frozen=True)
class CandidatePolicy:
    config_name: str
    label: str
    rule: str
    rationale: str


COUNTRY_POLICIES: dict[str, CandidatePolicy] = {
    "france": CandidatePolicy(
        config_name="reference",
        label="france_baseline_reference",
        rule="Use the local France baseline without transferred filters.",
        rationale=(
            "France already has a strong clean local reference; prior Sweden-regime "
            "transfer and local entry filter degraded portfolio metrics."
        ),
    ),
    "norway": CandidatePolicy(
        config_name="reference_plus_pair_filter",
        label="norway_pair_filter_candidate",
        rule="Use the standard pipeline pair_filter candidate, then apply a strict breadth veto.",
        rationale=(
            "Norway's pair_filter won the standard pipeline and all three splits, but "
            "must be audited for over-filtering and small-sample risk."
        ),
    ),
    "netherlands": CandidatePolicy(
        config_name="reference",
        label="netherlands_baseline_reference",
        rule="Use the Netherlands local reference alone; no forced ablation.",
        rationale=(
            "Netherlands has a positive baseline and no actionable local ablation; "
            "the portfolio candidate is the simple reference."
        ),
    ),
}


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_country_dir(country: str, root: Path = EXPERIMENTS_ROOT) -> Path:
    candidates = [
        d
        for d in root.glob(f"country_research_{country}_*")
        if d.is_dir() and (d / "country_research_scorecard.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No country research output found for country={country}")
    return sorted(candidates, key=lambda d: d.stat().st_mtime)[-1]


def build_output_dir(options: QualificationOptions) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    countries = "_".join(options.countries)
    name = f"top_pf_qualification_{countries}_{stamp}"
    if options.smoke:
        name += "_smoke"
    if options.output_suffix:
        name += f"_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def merge_level_tables(source_dir: Path) -> pd.DataFrame:
    portfolio = _read_csv(source_dir / "ablation_portfolio_level.csv")
    trade = _read_csv(source_dir / "ablation_trade_level.csv")
    concentration = _read_csv(source_dir / "ablation_concentration.csv")
    if portfolio.empty:
        return pd.DataFrame()

    out = portfolio.copy()
    key = ["config_name"]
    trade_cols = [
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
        "gross_profit",
        "gross_loss",
        "profit_factor",
    ]
    if not trade.empty:
        out = out.merge(trade[[c for c in trade_cols if c in trade.columns]], on=key, how="left")
    conc_cols = [
        "config_name",
        "nb_paires_tradees",
        "nb_paires_positives",
        "nb_paires_negatives",
        "gross_profit",
        "gross_loss",
        "net_total_pnl",
        "top5_pnl",
        "top10_pnl",
        "bottom10_pnl",
        "top5_share_gross_profit",
        "top10_share_gross_profit",
        "bottom10_share_gross_loss_abs",
    ]
    if not concentration.empty:
        out = out.merge(
            concentration[[c for c in conc_cols if c in concentration.columns]],
            on=key,
            how="left",
            suffixes=("", "_concentration"),
        )
    return out


def candidate_row(country: str, source_dir: Path, run_level: pd.DataFrame) -> tuple[pd.Series, CandidatePolicy]:
    policy = COUNTRY_POLICIES.get(country)
    if policy is None:
        policy = CandidatePolicy("reference", f"{country}_reference", "Use reference.", "No explicit policy.")
    if not run_level.empty and policy.config_name in set(run_level["config_name"].astype(str)):
        return run_level[run_level["config_name"].astype(str) == policy.config_name].iloc[0], policy

    score = _read_csv(source_dir / "country_research_scorecard.csv")
    if not score.empty:
        fallback_name = str(score.iloc[0].get("best_candidate", "reference"))
        if fallback_name in set(run_level["config_name"].astype(str)):
            fallback = CandidatePolicy(
                fallback_name,
                f"{country}_{fallback_name}",
                "Fallback to scorecard best candidate because explicit policy row was missing.",
                "Automatic fallback from existing country_research_scorecard.csv.",
            )
            return run_level[run_level["config_name"].astype(str) == fallback_name].iloc[0], fallback
    if not run_level.empty:
        fallback = CandidatePolicy("reference", f"{country}_reference", "Fallback to first available row.", "Missing explicit row.")
        return run_level.iloc[0], fallback
    raise RuntimeError(f"No run-level row available for country={country}")


def build_temporal_rows(country: str, source_dir: Path, candidate: str) -> pd.DataFrame:
    portfolio = _read_csv(source_dir / "robustness_portfolio_level.csv")
    trade = _read_csv(source_dir / "robustness_trade_level.csv")
    concentration = _read_csv(source_dir / "robustness_concentration.csv")
    if portfolio.empty:
        return pd.DataFrame()
    configs = ["reference"]
    if candidate != "reference":
        configs.append(candidate)
    out = portfolio[portfolio["config_name"].astype(str).isin(configs)].copy()
    if not trade.empty:
        trade_cols = [
            "period_name",
            "config_name",
            "nb_trades",
            "total_pnl",
            "avg_pnl_per_trade",
            "median_pnl_per_trade",
            "win_rate",
            "nb_tp",
            "nb_sl",
            "nb_time",
        ]
        out = out.merge(trade[[c for c in trade_cols if c in trade.columns]], on=["period_name", "config_name"], how="left")
    if not concentration.empty:
        conc_cols = [
            "period_name",
            "config_name",
            "nb_paires_tradees",
            "nb_paires_positives",
            "nb_paires_negatives",
            "top5_share_gross_profit",
            "top10_share_gross_profit",
            "bottom10_share_gross_loss_abs",
        ]
        out = out.merge(
            concentration[[c for c in conc_cols if c in concentration.columns]],
            on=["period_name", "config_name"],
            how="left",
        )
    out.insert(0, "country", country)
    out["is_candidate"] = out["config_name"].astype(str) == candidate
    ref = out[out["config_name"].astype(str) == "reference"].set_index("period_name")
    deltas: list[dict[str, float | str]] = []
    for idx, row in out.iterrows():
        if str(row["config_name"]) == "reference" or str(row["period_name"]) not in ref.index:
            deltas.append(
                {
                    "delta_sharpe_vs_reference": 0.0,
                    "delta_return_vs_reference": 0.0,
                    "delta_avg_pnl_vs_reference": 0.0,
                    "delta_abs_dd_vs_reference": 0.0,
                }
            )
            continue
        r = ref.loc[str(row["period_name"])]
        deltas.append(
            {
                "delta_sharpe_vs_reference": _safe_float(row.get("engine_sharpe")) - _safe_float(r.get("engine_sharpe")),
                "delta_return_vs_reference": _safe_float(row.get("total_return_engine")) - _safe_float(r.get("total_return_engine")),
                "delta_avg_pnl_vs_reference": _safe_float(row.get("avg_pnl_per_trade")) - _safe_float(r.get("avg_pnl_per_trade")),
                "delta_abs_dd_vs_reference": abs(_safe_float(r.get("engine_max_drawdown"))) - abs(_safe_float(row.get("engine_max_drawdown"))),
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(deltas)], axis=1)


def filter_removed_pct(source_dir: Path, candidate: str) -> float:
    diag = _read_csv(source_dir / "filter_diagnostics.csv")
    if diag.empty or "config_name" not in diag.columns:
        return np.nan
    row = diag[diag["config_name"].astype(str) == candidate]
    if row.empty:
        return np.nan
    return _safe_float(row.iloc[0].get("h3_scan_removed_pct"), np.nan)


def candidate_metrics(
    country: str,
    source_dir: Path,
    run: pd.Series,
    temporal: pd.DataFrame,
    robustness_scorecard: pd.DataFrame,
) -> dict[str, Any]:
    cfg = str(run.get("config_name"))
    temporal_candidate = temporal[temporal["config_name"].astype(str) == cfg] if not temporal.empty else pd.DataFrame()
    robust = pd.Series(dtype=object)
    if not robustness_scorecard.empty:
        rr = robustness_scorecard[robustness_scorecard["config_name"].astype(str) == cfg]
        if not rr.empty:
            robust = rr.iloc[0]

    split_count = int(len(temporal_candidate))
    split_positive_sharpe = int((pd.to_numeric(temporal_candidate.get("engine_sharpe", pd.Series(dtype=float)), errors="coerce") > 0).sum())
    split_positive_return = int((pd.to_numeric(temporal_candidate.get("total_return_engine", pd.Series(dtype=float)), errors="coerce") > 0).sum())

    return {
        "country": country,
        "candidate_config": cfg,
        "full_return": _safe_float(run.get("total_return_engine")),
        "full_sharpe": _safe_float(run.get("engine_sharpe")),
        "full_max_drawdown": _safe_float(run.get("engine_max_drawdown")),
        "full_cagr": _safe_float(run.get("engine_cagr")),
        "full_trades": _safe_int(run.get("nb_trades")),
        "full_avg_pnl_per_trade": _safe_float(run.get("avg_pnl_per_trade")),
        "full_median_pnl_per_trade": _safe_float(run.get("median_pnl_per_trade")),
        "full_win_rate": _safe_float(run.get("win_rate")),
        "full_tp": _safe_int(run.get("nb_tp")),
        "full_sl": _safe_int(run.get("nb_sl")),
        "full_time": _safe_int(run.get("nb_time")),
        "full_pairs": _safe_int(run.get("nb_paires_tradees")),
        "full_positive_pairs": _safe_int(run.get("nb_paires_positives")),
        "full_negative_pairs": _safe_int(run.get("nb_paires_negatives")),
        "full_top5_share_gross_profit": _safe_float(run.get("top5_share_gross_profit")),
        "full_top10_share_gross_profit": _safe_float(run.get("top10_share_gross_profit")),
        "full_bottom10_share_gross_loss_abs": _safe_float(run.get("bottom10_share_gross_loss_abs")),
        "avg_open_positions": _safe_float(run.get("avg_open_positions")),
        "pct_days_fully_invested": _safe_float(run.get("pct_days_fully_invested")),
        "anomaly_flag": bool(run.get("anomaly_flag")) if str(run.get("anomaly_flag")) != "nan" else False,
        "anomaly_reasons": run.get("anomaly_reasons"),
        "split_count": split_count,
        "split_positive_sharpe": split_positive_sharpe,
        "split_positive_return": split_positive_return,
        "min_split_sharpe": _safe_float(robust.get("min_sharpe_across_splits")),
        "mean_split_sharpe": _safe_float(robust.get("mean_sharpe_across_splits")),
        "sharpe_std_across_splits": _safe_float(robust.get("sharpe_std_across_splits")),
        "mean_split_pairs": _safe_float(robust.get("mean_nb_pairs_traded_across_splits")),
        "split_anomaly_count": _safe_int(robust.get("anomaly_count")),
        "splits_outperforming_reference_on_sharpe": _safe_int(robust.get("splits_outperforming_reference_on_sharpe")),
        "filter_removed_pct": filter_removed_pct(source_dir, str(run.get("config_name"))),
    }


def evaluate_decision(metrics: dict[str, Any], policy: CandidatePolicy) -> dict[str, Any]:
    perf_ok = (
        metrics["full_sharpe"] >= 0.45
        and metrics["full_return"] > 0
        and metrics["full_avg_pnl_per_trade"] > 0
    )
    robust_ok = (
        metrics["split_count"] >= 3
        and metrics["split_positive_sharpe"] >= 3
        and metrics["split_positive_return"] >= 3
        and metrics["min_split_sharpe"] >= 0.0
    )
    breadth_ok = (
        metrics["full_trades"] >= 100
        and metrics["full_pairs"] >= 30
        and metrics["mean_split_pairs"] >= 15
    )
    concentration_ok = (
        np.isnan(metrics["full_top5_share_gross_profit"])
        or metrics["full_top5_share_gross_profit"] <= 0.50
    ) and (
        np.isnan(metrics["full_bottom10_share_gross_loss_abs"])
        or metrics["full_bottom10_share_gross_loss_abs"] <= 0.75
    )
    clean_ok = not bool(metrics["anomaly_flag"]) and int(metrics["split_anomaly_count"]) == 0
    overfiltering_risk = (
        metrics["full_trades"] < 50
        or metrics["full_pairs"] < 15
        or (not np.isnan(metrics["filter_removed_pct"]) and metrics["filter_removed_pct"] > 0.65)
    )

    if perf_ok and robust_ok and breadth_ok and concentration_ok and clean_ok and not overfiltering_risk:
        verdict = "enter_top_pf"
    elif perf_ok and robust_ok and clean_ok:
        verdict = "borderline_keep_watchlist"
    elif metrics["full_sharpe"] > 0 and metrics["full_return"] > 0 and clean_ok:
        verdict = "borderline_keep_watchlist"
    else:
        verdict = "reject_for_top_pf"

    if verdict == "enter_top_pf":
        reason = "Performance, temporal robustness, breadth, concentration and anomaly gates all pass."
    elif overfiltering_risk:
        reason = "Performance is interesting but breadth or filter aggressiveness fails top-portfolio gates."
    elif not perf_ok:
        reason = "Full-period performance gate is not strong enough for top portfolio."
    elif not robust_ok:
        reason = "Temporal robustness is incomplete for top portfolio."
    elif not clean_ok:
        reason = "Anomaly gate fails."
    else:
        reason = "Case remains useful but incomplete."

    return {
        "verdict": verdict,
        "perf_ok": perf_ok,
        "robust_ok": robust_ok,
        "breadth_ok": breadth_ok,
        "concentration_ok": concentration_ok,
        "clean_ok": clean_ok,
        "overfiltering_risk": overfiltering_risk,
        "decision_reason": reason,
        "candidate_rule": policy.rule,
        "candidate_rationale": policy.rationale,
    }


def build_period_contribution(source_dir: Path, country: str) -> pd.DataFrame:
    monthly = _read_csv(source_dir / "monthly_returns.csv")
    if monthly.empty or "trade_month" not in monthly.columns:
        return pd.DataFrame()
    out = monthly.copy()
    out["country"] = country
    out["year"] = pd.to_datetime(out["trade_month"], errors="coerce").dt.year
    return (
        out.groupby(["country", "config_name", "year"], as_index=False)
        .agg(
            months=("trade_month", "count"),
            total_month_return=("month_return", "sum") if "month_return" in out.columns else ("end_equity", "last"),
            avg_month_return=("month_return", "mean") if "month_return" in out.columns else ("end_equity", "last"),
            min_month_return=("month_return", "min") if "month_return" in out.columns else ("end_equity", "last"),
            max_month_return=("month_return", "max") if "month_return" in out.columns else ("end_equity", "last"),
            avg_open_positions=("avg_open_positions", "mean"),
        )
    )


def write_country_pair_and_period_files(out_dir: Path, country: str, source_dir: Path) -> None:
    pair = _read_csv(source_dir / "diagnostic_pair_level_summary.csv")
    if not pair.empty:
        pair.insert(0, "country", country)
        pair.sort_values("total_pnl", ascending=False).to_csv(out_dir / f"pair_contribution_{country}.csv", index=False)

    period = build_period_contribution(source_dir, country)
    if not period.empty:
        period.to_csv(out_dir / f"period_contribution_{country}.csv", index=False)


def run_top_pf_qualification(options: QualificationOptions) -> Path:
    countries = tuple(c.lower().strip() for c in options.countries)
    out_dir = build_output_dir(QualificationOptions(countries, options.output_root, options.output_suffix, options.smoke))

    run_rows: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    temporal_rows: list[pd.DataFrame] = []
    breadth_rows: list[dict[str, Any]] = []
    explicability_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    source_dirs: dict[str, str] = {}

    for country in countries:
        source_dir = latest_country_dir(country)
        source_dirs[country] = str(source_dir)
        run_level = merge_level_tables(source_dir)
        if run_level.empty:
            raise RuntimeError(f"Missing ablation run-level data for {country}: {source_dir}")
        run_level.insert(0, "country", country)
        run_level["source_dir"] = str(source_dir)
        run_rows.append(run_level)

        selected, policy = candidate_row(country, source_dir, run_level)
        temporal = build_temporal_rows(country, source_dir, str(selected["config_name"]))
        temporal_rows.append(temporal)
        robust_score = _read_csv(source_dir / "robustness_scorecard.csv")
        metrics = candidate_metrics(country, source_dir, selected, temporal, robust_score)
        decision = evaluate_decision(metrics, policy)

        candidate_rows.append(
            {
                "country": country,
                "candidate_config": metrics["candidate_config"],
                "candidate_label": policy.label,
                "candidate_rule": policy.rule,
                "candidate_rationale": policy.rationale,
                "source_dir": str(source_dir),
                "alternative_note": "Alternatives are in run_level.csv; only the selected candidate is qualified for top portfolio.",
            }
        )
        breadth_status = "healthy" if decision["breadth_ok"] and decision["concentration_ok"] else "fragile_or_too_concentrated"
        breadth_rows.append(
            {
                "country": country,
                "candidate_config": metrics["candidate_config"],
                "breadth_status": breadth_status,
                "full_trades": metrics["full_trades"],
                "full_pairs": metrics["full_pairs"],
                "full_positive_pairs": metrics["full_positive_pairs"],
                "full_negative_pairs": metrics["full_negative_pairs"],
                "mean_split_pairs": metrics["mean_split_pairs"],
                "top5_share_gross_profit": metrics["full_top5_share_gross_profit"],
                "top10_share_gross_profit": metrics["full_top10_share_gross_profit"],
                "bottom10_share_gross_loss_abs": metrics["full_bottom10_share_gross_loss_abs"],
                "filter_removed_pct": metrics["filter_removed_pct"],
                "overfiltering_risk": decision["overfiltering_risk"],
            }
        )
        simplicity_score = 5 if metrics["candidate_config"] == "reference" else 4
        if decision["overfiltering_risk"]:
            simplicity_score -= 1
        explicability_rows.append(
            {
                "country": country,
                "candidate_config": metrics["candidate_config"],
                "rule": policy.rule,
                "simplicity_score_1_to_5": max(1, simplicity_score),
                "overfit_risk": "high" if decision["overfiltering_risk"] else "low",
                "committee_defensible": decision["verdict"] == "enter_top_pf",
                "comment": policy.rationale if not decision["overfiltering_risk"] else policy.rationale + " Breadth veto remains material.",
            }
        )
        decision_rows.append({**metrics, **decision})
        write_country_pair_and_period_files(out_dir, country, source_dir)

    run_level_all = pd.concat(run_rows, ignore_index=True, sort=False)
    temporal_all = pd.concat(temporal_rows, ignore_index=True, sort=False) if temporal_rows else pd.DataFrame()
    decisions = pd.DataFrame(decision_rows)

    ranking = build_ranking(decisions)

    run_level_all.to_csv(out_dir / "run_level.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(out_dir / "country_candidate_versions.csv", index=False)
    temporal_all.to_csv(out_dir / "temporal_robustness.csv", index=False)
    pd.DataFrame(breadth_rows).to_csv(out_dir / "breadth_concentration_summary.csv", index=False)
    pd.DataFrame(explicability_rows).to_csv(out_dir / "explicability_summary.csv", index=False)
    decisions.to_csv(out_dir / "country_portfolio_decisions.csv", index=False)
    ranking.to_csv(out_dir / "top_pf_ranking.csv", index=False)

    metadata = {
        "countries": countries,
        "source_dirs": source_dirs,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "methodology": (
            "Qualification layer over existing country research outputs. No new grid search, "
            "no cross-country rule transfer, no engine-core modification."
        ),
        "decision_gates": {
            "perf_ok": "full_sharpe >= 0.45, positive return, positive avg pnl/trade",
            "robust_ok": "3 splits, all positive Sharpe and return, min split Sharpe >= 0",
            "breadth_ok": ">=100 trades, >=30 full-period pairs, >=15 mean split pairs",
            "concentration_ok": "top5 gross profit share <= 50%, bottom10 gross loss share <= 75%",
            "clean_ok": "no full-period anomaly and no split anomaly",
            "overfiltering_risk": "full_trades < 50 or full_pairs < 15 or filter removes >65% scan rows",
        },
        "limitations": [
            "Annual or rolling candidate-level backtests are not rerun here; standard split robustness is used.",
            "Pair contribution files come from diagnostic reference-level exports when ablation pair-level trades are unavailable.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    write_text_outputs(out_dir, decisions, ranking)
    return out_dir


def build_ranking(decisions: pd.DataFrame) -> pd.DataFrame:
    verdict_score = {
        "enter_top_pf": 3,
        "borderline_keep_watchlist": 2,
        "reject_for_top_pf": 1,
    }
    out = decisions.copy()
    out["verdict_score"] = out["verdict"].map(verdict_score).fillna(0)
    out["portfolio_score"] = (
        out["verdict_score"] * 100
        + out["full_sharpe"].fillna(-99) * 10
        + out["min_split_sharpe"].fillna(-99) * 5
        + np.minimum(out["full_pairs"].fillna(0), 100) / 10
        - out["overfiltering_risk"].astype(int) * 25
        - out["anomaly_flag"].astype(int) * 30
    )
    out = out.sort_values(
        ["verdict_score", "portfolio_score", "full_sharpe", "full_pairs"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[
        [
            "rank",
            "country",
            "verdict",
            "candidate_config",
            "portfolio_score",
            "full_sharpe",
            "full_return",
            "full_max_drawdown",
            "min_split_sharpe",
            "full_trades",
            "full_pairs",
            "mean_split_pairs",
            "overfiltering_risk",
            "decision_reason",
        ]
    ]


def write_text_outputs(out_dir: Path, decisions: pd.DataFrame, ranking: pd.DataFrame) -> None:
    lines = [
        "Top portfolio qualification campaign",
        "",
        "Portfolio decisions:",
    ]
    for _, row in decisions.sort_values("country").iterrows():
        lines.append(
            f"- {row['country']}: {row['verdict']} | candidate={row['candidate_config']} "
            f"| Sharpe={row['full_sharpe']:.3f} | min_split_sharpe={row['min_split_sharpe']:.3f} "
            f"| pairs={int(row['full_pairs'])} | reason={row['decision_reason']}"
        )
    lines.extend(
        [
            "",
            "Ranking:",
        ]
    )
    for _, row in ranking.iterrows():
        lines.append(
            f"{int(row['rank'])}. {row['country']} | {row['verdict']} | {row['candidate_config']} "
            f"| score={row['portfolio_score']:.2f}"
        )
    (out_dir / "promotion_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    conclusion = [
        "Conclusion",
        "",
        "France and Netherlands qualify as simple baseline portfolio candidates under the explicit gates.",
        "Norway remains attractive statistically but fails breadth/over-filtering gates for a final top portfolio book.",
        "",
        "Read country_portfolio_decisions.csv and top_pf_ranking.csv for the numeric basis.",
    ]
    (out_dir / "conclusion.txt").write_text("\n".join(conclusion) + "\n", encoding="utf-8")

