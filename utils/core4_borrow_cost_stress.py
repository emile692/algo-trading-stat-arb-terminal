from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.core4_audit_common import (
    BORROW_REQUIRED_SCENARIOS,
    DEFAULT_AUDIT_OUTPUT_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DAILY_CACHE_DIR,
    Core4AuditBaseOptions,
    load_core4_audit_context,
    parse_date_series,
    resolve_path,
)
from utils.core4_position_reconstruction import Core4PositionReconstructionOptions, reconstruct_core4_positions
from utils.core4_validation_pack import compute_performance_metrics


@dataclass(frozen=True)
class Core4BorrowCostStressOptions(Core4AuditBaseOptions):
    output_root: Path = DEFAULT_AUDIT_OUTPUT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    daily_cache_dir: Path = DEFAULT_DAILY_CACHE_DIR


def run_core4_borrow_cost_stress(
    options: Core4BorrowCostStressOptions,
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bundle = reconstruct_core4_borrow_cost_stress(options, project_root=project_root)
    output_dir = resolve_path(project_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle["borrow_cost_stress"].to_csv(output_dir / "borrow_cost_stress.csv", index=False)
    (output_dir / "borrow_cost_stress_summary.md").write_text(bundle["borrow_cost_stress_summary"], encoding="utf-8")
    return bundle


def reconstruct_core4_borrow_cost_stress(
    options: Core4BorrowCostStressOptions,
    *,
    project_root: Path,
) -> dict[str, Any]:
    context = load_core4_audit_context(options, project_root=project_root)
    position_bundle = reconstruct_core4_positions(
        Core4PositionReconstructionOptions(
            output_root=options.output_root,
            config_path=options.config_path,
            daily_cache_dir=options.daily_cache_dir,
            start=options.start,
            end=options.end,
            rebuild_daily_cache=options.rebuild_daily_cache,
            smoke=options.smoke,
            allow_near_match=options.allow_near_match,
        ),
        project_root=project_root,
    )
    short_exposure = _extract_short_exposure_series(
        daily_portfolio_exposures=position_bundle["daily_portfolio_exposures"],
        returns=context["portfolio_returns"],
    )
    scenario_df = _compute_borrow_cost_table(
        portfolio_returns=context["portfolio_returns"],
        short_exposure=short_exposure,
    )
    summary = _build_borrow_summary(
        context=context,
        scenario_df=scenario_df,
        short_exposure=short_exposure,
    )
    return {
        "context": context,
        "position_bundle": position_bundle,
        "borrow_cost_stress": scenario_df,
        "borrow_cost_stress_summary": summary,
    }


def _extract_short_exposure_series(*, daily_portfolio_exposures: pd.DataFrame, returns: pd.Series) -> pd.Series:
    if daily_portfolio_exposures.empty:
        return pd.Series(0.0, index=returns.index, name="short_exposure")
    frame = daily_portfolio_exposures.copy()
    frame["date"] = parse_date_series(frame["date"])
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    series = pd.to_numeric(frame["short_notional"], errors="coerce").fillna(0.0)
    series.index = frame["date"]
    series = series.reindex(returns.index).fillna(method="ffill").fillna(0.0)
    series.name = "short_exposure"
    return series


def _compute_borrow_cost_table(*, portfolio_returns: pd.Series, short_exposure: pd.Series) -> pd.DataFrame:
    rows = []
    base_metrics = compute_performance_metrics(portfolio_returns)
    base_total_return = float(base_metrics.get("cumulative_return", np.nan))
    short_exposure_years = float(short_exposure.sum() / 252.0)
    break_even_bps = (
        float(base_total_return / short_exposure_years * 10000.0)
        if np.isfinite(base_total_return) and np.isfinite(short_exposure_years) and short_exposure_years > 0.0
        else np.nan
    )

    scenario_to_bps = {
        "borrow_0bps": 0.0,
        "borrow_50bps": 50.0,
        "borrow_100bps": 100.0,
        "borrow_200bps": 200.0,
        "borrow_500bps": 500.0,
    }
    for scenario_name in BORROW_REQUIRED_SCENARIOS:
        bps = scenario_to_bps[scenario_name]
        daily_cost = short_exposure * (bps / 10000.0) / 252.0
        stressed_returns = portfolio_returns - daily_cost
        metrics = compute_performance_metrics(stressed_returns)
        total_cost = float(daily_cost.sum())
        gross_pnl_proxy = base_total_return
        rows.append(
            {
                "scenario": scenario_name,
                "borrow_bps_annualized": bps,
                "annualized_return_after_borrow": metrics.get("ann_return"),
                "annualized_volatility_after_borrow": metrics.get("ann_vol"),
                "sharpe_after_borrow": metrics.get("sharpe"),
                "max_drawdown_after_borrow": metrics.get("max_drawdown"),
                "cumulative_return_after_borrow": metrics.get("cumulative_return"),
                "total_estimated_borrow_cost": total_cost,
                "borrow_cost_as_pct_of_gross_pnl": (
                    float(total_cost / gross_pnl_proxy)
                    if np.isfinite(gross_pnl_proxy) and abs(gross_pnl_proxy) > 1e-12
                    else np.nan
                ),
                "turnover_proxy": np.nan,
                "delta_sharpe_vs_reference": _delta(metrics.get("sharpe"), base_metrics.get("sharpe")),
                "delta_max_drawdown_vs_reference": _delta(metrics.get("max_drawdown"), base_metrics.get("max_drawdown")),
                "break_even_borrow_bps_estimate": break_even_bps,
                "assumption": (
                    "Borrow cost is modeled as annualized bps applied to the reconstructed short exposure proxy "
                    "from daily portfolio exposures."
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_borrow_summary(
    *,
    context: dict[str, Any],
    scenario_df: pd.DataFrame,
    short_exposure: pd.Series,
) -> str:
    lines = [
        "# Core 4 Borrow Cost Stress Summary",
        "",
        f"- Analysis window: {context['analysis_start'].date()} to {context['analysis_end'].date()}",
        f"- Average short exposure proxy: {_fmt_number(short_exposure.mean())}",
        f"- Break-even borrow estimate: {_fmt_number(scenario_df['break_even_borrow_bps_estimate'].iloc[0])} bps annualized"
        if not scenario_df.empty
        else "- Break-even borrow estimate unavailable.",
        "",
        "## Scenario Results",
    ]
    for row in scenario_df.itertuples(index=False):
        lines.append(
            f"- {row.scenario}: ann_return={_fmt_pct(row.annualized_return_after_borrow)} | "
            f"Sharpe={_fmt_number(row.sharpe_after_borrow)} | maxDD={_fmt_pct(row.max_drawdown_after_borrow)} | "
            f"borrow_cost={_fmt_number(row.total_estimated_borrow_cost)} | cost_pct_gross_pnl={_fmt_pct(row.borrow_cost_as_pct_of_gross_pnl)}"
        )
    lines.extend(
        [
            "",
            "## Assumption",
            "- Borrow is charged on the short-exposure proxy exported by the Core 4 position reconstruction layer.",
            "- No security-level locate fee, hard-to-borrow dispersion, or country-specific borrow schedule is modeled here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _delta(value: Any, reference: Any) -> float:
    if value is None or reference is None or pd.isna(value) or pd.isna(reference):
        return np.nan
    return float(value) - float(reference)


def _fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{100.0 * float(value):.2f}%"
