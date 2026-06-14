from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.core4_audit_common import (
    DEFAULT_AUDIT_OUTPUT_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DAILY_CACHE_DIR,
    EXECUTION_DELAY_REQUIRED_SCENARIOS,
    Core4AuditBaseOptions,
    REFERENCE_ALLOCATOR_ID,
    load_core4_audit_context,
    parse_date_series,
    resolve_path,
)
from utils.core4_position_reconstruction import Core4PositionReconstructionOptions, reconstruct_core4_positions
from utils.core4_validation_pack import compute_performance_metrics


@dataclass(frozen=True)
class Core4ExecutionDelayStressOptions(Core4AuditBaseOptions):
    output_root: Path = DEFAULT_AUDIT_OUTPUT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    daily_cache_dir: Path = DEFAULT_DAILY_CACHE_DIR


def run_core4_execution_delay_stress(
    options: Core4ExecutionDelayStressOptions,
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bundle = reconstruct_core4_execution_delay_stress(options, project_root=project_root)
    output_dir = resolve_path(project_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle["execution_delay_stress"].to_csv(output_dir / "execution_delay_stress.csv", index=False)
    (output_dir / "execution_delay_stress_summary.md").write_text(bundle["execution_delay_stress_summary"], encoding="utf-8")
    return bundle


def reconstruct_core4_execution_delay_stress(
    options: Core4ExecutionDelayStressOptions,
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

    exposure_ratio = _build_book_activity_ratio(
        daily_book_exposures=position_bundle["daily_book_exposures"],
        frozen_weights=context["frozen_weights"],
        returns=context["country_returns"],
    )
    scenario_df = _compute_execution_delay_table(
        returns=context["country_returns"],
        frozen_weights=context["frozen_weights"],
        exposure_ratio=exposure_ratio,
    )
    summary = _build_execution_delay_summary(
        context=context,
        scenario_df=scenario_df,
    )
    return {
        "context": context,
        "position_bundle": position_bundle,
        "execution_delay_stress": scenario_df,
        "execution_delay_stress_summary": summary,
    }


def _build_book_activity_ratio(
    *,
    daily_book_exposures: pd.DataFrame,
    frozen_weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if daily_book_exposures.empty:
        return pd.DataFrame(index=returns.index, columns=returns.columns, data=0.0)

    exposure = daily_book_exposures.copy()
    exposure["date"] = parse_date_series(exposure["date"])
    wide = (
        exposure.pivot_table(index="date", columns="book", values="gross_exposure", aggfunc="sum")
        .reindex(index=returns.index, columns=returns.columns)
        .fillna(0.0)
    )
    weights = pd.to_numeric(frozen_weights.reindex(wide.columns), errors="coerce").replace(0.0, np.nan)
    ratio = wide.div(weights, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    return ratio


def _compute_execution_delay_table(
    *,
    returns: pd.DataFrame,
    frozen_weights: pd.Series,
    exposure_ratio: pd.DataFrame,
) -> pd.DataFrame:
    base_weights = pd.to_numeric(frozen_weights.reindex(returns.columns), errors="coerce").fillna(0.0)
    rows = []
    scenario_returns: dict[str, pd.Series] = {}
    scenario_turnover: dict[str, float] = {}

    scenario_specs = [
        ("delay_0d", exposure_ratio),
        ("delay_1d", exposure_ratio.shift(1).fillna(0.0)),
        ("delay_2d", exposure_ratio.shift(2).fillna(0.0)),
        ("delay_3d", exposure_ratio.shift(3).fillna(0.0)),
        ("entry_delay_only", np.minimum(exposure_ratio, exposure_ratio.shift(1).fillna(0.0))),
        ("exit_delay_only", np.maximum(exposure_ratio, exposure_ratio.shift(1).fillna(0.0))),
        ("entry_and_exit_delay", exposure_ratio.shift(1).fillna(0.0)),
    ]

    for scenario_name, scenario_ratio in scenario_specs:
        weighted = returns.mul(base_weights, axis=1).mul(scenario_ratio.reindex_like(returns).fillna(0.0), axis=0)
        port = weighted.sum(axis=1)
        port.name = scenario_name
        scenario_returns[scenario_name] = port
        scenario_turnover[scenario_name] = float(
            scenario_ratio.diff().abs().sum(axis=1).fillna(scenario_ratio.abs().sum(axis=1)).mean()
        )

    reference_metrics = compute_performance_metrics(scenario_returns["delay_0d"])
    for scenario_name in EXECUTION_DELAY_REQUIRED_SCENARIOS:
        port = scenario_returns[scenario_name]
        metrics = compute_performance_metrics(port)
        rows.append(
            {
                "scenario": scenario_name,
                "delay_days": float(_delay_days_for_scenario(scenario_name)),
                "implementation": _implementation_label(scenario_name),
                "allocator_id": REFERENCE_ALLOCATOR_ID,
                "annualized_return": metrics.get("ann_return"),
                "annualized_volatility": metrics.get("ann_vol"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "cumulative_return": metrics.get("cumulative_return"),
                "turnover_proxy": scenario_turnover[scenario_name],
                "delta_sharpe_vs_reference": _delta(metrics.get("sharpe"), reference_metrics.get("sharpe")),
                "delta_max_drawdown_vs_reference": _delta(metrics.get("max_drawdown"), reference_metrics.get("max_drawdown")),
                "assumption": (
                    "Timing stress is approximated by shifting reconstructed book activity ratios derived from "
                    "frozen weights and `n_open_positions` proxies, then applying those delayed activity ratios "
                    "to the realized frozen daily book returns."
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_execution_delay_summary(*, context: dict[str, Any], scenario_df: pd.DataFrame) -> str:
    lines = [
        "# Core 4 Execution Delay Stress Summary",
        "",
        f"- Analysis window: {context['analysis_start'].date()} to {context['analysis_end'].date()}",
        "- Stress type: delayed application of reconstructed book activity ratios on frozen daily returns.",
        "- This is an approximation layer, not a replay of true persisted signal states.",
        "",
        "## Scenario Results",
    ]
    for row in scenario_df.itertuples(index=False):
        lines.append(
            f"- {row.scenario}: ann_return={_fmt_pct(row.annualized_return)} | ann_vol={_fmt_pct(row.annualized_volatility)} | "
            f"Sharpe={_fmt_number(row.sharpe)} | maxDD={_fmt_pct(row.max_drawdown)} | "
            f"deltaSharpe={_fmt_number(row.delta_sharpe_vs_reference)} | deltaMaxDD={_fmt_pct(row.delta_max_drawdown_vs_reference)}"
        )
    return "\n".join(lines) + "\n"


def _delay_days_for_scenario(name: str) -> int:
    if name.startswith("delay_") and name.endswith("d"):
        return int(name.split("_")[1][:-1])
    if name in {"entry_delay_only", "exit_delay_only", "entry_and_exit_delay"}:
        return 1
    return 0


def _implementation_label(name: str) -> str:
    if name == "entry_delay_only":
        return "min(current_activity, lagged_activity)"
    if name == "exit_delay_only":
        return "max(current_activity, lagged_activity)"
    if name == "entry_and_exit_delay":
        return "lagged_activity_1d"
    return "lagged_activity_nd"


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
