from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.core_portfolio_daily import load_or_build_core_daily_returns
from utils.portfolio_allocation_research import (
    AllocationConfig,
    compute_portfolio_metrics,
    compute_portfolio_outputs,
    drawdown_from_equity,
    equity_from_returns,
    rolling_sharpe,
    walk_forward_backtest_allocations,
)


TRADING_DAYS = 252
DEFAULT_OUTPUT_DIR = Path("data/reports/core4_daily_reporting")
DEFAULT_DAILY_CACHE_DIR = Path("data/experiments/core_portfolio_reference_daily_cache")
DEFAULT_CONFIG_PATH = Path("config/core_portfolio_reference.json")
PRIMARY_ALLOCATOR_IDS = (
    "inverse_vol__lb126__weekly__floor_cap",
    "equal_weight__lb126__monthly__unconstrained",
)
OPTIONAL_ALLOCATOR_IDS = ("risk_parity__lb126__weekly__floor_cap",)
BOOK_COLORS = {
    "france": "#1f4e79",
    "sweden": "#c98f00",
    "netherlands": "#2d6a4f",
    "germany": "#8f2d56",
}
ALLOCATOR_COLORS = {
    "inverse_vol__lb126__weekly__floor_cap": "#0f4c5c",
    "equal_weight__lb126__monthly__unconstrained": "#8d6a00",
    "risk_parity__lb126__weekly__floor_cap": "#5c4d7d",
}
MONTH_NAME_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


@dataclass(frozen=True)
class Core4AllocatorSpec:
    config_id: str
    label: str
    role: str
    method: str
    lookback_days: int
    rebalance_frequency: str
    weight_floor: float
    weight_cap: float
    notes: str = ""


@dataclass(frozen=True)
class Core4ReportingOptions:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    config_path: Path = DEFAULT_CONFIG_PATH
    daily_cache_dir: Path = DEFAULT_DAILY_CACHE_DIR
    start: str | None = None
    end: str | None = None
    lag_days: int = 1
    include_optional_allocators: bool = True
    rebuild_daily_cache: bool = False
    smoke: bool = False


def default_allocator_specs(*, include_optional_allocators: bool = True) -> tuple[Core4AllocatorSpec, ...]:
    specs = [
        Core4AllocatorSpec(
            config_id="inverse_vol__lb126__weekly__floor_cap",
            label="Reference allocator",
            role="reference",
            method="inverse_vol",
            lookback_days=126,
            rebalance_frequency="weekly",
            weight_floor=0.10,
            weight_cap=0.40,
            notes="Primary production-style allocator for the frozen core 4 books.",
        ),
        Core4AllocatorSpec(
            config_id="equal_weight__lb126__monthly__unconstrained",
            label="Equal-weight benchmark",
            role="benchmark",
            method="equal_weight",
            lookback_days=126,
            rebalance_frequency="monthly",
            weight_floor=0.0,
            weight_cap=1.0,
            notes="Simple benchmark kept as the control portfolio.",
        ),
    ]
    if include_optional_allocators:
        specs.append(
            Core4AllocatorSpec(
                config_id="risk_parity__lb126__weekly__floor_cap",
                label="Risk parity monitor",
                role="comparison",
                method="risk_parity",
                lookback_days=126,
                rebalance_frequency="weekly",
                weight_floor=0.10,
                weight_cap=0.40,
                notes="Optional comparison allocator retained because it is close to the reference and simple to maintain.",
            )
        )
    return tuple(specs)


def run_core4_daily_reporting(options: Core4ReportingOptions, *, project_root: Path) -> Path:
    generated_at = datetime.now().replace(microsecond=0)
    output_dir = _resolve_output_dir(project_root, options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, book_daily_long, book_returns_wide = _load_core_inputs(options, project_root=project_root)
    specs = default_allocator_specs(include_optional_allocators=options.include_optional_allocators)
    allocation_configs = _allocation_configs_from_specs(specs)

    wf_outputs = walk_forward_backtest_allocations(book_returns_wide, allocation_configs, lag_days=options.lag_days)
    portfolio_outputs = compute_portfolio_outputs(
        book_returns_wide,
        wf_outputs["portfolio_daily_returns"],
        wf_outputs["weights_timeseries"],
        wf_outputs["turnover"],
        allocation_configs,
        compounding_modes=("compounded",),
    )

    bundle = _build_reporting_bundle(
        config=config,
        specs=specs,
        generated_at=generated_at,
        book_daily_long=book_daily_long,
        book_returns_wide=book_returns_wide,
        wf_outputs=wf_outputs,
        portfolio_outputs=portfolio_outputs,
    )
    _write_bundle(output_dir, bundle)
    _write_plots(output_dir, bundle)
    html = _render_html_report(bundle)
    (output_dir / "core4_daily_report.html").write_text(html, encoding="utf-8")
    metadata = _build_metadata(
        config=config,
        specs=specs,
        options=options,
        generated_at=generated_at,
        bundle=bundle,
        project_root=project_root,
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    conclusion = _build_conclusion(bundle)
    (output_dir / "conclusion.txt").write_text(conclusion, encoding="utf-8")
    return output_dir


def _load_core_inputs(options: Core4ReportingOptions, *, project_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config_path = _resolve_path(project_root, options.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Core portfolio config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))

    if options.smoke:
        config.setdefault("period", {})
        config["period"]["start"] = max(str(config["period"].get("start", "2018-01-01")), "2024-01-01")
        config["period"]["end"] = min(str(config["period"].get("end", "2025-12-31")), "2025-12-31")

    if options.start is not None:
        config.setdefault("period", {})
        config["period"]["start"] = str(options.start)
    if options.end is not None:
        config.setdefault("period", {})
        config["period"]["end"] = str(options.end)

    cache_dir = _resolve_path(project_root, options.daily_cache_dir)
    book_daily_long, book_returns_wide = load_or_build_core_daily_returns(
        config,
        root=project_root,
        cache_dir=cache_dir,
        rebuild=options.rebuild_daily_cache,
    )
    if book_returns_wide.empty:
        raise RuntimeError("No daily book returns available for core 4 reporting.")

    ordered_books = [str(book["book"]).strip().lower() for book in config.get("books", [])]
    available_books = [book for book in ordered_books if book in book_returns_wide.columns]
    if not available_books:
        raise RuntimeError(f"Configured core books not available in daily returns. Available={list(book_returns_wide.columns)}")

    book_returns_wide = book_returns_wide.loc[:, available_books].copy()
    book_returns_wide.index = pd.to_datetime(book_returns_wide.index, errors="coerce").normalize()
    book_returns_wide = book_returns_wide[~book_returns_wide.index.isna()].sort_index()
    book_returns_wide = book_returns_wide.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    book_daily_long = book_daily_long.copy()
    book_daily_long["trade_date"] = pd.to_datetime(book_daily_long["trade_date"], errors="coerce").dt.normalize()
    book_daily_long = book_daily_long.dropna(subset=["trade_date"]).sort_values(["trade_date", "book"]).reset_index(drop=True)
    book_daily_long = book_daily_long[book_daily_long["book"].isin(available_books)].copy()

    return config, book_daily_long, book_returns_wide


def _allocation_configs_from_specs(specs: tuple[Core4AllocatorSpec, ...]) -> list[AllocationConfig]:
    return [
        AllocationConfig(
            config_id=spec.config_id,
            method=spec.method,
            lookback_days=int(spec.lookback_days),
            rebalance_frequency=str(spec.rebalance_frequency),
            weight_floor=float(spec.weight_floor),
            weight_cap=float(spec.weight_cap),
            constraint_label="floor_cap" if (spec.weight_floor > 0.0 or spec.weight_cap < 1.0) else "unconstrained",
        )
        for spec in specs
    ]


def _build_reporting_bundle(
    *,
    config: dict[str, Any],
    specs: tuple[Core4AllocatorSpec, ...],
    generated_at: datetime,
    book_daily_long: pd.DataFrame,
    book_returns_wide: pd.DataFrame,
    wf_outputs: dict[str, pd.DataFrame],
    portfolio_outputs: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    spec_map = {spec.config_id: spec for spec in specs}
    allocator_order = [spec.config_id for spec in specs]

    book_daily_returns = _prepare_book_daily_returns(book_daily_long)
    portfolio_daily_returns = _prepare_portfolio_daily_returns(wf_outputs["portfolio_daily_returns"], spec_map)
    portfolio_equity = _prepare_portfolio_equity(portfolio_outputs["portfolio_equity_curves"], spec_map)
    portfolio_drawdown = portfolio_equity.loc[:, ["config_id", "role", "label", "date", "drawdown"]].copy()
    portfolio_equity_only = portfolio_equity.loc[:, ["config_id", "role", "label", "date", "equity"]].copy()
    portfolio_monthly_returns = _prepare_portfolio_monthly_returns(portfolio_outputs["monthly_returns"], spec_map)
    weights_history = _prepare_weights_history(
        wf_outputs["weights_timeseries"],
        wf_outputs["turnover"],
        wf_outputs["rebalance_log"],
        spec_map,
    )
    book_daily_contribution = _prepare_book_daily_contribution(
        book_returns_wide=book_returns_wide,
        weights_history=weights_history,
        portfolio_daily_returns=portfolio_daily_returns,
    )
    book_cumulative_contribution = _prepare_book_cumulative_contribution(book_daily_contribution)
    rolling_metrics = _prepare_rolling_metrics(portfolio_daily_returns, spec_map)
    drawdown_details = _prepare_drawdown_contribution_details(portfolio_equity, book_daily_contribution, spec_map)
    portfolio_summary = _prepare_portfolio_summary(
        generated_at=generated_at,
        portfolio_daily_returns=portfolio_daily_returns,
        portfolio_equity=portfolio_equity,
        rolling_metrics=rolling_metrics,
        run_level=portfolio_outputs["allocation_run_level"],
        spec_map=spec_map,
        allocator_order=allocator_order,
    )
    allocator_comparison = _prepare_allocator_comparison_summary(portfolio_summary)
    book_summary = _prepare_book_summary(
        book_returns_wide=book_returns_wide,
        weights_history=weights_history,
        contribution_summary=portfolio_outputs["book_contribution_summary"],
        portfolio_summary=portfolio_summary,
        drawdown_details=drawdown_details,
        spec_map=spec_map,
        allocator_order=allocator_order,
    )

    return {
        "generated_at": generated_at,
        "config": config,
        "specs": specs,
        "book_daily_returns": book_daily_returns,
        "portfolio_daily_returns": portfolio_daily_returns,
        "portfolio_daily_equity": portfolio_equity_only,
        "portfolio_daily_drawdown": portfolio_drawdown,
        "portfolio_monthly_returns": portfolio_monthly_returns,
        "book_weights_history": weights_history,
        "book_daily_contribution": book_daily_contribution,
        "book_cumulative_contribution": book_cumulative_contribution,
        "rolling_metrics": rolling_metrics,
        "drawdown_contribution_details": drawdown_details,
        "portfolio_daily_summary": portfolio_summary,
        "allocator_comparison_summary": allocator_comparison,
        "book_summary": book_summary,
    }


def _prepare_book_daily_returns(book_daily_long: pd.DataFrame) -> pd.DataFrame:
    out = book_daily_long.copy()
    out = out.rename(columns={"trade_date": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["daily_return"] = pd.to_numeric(out["daily_return"], errors="coerce")
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    return out.sort_values(["date", "book"]).reset_index(drop=True)


def _prepare_portfolio_daily_returns(portfolio_daily_returns: pd.DataFrame, spec_map: dict[str, Core4AllocatorSpec]) -> pd.DataFrame:
    out = portfolio_daily_returns.copy()
    out = out.rename(columns={"portfolio_daily_return": "daily_return"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).copy()
    out["daily_return"] = pd.to_numeric(out["daily_return"], errors="coerce").fillna(0.0)
    out["role"] = out["config_id"].map(lambda value: spec_map[str(value)].role)
    out["label"] = out["config_id"].map(lambda value: spec_map[str(value)].label)
    keep = ["config_id", "role", "label", "date", "daily_return", "method", "lookback_days", "rebalance_frequency", "constraint_label", "weight_floor", "weight_cap"]
    return out.loc[:, keep].sort_values(["config_id", "date"]).reset_index(drop=True)


def _prepare_portfolio_equity(portfolio_equity_curves: pd.DataFrame, spec_map: dict[str, Core4AllocatorSpec]) -> pd.DataFrame:
    out = portfolio_equity_curves.copy()
    out = out[out["compounding_mode"].eq("compounded")].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).copy()
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out["drawdown"] = pd.to_numeric(out["drawdown"], errors="coerce")
    out["role"] = out["config_id"].map(lambda value: spec_map[str(value)].role)
    out["label"] = out["config_id"].map(lambda value: spec_map[str(value)].label)
    out = out.loc[:, ["config_id", "role", "label", "date", "equity", "drawdown"]]
    return out.sort_values(["config_id", "date"]).reset_index(drop=True)


def _prepare_portfolio_monthly_returns(monthly_returns: pd.DataFrame, spec_map: dict[str, Core4AllocatorSpec]) -> pd.DataFrame:
    out = monthly_returns.copy()
    out = out[out["compounding_mode"].eq("compounded")].copy()
    out["trade_month"] = pd.to_datetime(out["trade_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["month_return"] = pd.to_numeric(out["month_return"], errors="coerce")
    out["role"] = out["config_id"].map(lambda value: spec_map[str(value)].role)
    out["label"] = out["config_id"].map(lambda value: spec_map[str(value)].label)
    keep = ["config_id", "role", "label", "trade_month", "month_return"]
    return out.loc[:, keep].sort_values(["config_id", "trade_month"]).reset_index(drop=True)


def _prepare_weights_history(
    weights_timeseries: pd.DataFrame,
    turnover: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    spec_map: dict[str, Core4AllocatorSpec],
) -> pd.DataFrame:
    weights = weights_timeseries.copy()
    weights["date"] = pd.to_datetime(weights["date"], errors="coerce").dt.normalize()
    weight_cols = [col for col in weights.columns if col.startswith("weight_")]
    books = [col.replace("weight_", "") for col in weight_cols]
    out = weights.melt(
        id_vars=["config_id", "method", "lookback_days", "rebalance_frequency", "constraint_label", "weight_floor", "weight_cap", "date"],
        value_vars=weight_cols,
        var_name="book",
        value_name="weight",
    )
    out["book"] = out["book"].str.replace("weight_", "", regex=False)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["role"] = out["config_id"].map(lambda value: spec_map[str(value)].role)
    out["label"] = out["config_id"].map(lambda value: spec_map[str(value)].label)

    turnover_map = turnover.copy()
    turnover_map["date"] = pd.to_datetime(turnover_map["date"], errors="coerce").dt.normalize()
    turnover_map["turnover"] = pd.to_numeric(turnover_map["turnover"], errors="coerce").fillna(0.0)
    out = out.merge(turnover_map.loc[:, ["config_id", "date", "turnover"]], on=["config_id", "date"], how="left")
    out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce").fillna(0.0)

    rebalance_dates = rebalance_log.copy()
    rebalance_dates["rebalance_date"] = pd.to_datetime(rebalance_dates["rebalance_date"], errors="coerce").dt.normalize()
    rebalance_dates = rebalance_dates.dropna(subset=["rebalance_date"])
    rebalance_pairs = set(zip(rebalance_dates["config_id"], rebalance_dates["rebalance_date"]))
    out["is_rebalance_date"] = out.apply(lambda row: (row["config_id"], row["date"]) in rebalance_pairs, axis=1)

    out = out.sort_values(["config_id", "book", "date"]).reset_index(drop=True)
    out["previous_weight"] = out.groupby(["config_id", "book"])["weight"].shift(1)
    out["weight_change"] = out["weight"] - out["previous_weight"].fillna(out["weight"])
    out["is_latest"] = False
    latest_idx = out.groupby(["config_id", "book"])["date"].idxmax()
    out.loc[latest_idx, "is_latest"] = True

    keep = [
        "config_id",
        "role",
        "label",
        "date",
        "book",
        "weight",
        "previous_weight",
        "weight_change",
        "is_rebalance_date",
        "turnover",
        "method",
        "lookback_days",
        "rebalance_frequency",
        "constraint_label",
        "weight_floor",
        "weight_cap",
        "is_latest",
    ]
    return out.loc[:, keep].sort_values(["config_id", "date", "book"]).reset_index(drop=True)


def _prepare_book_daily_contribution(
    *,
    book_returns_wide: pd.DataFrame,
    weights_history: pd.DataFrame,
    portfolio_daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    returns_long = (
        book_returns_wide.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="book", value_name="book_daily_return")
    )
    returns_long["date"] = pd.to_datetime(returns_long["date"], errors="coerce").dt.normalize()
    returns_long["book_daily_return"] = pd.to_numeric(returns_long["book_daily_return"], errors="coerce").fillna(0.0)

    out = weights_history.merge(returns_long, on=["date", "book"], how="left")
    out["book_daily_return"] = pd.to_numeric(out["book_daily_return"], errors="coerce").fillna(0.0)
    out["daily_contribution"] = out["weight"] * out["book_daily_return"]

    port = portfolio_daily_returns.loc[:, ["config_id", "date", "daily_return"]].rename(columns={"daily_return": "portfolio_daily_return"})
    out = out.merge(port, on=["config_id", "date"], how="left")
    out["portfolio_daily_return"] = pd.to_numeric(out["portfolio_daily_return"], errors="coerce").fillna(0.0)
    out["daily_contribution_share"] = np.where(
        out["portfolio_daily_return"].abs() > 1e-12,
        out["daily_contribution"] / out["portfolio_daily_return"],
        np.nan,
    )
    keep = [
        "config_id",
        "role",
        "label",
        "date",
        "book",
        "weight",
        "book_daily_return",
        "daily_contribution",
        "daily_contribution_share",
        "portfolio_daily_return",
        "is_rebalance_date",
        "turnover",
    ]
    return out.loc[:, keep].sort_values(["config_id", "date", "book"]).reset_index(drop=True)


def _prepare_book_cumulative_contribution(book_daily_contribution: pd.DataFrame) -> pd.DataFrame:
    out = book_daily_contribution.loc[:, ["config_id", "role", "label", "date", "book", "daily_contribution"]].copy()
    out["cumulative_contribution"] = out.groupby(["config_id", "book"])["daily_contribution"].cumsum()
    return out.sort_values(["config_id", "date", "book"]).reset_index(drop=True)


def _prepare_rolling_metrics(
    portfolio_daily_returns: pd.DataFrame,
    spec_map: dict[str, Core4AllocatorSpec],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config_id, group in portfolio_daily_returns.groupby("config_id", sort=False):
        series = group.set_index("date")["daily_return"].astype(float).sort_index()
        equity = equity_from_returns(series, "compounded")
        frame = pd.DataFrame(
            {
                "date": series.index,
                "config_id": config_id,
                "role": spec_map[str(config_id)].role,
                "label": spec_map[str(config_id)].label,
                "rolling_vol_63d": series.rolling(63, min_periods=21).std(ddof=1) * np.sqrt(TRADING_DAYS),
                "rolling_sharpe_126d": rolling_sharpe(series, window=126),
                "rolling_worst_drawdown_126d": _rolling_worst_drawdown(equity, window=126, min_periods=42),
                "rolling_worst_drawdown_252d": _rolling_worst_drawdown(equity, window=252, min_periods=63),
            }
        )
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    keep = [
        "config_id",
        "role",
        "label",
        "date",
        "rolling_vol_63d",
        "rolling_sharpe_126d",
        "rolling_worst_drawdown_126d",
        "rolling_worst_drawdown_252d",
    ]
    return out.loc[:, keep].sort_values(["config_id", "date"]).reset_index(drop=True)


def _prepare_drawdown_contribution_details(
    portfolio_equity: pd.DataFrame,
    book_daily_contribution: pd.DataFrame,
    spec_map: dict[str, Core4AllocatorSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_id, group in portfolio_equity.groupby("config_id", sort=False):
        equity = group.set_index("date")["equity"].astype(float).sort_index()
        if equity.empty:
            continue
        peak_date, trough_date = _peak_to_trough_window(equity)
        if pd.isna(peak_date) or pd.isna(trough_date):
            continue
        contrib = book_daily_contribution[
            book_daily_contribution["config_id"].eq(config_id)
            & book_daily_contribution["date"].gt(peak_date)
            & book_daily_contribution["date"].le(trough_date)
        ].copy()
        if contrib.empty:
            continue
        total_window_contribution = float(contrib["daily_contribution"].sum())
        max_drawdown = float(drawdown_from_equity(equity).min())
        for book, sub in contrib.groupby("book", sort=False):
            contribution = float(sub["daily_contribution"].sum())
            rows.append(
                {
                    "config_id": config_id,
                    "role": spec_map[str(config_id)].role,
                    "label": spec_map[str(config_id)].label,
                    "book": str(book),
                    "dd_peak_date": peak_date,
                    "dd_trough_date": trough_date,
                    "dd_window_days": int(contrib["date"].nunique()),
                    "max_drawdown": max_drawdown,
                    "window_additive_return": total_window_contribution,
                    "drawdown_contribution": contribution,
                    "drawdown_contribution_share": contribution / total_window_contribution if abs(total_window_contribution) > 1e-12 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["config_id", "book"]).reset_index(drop=True)


def _prepare_portfolio_summary(
    *,
    generated_at: datetime,
    portfolio_daily_returns: pd.DataFrame,
    portfolio_equity: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    run_level: pd.DataFrame,
    spec_map: dict[str, Core4AllocatorSpec],
    allocator_order: list[str],
) -> pd.DataFrame:
    run = run_level[run_level["compounding_mode"].eq("compounded")].copy()
    out = run.loc[
        :,
        [
            "config_id",
            "n_days",
            "start",
            "end",
            "total_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "avg_daily_turnover",
            "avg_rebalance_turnover",
            "cumulative_turnover",
            "nb_rebalances_with_turnover",
            "avg_effective_n_books",
            "avg_max_book_weight",
            "avg_weight_hhi",
            "avg_weight_std",
        ],
    ].copy()
    out["start"] = pd.to_datetime(out["start"], errors="coerce").dt.normalize()
    out["end"] = pd.to_datetime(out["end"], errors="coerce").dt.normalize()
    out["role"] = out["config_id"].map(lambda value: spec_map[str(value)].role)
    out["label"] = out["config_id"].map(lambda value: spec_map[str(value)].label)

    current_rows: list[dict[str, Any]] = []
    for config_id, group in portfolio_daily_returns.groupby("config_id", sort=False):
        daily = group.set_index("date")["daily_return"].astype(float).sort_index()
        eq = portfolio_equity[portfolio_equity["config_id"].eq(config_id)].set_index("date").sort_index()
        latest_date = pd.Timestamp(daily.index.max()) if not daily.empty else pd.NaT
        current_drawdown = float(eq["drawdown"].iloc[-1]) if not eq.empty else np.nan
        latest_equity = float(eq["equity"].iloc[-1]) if not eq.empty else np.nan
        ytd_year = int(latest_date.year) if pd.notna(latest_date) else np.nan
        ytd_return = _calendar_ytd_return(daily) if not daily.empty else np.nan
        current_rows.append(
            {
                "config_id": config_id,
                "latest_date": latest_date,
                "latest_equity": latest_equity,
                "current_drawdown": current_drawdown,
                "ytd_year": ytd_year,
                "ytd_return": ytd_return,
                "days_since_latest_data": (generated_at.date() - latest_date.date()).days if pd.notna(latest_date) else np.nan,
            }
        )
    current_df = pd.DataFrame(current_rows)
    out = out.merge(current_df, on="config_id", how="left")

    latest_rolling = rolling_metrics.sort_values("date").groupby("config_id", as_index=False).tail(1)
    latest_rolling = latest_rolling.loc[
        :,
        ["config_id", "rolling_vol_63d", "rolling_sharpe_126d", "rolling_worst_drawdown_126d", "rolling_worst_drawdown_252d"],
    ].copy()
    latest_rolling = latest_rolling.rename(
        columns={
            "rolling_vol_63d": "latest_rolling_vol_63d",
            "rolling_sharpe_126d": "latest_rolling_sharpe_126d",
            "rolling_worst_drawdown_126d": "latest_rolling_worst_drawdown_126d",
            "rolling_worst_drawdown_252d": "latest_rolling_worst_drawdown_252d",
        }
    )
    out = out.merge(latest_rolling, on="config_id", how="left")

    out["sort_order"] = out["config_id"].map({config_id: idx for idx, config_id in enumerate(allocator_order)})
    out = out.sort_values(["sort_order", "config_id"]).reset_index(drop=True)
    keep = [
        "config_id",
        "role",
        "label",
        "n_days",
        "start",
        "end",
        "latest_date",
        "days_since_latest_data",
        "total_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "current_drawdown",
        "ytd_year",
        "ytd_return",
        "latest_equity",
        "avg_daily_turnover",
        "avg_rebalance_turnover",
        "cumulative_turnover",
        "nb_rebalances_with_turnover",
        "avg_effective_n_books",
        "avg_max_book_weight",
        "avg_weight_hhi",
        "avg_weight_std",
        "latest_rolling_vol_63d",
        "latest_rolling_sharpe_126d",
        "latest_rolling_worst_drawdown_126d",
        "latest_rolling_worst_drawdown_252d",
    ]
    return out.loc[:, keep]


def _prepare_allocator_comparison_summary(portfolio_summary: pd.DataFrame) -> pd.DataFrame:
    out = portfolio_summary.copy()
    reference = out[out["role"].eq("reference")]
    if reference.empty:
        return out
    ref_row = reference.iloc[0]
    for metric in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "current_drawdown", "avg_rebalance_turnover", "avg_effective_n_books", "avg_max_book_weight"]:
        out[f"delta_{metric}_vs_reference"] = pd.to_numeric(out[metric], errors="coerce") - float(ref_row[metric])
    return out


def _prepare_book_summary(
    *,
    book_returns_wide: pd.DataFrame,
    weights_history: pd.DataFrame,
    contribution_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    drawdown_details: pd.DataFrame,
    spec_map: dict[str, Core4AllocatorSpec],
    allocator_order: list[str],
) -> pd.DataFrame:
    contrib = contribution_summary.copy()
    contrib["role"] = contrib["config_id"].map(lambda value: spec_map[str(value)].role)
    contrib["label"] = contrib["config_id"].map(lambda value: spec_map[str(value)].label)

    latest_weights = (
        weights_history[weights_history["is_latest"]]
        .loc[:, ["config_id", "book", "weight"]]
        .rename(columns={"weight": "current_weight"})
    )
    contrib = contrib.merge(latest_weights, on=["config_id", "book"], how="left")

    corr_rows: list[dict[str, Any]] = []
    for _, row in portfolio_summary.iterrows():
        config_id = str(row["config_id"])
        for book in book_returns_wide.columns:
            portfolio_corr = np.nan
            corr_col = f"corr_portfolio_{book}"
            if corr_col in portfolio_summary.columns:
                portfolio_corr = row.get(corr_col, np.nan)
            corr_rows.append(
                {
                    "config_id": config_id,
                    "book": str(book),
                    "portfolio_correlation": portfolio_corr,
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    if corr_df["portfolio_correlation"].isna().all():
        corr_rows = []
        for config_id, group in weights_history.groupby("config_id", sort=False):
            dates = pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())
            portfolio_returns = (
                group.pivot_table(index="date", columns="book", values="weight", aggfunc="last")
                .reindex(index=dates)
                .mul(book_returns_wide.reindex(index=dates), axis=0)
                .sum(axis=1)
            )
            for book in book_returns_wide.columns:
                book_returns = book_returns_wide.reindex(index=dates)[book]
                corr_rows.append(
                    {
                        "config_id": config_id,
                        "book": str(book),
                        "portfolio_correlation": float(portfolio_returns.corr(book_returns)) if len(portfolio_returns) > 2 else np.nan,
                    }
                )
        corr_df = pd.DataFrame(corr_rows)

    vol_rows = []
    for config_id, group in weights_history.groupby("config_id", sort=False):
        dates = pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())
        returns_slice = book_returns_wide.reindex(index=dates)
        for book in returns_slice.columns:
            series = pd.to_numeric(returns_slice[book], errors="coerce").dropna()
            vol_rows.append(
                {
                    "config_id": config_id,
                    "book": str(book),
                    "standalone_ann_vol": float(series.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(series) > 1 else np.nan,
                    "cumulative_book_return": float((1.0 + series).prod() - 1.0) if not series.empty else np.nan,
                }
            )
    vol_df = pd.DataFrame(vol_rows)

    drawdown_df = drawdown_details.loc[:, ["config_id", "book", "drawdown_contribution", "drawdown_contribution_share"]].copy()
    out = contrib.merge(corr_df, on=["config_id", "book"], how="left")
    out = out.merge(vol_df, on=["config_id", "book"], how="left")
    out = out.merge(drawdown_df, on=["config_id", "book"], how="left")
    out["sort_order"] = out["config_id"].map({config_id: idx for idx, config_id in enumerate(allocator_order)})
    out = out.sort_values(["sort_order", "book"]).reset_index(drop=True)
    keep = [
        "config_id",
        "role",
        "label",
        "book",
        "current_weight",
        "avg_weight",
        "median_weight",
        "min_weight",
        "max_weight",
        "weight_std",
        "cumulative_contribution",
        "average_daily_contribution",
        "contribution_share",
        "cumulative_book_return",
        "standalone_total_return",
        "standalone_ann_vol",
        "portfolio_correlation",
        "drawdown_contribution",
        "drawdown_contribution_share",
    ]
    return out.loc[:, keep]


def _write_bundle(output_dir: Path, bundle: dict[str, Any]) -> None:
    file_map = {
        "portfolio_daily_summary": "portfolio_daily_summary.csv",
        "portfolio_daily_returns": "portfolio_daily_returns.csv",
        "portfolio_daily_equity": "portfolio_daily_equity.csv",
        "portfolio_daily_drawdown": "portfolio_daily_drawdown.csv",
        "portfolio_monthly_returns": "portfolio_monthly_returns.csv",
        "book_daily_returns": "book_daily_returns.csv",
        "book_daily_contribution": "book_daily_contribution.csv",
        "book_cumulative_contribution": "book_cumulative_contribution.csv",
        "book_weights_history": "book_weights_history.csv",
        "rolling_metrics": "rolling_metrics.csv",
        "allocator_comparison_summary": "allocator_comparison_summary.csv",
        "drawdown_contribution_details": "drawdown_contribution_details.csv",
        "book_summary": "book_summary.csv",
    }
    for key, filename in file_map.items():
        frame = bundle.get(key, pd.DataFrame())
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(output_dir / filename, index=False)


def _write_plots(output_dir: Path, bundle: dict[str, Any]) -> None:
    _plot_equity_drawdown(
        output_dir / "core4_equity_drawdown.png",
        bundle["portfolio_daily_equity"],
        bundle["portfolio_daily_drawdown"],
    )
    _plot_weights(output_dir / "core4_weights.png", bundle["book_weights_history"])
    _plot_book_contribution(output_dir / "core4_book_contribution.png", bundle["book_cumulative_contribution"])
    _plot_rolling_metrics(output_dir / "core4_rolling_metrics.png", bundle["rolling_metrics"])
    _plot_allocator_comparison(output_dir / "core4_allocator_comparison.png", bundle["allocator_comparison_summary"])
    _plot_monthly_heatmap(output_dir / "core4_monthly_heatmap.png", bundle["portfolio_monthly_returns"])


def _plot_equity_drawdown(path: Path, equity: pd.DataFrame, drawdown: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})
    for config_id, group in equity.groupby("config_id", sort=False):
        color = ALLOCATOR_COLORS.get(str(config_id), "#4c566a")
        label = group["label"].iloc[0]
        axes[0].plot(group["date"], group["equity"], label=label, color=color, linewidth=2.0)
    axes[0].set_title("Core 4 portfolio equity")
    axes[0].set_ylabel("Equity (base 1.0)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    for config_id, group in drawdown.groupby("config_id", sort=False):
        color = ALLOCATOR_COLORS.get(str(config_id), "#4c566a")
        axes[1].plot(group["date"], group["drawdown"], label=group["label"].iloc[0], color=color, linewidth=1.8)
    axes[1].axhline(0.0, color="#808080", linewidth=0.8)
    axes[1].set_title("Core 4 portfolio drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_weights(path: Path, weights_history: pd.DataFrame) -> None:
    config_ids = list(weights_history["config_id"].drop_duplicates())
    n_rows = len(config_ids)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.3 * max(1, n_rows)), sharex=True)
    axes_list = list(np.atleast_1d(axes))
    for ax, config_id in zip(axes_list, config_ids):
        sub = weights_history[weights_history["config_id"].eq(config_id)].copy()
        for book, group in sub.groupby("book", sort=False):
            ax.plot(group["date"], group["weight"], label=str(book), linewidth=1.8, color=BOOK_COLORS.get(str(book), "#4c566a"))
        label = sub["label"].iloc[0]
        ax.set_title(f"{label} weights")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncol=4)
    axes_list[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_book_contribution(path: Path, cumulative_contribution: pd.DataFrame) -> None:
    config_ids = list(cumulative_contribution["config_id"].drop_duplicates())
    n_rows = len(config_ids)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.3 * max(1, n_rows)), sharex=True)
    axes_list = list(np.atleast_1d(axes))
    for ax, config_id in zip(axes_list, config_ids):
        sub = cumulative_contribution[cumulative_contribution["config_id"].eq(config_id)].copy()
        for book, group in sub.groupby("book", sort=False):
            ax.plot(group["date"], group["cumulative_contribution"], label=str(book), linewidth=1.8, color=BOOK_COLORS.get(str(book), "#4c566a"))
        label = sub["label"].iloc[0]
        ax.axhline(0.0, color="#808080", linewidth=0.8)
        ax.set_title(f"{label} cumulative contribution by book")
        ax.set_ylabel("Additive contribution")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncol=4)
    axes_list[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rolling_metrics(path: Path, rolling_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    metric_specs = [
        ("rolling_vol_63d", "Rolling vol 63d"),
        ("rolling_sharpe_126d", "Rolling Sharpe 126d"),
        ("rolling_worst_drawdown_252d", "Rolling worst drawdown 252d"),
    ]
    for ax, (column, title) in zip(axes, metric_specs):
        for config_id, group in rolling_metrics.groupby("config_id", sort=False):
            color = ALLOCATOR_COLORS.get(str(config_id), "#4c566a")
            ax.plot(group["date"], group[column], label=group["label"].iloc[0], color=color, linewidth=1.8)
        if "drawdown" in column:
            ax.axhline(0.0, color="#808080", linewidth=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_allocator_comparison(path: Path, allocator_summary: pd.DataFrame) -> None:
    metrics = [
        ("ann_return", "Annualized return", True),
        ("ann_vol", "Annualized vol", True),
        ("sharpe", "Sharpe", False),
        ("max_drawdown", "Max drawdown", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows = allocator_summary.copy()
    labels = rows["label"].tolist()
    colors = [ALLOCATOR_COLORS.get(str(config_id), "#4c566a") for config_id in rows["config_id"]]
    for ax, (column, title, percent_axis) in zip(axes.flatten(), metrics):
        values = pd.to_numeric(rows[column], errors="coerce")
        ax.bar(labels, values, color=colors, alpha=0.92)
        ax.set_title(title)
        if percent_axis:
            ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_monthly_heatmap(path: Path, monthly_returns: pd.DataFrame) -> None:
    ref = monthly_returns[monthly_returns["role"].eq("reference")].copy()
    if ref.empty:
        return
    ref["year"] = ref["trade_month"].dt.year
    ref["month_name"] = ref["trade_month"].dt.month.map(MONTH_NAME_MAP)
    pivot = ref.pivot_table(index="year", columns="month_name", values="month_return", aggfunc="last")
    ordered_columns = [MONTH_NAME_MAP[m] for m in range(1, 13)]
    pivot = pivot.reindex(columns=ordered_columns)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    values = pivot.to_numpy(dtype=float)
    masked = np.where(np.isfinite(values), values, 0.0)
    vmax = max(np.nanpercentile(np.abs(masked), 95), 0.01)
    image = ax.imshow(masked, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Reference allocator monthly returns heatmap")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.1%}", ha="center", va="center", fontsize=8, color="#1f1f1f")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_metadata(
    *,
    config: dict[str, Any],
    specs: tuple[Core4AllocatorSpec, ...],
    options: Core4ReportingOptions,
    generated_at: datetime,
    bundle: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    summary = bundle["portfolio_daily_summary"].copy()
    latest_data_date = pd.to_datetime(summary["latest_date"], errors="coerce").max()
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "output_dir": str(_resolve_output_dir(project_root, options.output_dir)),
        "core_config_path": str(_resolve_path(project_root, options.config_path)),
        "daily_cache_dir": str(_resolve_path(project_root, options.daily_cache_dir)),
        "latest_available_data_date": latest_data_date.strftime("%Y-%m-%d") if pd.notna(latest_data_date) else None,
        "days_since_latest_data": int((generated_at.date() - latest_data_date.date()).days) if pd.notna(latest_data_date) else None,
        "portfolio_definition": config,
        "selected_allocators": [asdict(spec) for spec in specs],
        "primary_allocators": list(PRIMARY_ALLOCATOR_IDS),
        "optional_allocators": list(OPTIONAL_ALLOCATOR_IDS),
        "reporting_mode": "compounded",
        "assumptions": [
            "Daily book returns come from the frozen core 4 country books and reuse the existing cache when available.",
            "Allocator weights are walk-forward and use only trailing lookback data with lag_days applied.",
            "Portfolio equity, drawdown and monthly returns are reported in compounded mode.",
            "Book cumulative contribution is additive: it is the running sum of daily weighted contributions.",
            "Drawdown contribution is approximated over the peak-to-trough window of each allocator max drawdown using additive daily contributions.",
            "YTD return is computed for the calendar year of the latest available data date, not the generation date when data are stale.",
        ],
        "run_options": {
            "output_dir": str(options.output_dir),
            "config_path": str(options.config_path),
            "daily_cache_dir": str(options.daily_cache_dir),
            "start": options.start,
            "end": options.end,
            "lag_days": options.lag_days,
            "include_optional_allocators": options.include_optional_allocators,
            "rebuild_daily_cache": options.rebuild_daily_cache,
            "smoke": options.smoke,
        },
        "files": {
            "portfolio_daily_summary": "portfolio_daily_summary.csv",
            "portfolio_daily_returns": "portfolio_daily_returns.csv",
            "portfolio_daily_equity": "portfolio_daily_equity.csv",
            "portfolio_daily_drawdown": "portfolio_daily_drawdown.csv",
            "portfolio_monthly_returns": "portfolio_monthly_returns.csv",
            "book_daily_returns": "book_daily_returns.csv",
            "book_daily_contribution": "book_daily_contribution.csv",
            "book_cumulative_contribution": "book_cumulative_contribution.csv",
            "book_weights_history": "book_weights_history.csv",
            "rolling_metrics": "rolling_metrics.csv",
            "allocator_comparison_summary": "allocator_comparison_summary.csv",
            "drawdown_contribution_details": "drawdown_contribution_details.csv",
            "book_summary": "book_summary.csv",
            "core4_equity_drawdown_png": "core4_equity_drawdown.png",
            "core4_weights_png": "core4_weights.png",
            "core4_book_contribution_png": "core4_book_contribution.png",
            "core4_rolling_metrics_png": "core4_rolling_metrics.png",
            "core4_allocator_comparison_png": "core4_allocator_comparison.png",
            "core4_monthly_heatmap_png": "core4_monthly_heatmap.png",
            "core4_daily_report_html": "core4_daily_report.html",
            "conclusion_txt": "conclusion.txt",
            "metadata_json": "metadata.json",
        },
    }


def _build_conclusion(bundle: dict[str, Any]) -> str:
    summary = bundle["portfolio_daily_summary"].copy()
    reference = summary[summary["role"].eq("reference")]
    benchmark = summary[summary["role"].eq("benchmark")]
    optional = summary[summary["role"].eq("comparison")]
    latest_date = pd.to_datetime(summary["latest_date"], errors="coerce").max()
    lag_days = int(summary["days_since_latest_data"].max()) if not summary["days_since_latest_data"].isna().all() else np.nan

    lines = [
        "Core 4 daily reporting conclusion",
        "",
        f"Latest available data date: {latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else 'n/a'}",
        f"Data staleness vs generation date: {lag_days if np.isfinite(lag_days) else 'n/a'} calendar days",
    ]

    if not reference.empty:
        row = reference.iloc[0]
        lines.append(
            "Reference allocator: "
            f"{row['config_id']} | ann_return={row['ann_return']:.2%} | ann_vol={row['ann_vol']:.2%} | "
            f"Sharpe={row['sharpe']:.2f} | maxDD={row['max_drawdown']:.2%} | currentDD={row['current_drawdown']:.2%}"
        )
    if not benchmark.empty:
        row = benchmark.iloc[0]
        lines.append(
            "Benchmark allocator: "
            f"{row['config_id']} | ann_return={row['ann_return']:.2%} | ann_vol={row['ann_vol']:.2%} | "
            f"Sharpe={row['sharpe']:.2f} | maxDD={row['max_drawdown']:.2%} | currentDD={row['current_drawdown']:.2%}"
        )
    if not reference.empty and not benchmark.empty:
        ref = reference.iloc[0]
        bench = benchmark.iloc[0]
        lines.append(
            "Reference vs benchmark: "
            f"delta ann_return={ref['ann_return'] - bench['ann_return']:+.2%} | "
            f"delta Sharpe={ref['sharpe'] - bench['sharpe']:+.2f} | "
            f"delta maxDD={ref['max_drawdown'] - bench['max_drawdown']:+.2%}"
        )
        lines.append(
            "Decision read-through: "
            "the designated reference allocator keeps a shallower max drawdown than the equal-weight benchmark, "
            "while the benchmark remains the higher-return/higher-Sharpe control in this sample."
        )
    if not optional.empty:
        row = optional.iloc[0]
        lines.append(
            "Optional monitor: "
            f"{row['config_id']} | ann_return={row['ann_return']:.2%} | ann_vol={row['ann_vol']:.2%} | "
            f"Sharpe={row['sharpe']:.2f} | maxDD={row['max_drawdown']:.2%}"
        )
    return "\n".join(lines) + "\n"


def _render_html_report(bundle: dict[str, Any]) -> str:
    config = bundle["config"]
    summary = bundle["portfolio_daily_summary"].copy()
    comparison = bundle["allocator_comparison_summary"].copy()
    book_summary = bundle["book_summary"].copy()
    weights_history = bundle["book_weights_history"].copy()
    latest_weights = weights_history[weights_history["is_latest"]].copy()
    generated_at = bundle["generated_at"]
    latest_date = pd.to_datetime(summary["latest_date"], errors="coerce").max()
    data_lag = int(summary["days_since_latest_data"].max()) if not summary["days_since_latest_data"].isna().all() else np.nan

    core_definition = pd.DataFrame(
        [
            {
                "book": book.get("book"),
                "country": book.get("country"),
                "config_name": book.get("config_name"),
                "role": book.get("role"),
                "logic": book.get("logic"),
            }
            for book in config.get("books", [])
        ]
    )

    summary_display = summary.loc[
        :,
        [
            "label",
            "config_id",
            "ann_return",
            "ann_vol",
            "sharpe",
            "sortino",
            "max_drawdown",
            "current_drawdown",
            "calmar",
            "avg_rebalance_turnover",
            "avg_effective_n_books",
            "avg_max_book_weight",
            "latest_date",
            "days_since_latest_data",
        ],
    ].copy()
    comparison_display = comparison.loc[
        :,
        [
            "label",
            "config_id",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "delta_ann_return_vs_reference",
            "delta_sharpe_vs_reference",
            "delta_max_drawdown_vs_reference",
        ],
    ].copy()

    current_weights_display = latest_weights.loc[:, ["label", "book", "weight", "date"]].copy()
    current_weights_display = current_weights_display.rename(columns={"date": "latest_weight_date"})

    reference_books = book_summary[book_summary["role"].eq("reference")].copy()
    benchmark_books = book_summary[book_summary["role"].eq("benchmark")].copy()

    auto_comment = _automatic_comment(summary)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Core 4 Daily Report</title>
  <style>
    body {{
      font-family: "Segoe UI", Tahoma, Arial, sans-serif;
      margin: 28px auto;
      max-width: 1180px;
      color: #1f1f1f;
      line-height: 1.45;
      background: #f7f7f5;
      padding: 0 20px 36px 20px;
    }}
    .page {{
      background: #ffffff;
      border: 1px solid #d9ddd7;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.05);
      padding: 28px 34px 34px 34px;
    }}
    h1, h2 {{
      color: #17324d;
      margin-top: 0;
    }}
    h2 {{
      margin-top: 34px;
      border-bottom: 1px solid #d9ddd7;
      padding-bottom: 6px;
    }}
    p.note {{
      color: #4c5a67;
      margin-top: 6px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 14px;
      margin: 20px 0 10px 0;
    }}
    .card {{
      border: 1px solid #d9ddd7;
      background: #fbfcfa;
      padding: 12px 14px;
    }}
    .card strong {{
      display: block;
      color: #17324d;
      margin-bottom: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 18px 0;
      font-size: 0.94rem;
    }}
    th, td {{
      border: 1px solid #d9ddd7;
      padding: 7px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef2f0;
      color: #17324d;
    }}
    img {{
      width: 100%;
      border: 1px solid #d9ddd7;
      margin: 12px 0 18px 0;
      background: white;
    }}
    .small {{
      font-size: 0.9rem;
      color: #4c5a67;
    }}
    ul {{
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h2>1. Titre + Date De Generation</h2>
    <h1>Core 4 Daily Report</h1>
    <p class="note">Generated at {escape(generated_at.isoformat(sep=" ", timespec="seconds"))}. Latest available data date: {escape(latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "n/a")}.</p>
    <div class="meta">
      <div class="card"><strong>Portfolio</strong>{escape(str(config.get("portfolio_id", "core_4_country_v1")))}</div>
      <div class="card"><strong>Reference allocator</strong>inverse_vol__lb126__weekly__floor_cap</div>
      <div class="card"><strong>Data freshness</strong>{escape(str(data_lag))} calendar days vs generation date</div>
    </div>

    <h2>2. Definition Du Core 4-Livres</h2>
    {_table_html(core_definition, percent_columns=set(), date_columns=set())}

    <h2>3. Allocateur De Reference</h2>
    <p>Le reporting est pilote sur <code>inverse_vol__lb126__weekly__floor_cap</code>. Le benchmark de controle conserve est <code>equal_weight__lb126__monthly__unconstrained</code>.</p>

    <h2>4. Metriques Globales Portefeuille</h2>
    {_table_html(summary_display, percent_columns={"ann_return", "ann_vol", "max_drawdown", "current_drawdown", "avg_rebalance_turnover", "avg_max_book_weight"}, date_columns={"latest_date"})}

    <h2>5. Equity + Drawdown</h2>
    <img src="core4_equity_drawdown.png" alt="Core 4 equity and drawdown">

    <h2>6. Comparaison Reference Vs Equal Weight</h2>
    {_table_html(comparison_display[comparison_display["config_id"].isin(PRIMARY_ALLOCATOR_IDS)], percent_columns={"ann_return", "ann_vol", "max_drawdown", "delta_ann_return_vs_reference", "delta_max_drawdown_vs_reference"}, date_columns=set())}
    <img src="core4_allocator_comparison.png" alt="Allocator comparison">

    <h2>7. Poids Par Livre</h2>
    {_table_html(current_weights_display, percent_columns={"weight"}, date_columns={"latest_weight_date"})}
    <img src="core4_weights.png" alt="Core 4 weights by book">

    <h2>8. Contribution Par Livre</h2>
    <p class="small">Contribution cumulee et contribution au drawdown sont presentees en additif pour garder une attribution simple et stable.</p>
    <h3>Reference allocator</h3>
    {_table_html(reference_books, percent_columns={"current_weight", "avg_weight", "median_weight", "min_weight", "max_weight", "weight_std", "contribution_share", "cumulative_book_return", "standalone_total_return", "standalone_ann_vol", "drawdown_contribution_share"}, date_columns=set())}
    <h3>Benchmark</h3>
    {_table_html(benchmark_books, percent_columns={"current_weight", "avg_weight", "median_weight", "min_weight", "max_weight", "weight_std", "contribution_share", "cumulative_book_return", "standalone_total_return", "standalone_ann_vol", "drawdown_contribution_share"}, date_columns=set())}
    <img src="core4_book_contribution.png" alt="Core 4 cumulative contribution by book">

    <h2>9. Rolling Metrics</h2>
    <img src="core4_rolling_metrics.png" alt="Core 4 rolling metrics">

    <h2>10. Commentaires / Conclusion Automatique Courte</h2>
    <ul>
      {"".join(f"<li>{escape(line)}</li>" for line in auto_comment)}
    </ul>

    <h2>Appendix. Fichiers Produits</h2>
    <p class="small">CSV principaux: <code>portfolio_daily_summary.csv</code>, <code>portfolio_daily_returns.csv</code>, <code>portfolio_daily_equity.csv</code>, <code>portfolio_daily_drawdown.csv</code>, <code>portfolio_monthly_returns.csv</code>, <code>book_daily_returns.csv</code>, <code>book_daily_contribution.csv</code>, <code>book_cumulative_contribution.csv</code>, <code>book_weights_history.csv</code>, <code>rolling_metrics.csv</code>, <code>allocator_comparison_summary.csv</code>, <code>drawdown_contribution_details.csv</code>.</p>
  </div>
</body>
</html>
"""
    return html


def _automatic_comment(summary: pd.DataFrame) -> list[str]:
    comments: list[str] = []
    reference = summary[summary["role"].eq("reference")]
    benchmark = summary[summary["role"].eq("benchmark")]
    optional = summary[summary["role"].eq("comparison")]
    if not reference.empty:
        ref = reference.iloc[0]
        comments.append(
            f"Reference allocator annualized return {ref['ann_return']:.2%}, vol {ref['ann_vol']:.2%}, Sharpe {ref['sharpe']:.2f}, max drawdown {ref['max_drawdown']:.2%}."
        )
        comments.append(
            f"Reference allocator current drawdown {ref['current_drawdown']:.2%} and latest 63d rolling vol {ref['latest_rolling_vol_63d']:.2%}."
        )
    if not reference.empty and not benchmark.empty:
        ref = reference.iloc[0]
        bench = benchmark.iloc[0]
        comments.append(
            f"Benchmark equal weight is ahead by {bench['ann_return'] - ref['ann_return']:+.2%} annualized return and {bench['sharpe'] - ref['sharpe']:+.2f} Sharpe versus the designated reference."
        )
        comments.append(
            f"Reference allocator keeps max drawdown tighter by {ref['max_drawdown'] - bench['max_drawdown']:+.2%} and uses average max book weight {ref['avg_max_book_weight']:.2%}."
        )
    if not optional.empty:
        row = optional.iloc[0]
        comments.append(
            f"Optional risk parity monitor stays close to the reference with annualized return {row['ann_return']:.2%} and max drawdown {row['max_drawdown']:.2%}."
        )
    lag = summary["days_since_latest_data"].max()
    if pd.notna(lag) and float(lag) > 0:
        comments.append(
            f"Data are not current to the generation date: latest available portfolio date lags by {int(lag)} calendar days."
        )
    return comments


def _rolling_worst_drawdown(equity: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    out = pd.Series(np.nan, index=eq.index, dtype=float)
    values = eq.to_numpy(dtype=float)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        span = values[left : idx + 1]
        if len(span) < min_periods:
            continue
        local_peak = np.maximum.accumulate(span)
        drawdown = span / local_peak - 1.0
        out.iloc[idx] = float(np.min(drawdown))
    return out


def _calendar_ytd_return(returns: pd.Series) -> float:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return np.nan
    latest_year = int(series.index.max().year)
    ytd = series[series.index.year == latest_year]
    if ytd.empty:
        return np.nan
    return float((1.0 + ytd).prod() - 1.0)


def _peak_to_trough_window(equity: pd.Series) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    drawdown = drawdown_from_equity(equity)
    if drawdown.empty:
        return pd.NaT, pd.NaT
    trough = drawdown.idxmin()
    if pd.isna(trough):
        return pd.NaT, pd.NaT
    peak = equity.loc[:trough].idxmax()
    return pd.Timestamp(peak), pd.Timestamp(trough)


def _table_html(df: pd.DataFrame, *, percent_columns: set[str], date_columns: set[str]) -> str:
    if df.empty:
        return "<p class=\"small\">No data available.</p>"

    out = df.copy()
    for column in out.columns:
        if column in date_columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
        elif column in percent_columns:
            out[column] = out[column].map(_fmt_percent)
        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(_fmt_number)
        else:
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else str(value))
    return out.to_html(index=False, escape=False, border=0)


def _fmt_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.2%}"


def _fmt_number(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return ""
    if not np.isfinite(numeric):
        return ""
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


def _resolve_output_dir(project_root: Path, output_dir: Path) -> Path:
    return _resolve_path(project_root, output_dir)


def _resolve_path(project_root: Path, path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")
