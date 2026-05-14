from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.core_portfolio_daily import load_or_build_core_daily_returns
from utils.portfolio_allocation_research import apply_floor_cap, compute_allocation_weights, drawdown_from_equity, equity_from_returns


LOGGER = logging.getLogger("core4_validation_pack")
TRADING_DAYS = 252
DEFAULT_OUTPUT_ROOT = Path("data/experiments/core4_validation_pack")
DEFAULT_CONFIG_PATH = Path("config/core_portfolio_reference.json")
DEFAULT_DAILY_CACHE_DIR = Path("data/experiments/core_portfolio_reference_daily_cache")
SMOKE_START = "2024-01-01"


@dataclass(frozen=True)
class Core4ValidationOptions:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    daily_cache_dir: Path = DEFAULT_DAILY_CACHE_DIR
    start: str | None = None
    end: str | None = None
    rebuild_daily_cache: bool = False
    smoke: bool = False
    random_seed: int = 42
    random_portfolios: int = 100


def run_core4_validation_pack(options: Core4ValidationOptions, *, project_root: Path) -> Path:
    generated_at = datetime.now().replace(microsecond=0)
    output_dir = build_output_dir(_resolve_path(project_root, options.output_root), generated_at, smoke=options.smoke)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading frozen core4 reference inputs")
    context = load_frozen_reference_context(options, project_root=project_root)

    country_returns = context["country_returns"]
    portfolio_returns = context["portfolio_returns"]
    portfolio_equity = context["portfolio_equity"]
    frozen_weights = context["frozen_weights"]

    LOGGER.info("Computing validation diagnostics")
    performance_by_year = compute_performance_by_year(portfolio_returns)
    performance_by_period, skipped_periods = compute_period_metrics(portfolio_returns, build_custom_periods())
    daily_country_contribution, daily_country_contribution_wide = compute_daily_country_contribution(
        country_returns,
        frozen_weights,
        portfolio_returns=portfolio_returns,
    )
    annual_country_contribution = compute_annual_country_contribution(daily_country_contribution)
    annual_country_metrics = compute_annual_country_metrics(country_returns, portfolio_returns)
    leave_one_out = compute_leave_one_out(country_returns, portfolio_returns)
    leave_one_out_by_year = compute_leave_one_out_by_year(country_returns, portfolio_returns)
    country_standalone = compute_country_standalone(country_returns, portfolio_returns, frozen_weights=frozen_weights)
    weight_sensitivity = compute_weight_sensitivity(
        country_returns,
        frozen_weights=frozen_weights,
        seed=int(options.random_seed),
        n_random=int(options.random_portfolios),
    )
    rolling_metrics = compute_rolling_metrics(portfolio_returns, windows=(63, 126, 252))
    drawdown_events = compute_drawdown_events(portfolio_equity)
    drawdown_attribution_by_country = compute_drawdown_attribution(drawdown_events, daily_country_contribution)
    worst_periods_attribution = compute_worst_periods_attribution(
        portfolio_returns,
        daily_country_contribution,
        windows=(21, 63, 126),
        top_n=10,
    )
    correlation_matrix = country_returns.corr()
    monthly_returns, monthly_returns_pivot = compute_monthly_returns(portfolio_returns)
    country_contribution = compute_country_contribution(country_returns, frozen_weights, portfolio_returns)
    daily_exposure = compute_daily_exposure(context["daily_long"], frozen_weights)
    daily_turnover = compute_daily_turnover_from_positions(daily_exposure)
    portfolio_turnover = extract_portfolio_turnover_series(daily_turnover)
    turnover_info = summarize_turnover_input(daily_turnover)
    cost_stress, cost_warning = compute_cost_stress(portfolio_returns, turnover=portfolio_turnover)
    cost_stress_turnover_based = compute_turnover_based_cost_stress(portfolio_returns, daily_turnover)
    execution_delay_stress = compute_execution_delay_stress(
        country_returns=country_returns,
        portfolio_returns=portfolio_returns,
        daily_exposure=daily_exposure,
    )
    trade_sources = discover_trade_ledgers(context["config"], project_root=project_root)
    trade_ledger_portfolio, trade_ledger_schema_md, trade_warning = harmonize_trade_ledger(trade_sources)
    trade_concentration = compute_trade_concentration_summary(trade_ledger_portfolio, trade_sources)
    trade_concentration_extended = compute_trade_concentration_extended(trade_ledger_portfolio, trade_sources)
    drawdown_2023_md = write_drawdown_2023_deep_dive(
        drawdown_events=drawdown_events,
        drawdown_attribution_by_country=drawdown_attribution_by_country,
        country_returns=country_returns,
        daily_turnover=daily_turnover,
        trade_ledger=trade_ledger_portfolio,
        output_path=output_dir / "drawdown_2023_deep_dive.md",
    )

    warnings_list: list[str] = []
    if skipped_periods:
        warnings_list.append(f"Skipped custom periods with no overlap in the analysis window: {', '.join(skipped_periods)}.")
    if cost_warning:
        warnings_list.append(cost_warning)
    if turnover_info.get("warning"):
        warnings_list.append(str(turnover_info["warning"]))
    if trade_warning:
        warnings_list.append(trade_warning)
    if not execution_delay_stress.empty and str(execution_delay_stress.iloc[0].get("scenario", "")) == "not_available":
        warnings_list.append(str(execution_delay_stress.iloc[0].get("assumption", "")))

    bundle = {
        "performance_by_year": performance_by_year,
        "performance_by_period": performance_by_period,
        "daily_country_contribution": daily_country_contribution,
        "daily_country_contribution_wide": daily_country_contribution_wide,
        "annual_country_contribution": annual_country_contribution,
        "annual_country_metrics": annual_country_metrics,
        "leave_one_country_out": leave_one_out,
        "leave_one_country_out_by_year": leave_one_out_by_year,
        "country_standalone": country_standalone,
        "weight_sensitivity": weight_sensitivity,
        "cost_stress": cost_stress,
        "cost_stress_turnover_based": cost_stress_turnover_based,
        "trade_concentration": trade_concentration,
        "trade_concentration_extended": trade_concentration_extended,
        "trade_ledger_portfolio": trade_ledger_portfolio,
        "rolling_metrics": rolling_metrics,
        "drawdown_events": drawdown_events,
        "drawdown_attribution_by_country": drawdown_attribution_by_country,
        "worst_periods_attribution": worst_periods_attribution,
        "correlation_matrix": correlation_matrix,
        "monthly_returns": monthly_returns,
        "monthly_returns_pivot": monthly_returns_pivot,
        "daily_exposure": daily_exposure,
        "daily_turnover": daily_turnover,
        "execution_delay_stress": execution_delay_stress,
    }

    LOGGER.info("Writing CSV exports")
    export_validation_pack(output_dir, bundle)
    (output_dir / "trade_ledger_schema.md").write_text(trade_ledger_schema_md, encoding="utf-8")
    (output_dir / "drawdown_2023_deep_dive.md").write_text(drawdown_2023_md, encoding="utf-8")

    LOGGER.info("Writing plots")
    plot_equity_curve(plots_dir / "equity_curve.png", portfolio_equity)
    plot_drawdown(plots_dir / "drawdown.png", compute_drawdown(portfolio_equity))
    plot_yearly_returns(plots_dir / "yearly_returns.png", performance_by_year)
    plot_rolling_sharpe(plots_dir / "rolling_sharpe.png", rolling_metrics)
    plot_country_contribution(plots_dir / "country_contribution.png", country_contribution)
    plot_correlation_matrix(plots_dir / "correlation_matrix.png", correlation_matrix)
    plot_annual_country_contribution(plots_dir / "annual_country_contribution.png", annual_country_contribution)
    plot_drawdown_attribution_by_country(plots_dir / "drawdown_attribution_by_country.png", drawdown_attribution_by_country)
    plot_daily_turnover(plots_dir / "daily_turnover.png", daily_turnover)
    plot_cost_stress_turnover_based(plots_dir / "cost_stress_turnover_based.png", cost_stress_turnover_based)
    plot_worst_periods_by_country(plots_dir / "worst_periods_by_country.png", worst_periods_attribution)

    LOGGER.info("Building markdown summary")
    summary_text = build_validation_summary(
        generated_at=generated_at,
        context=context,
        performance_by_year=performance_by_year,
        performance_by_period=performance_by_period,
        annual_country_contribution=annual_country_contribution,
        annual_country_metrics=annual_country_metrics,
        leave_one_out=leave_one_out,
        leave_one_out_by_year=leave_one_out_by_year,
        country_standalone=country_standalone,
        weight_sensitivity=weight_sensitivity,
        cost_stress=cost_stress,
        cost_stress_turnover_based=cost_stress_turnover_based,
        trade_concentration=trade_concentration,
        trade_concentration_extended=trade_concentration_extended,
        trade_ledger_portfolio=trade_ledger_portfolio,
        drawdown_events=drawdown_events,
        drawdown_attribution_by_country=drawdown_attribution_by_country,
        correlation_matrix=correlation_matrix,
        country_contribution=country_contribution,
        daily_turnover=daily_turnover,
        execution_delay_stress=execution_delay_stress,
        drawdown_2023_md=drawdown_2023_md,
        warnings_list=warnings_list,
    )
    (output_dir / "validation_summary.md").write_text(summary_text, encoding="utf-8")

    LOGGER.info("Core4 validation pack generated in %s", output_dir)
    return output_dir


def load_frozen_reference_context(options: Core4ValidationOptions, *, project_root: Path) -> dict[str, Any]:
    config_path = _resolve_path(project_root, options.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Frozen core4 config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))
    period = config.get("period", {})
    full_start = str(period.get("start", "2018-01-01"))
    full_end = str(period.get("end", "2025-12-31"))

    analysis_start = str(options.start or (max(full_start, SMOKE_START) if options.smoke else full_start))
    analysis_end = str(options.end or full_end)
    if pd.Timestamp(analysis_end) < pd.Timestamp(analysis_start):
        raise ValueError(f"Invalid analysis window: start={analysis_start} is after end={analysis_end}")

    cache_dir = _resolve_path(project_root, options.daily_cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Daily cache directory not found: {cache_dir}")

    analysis_config = json.loads(json.dumps(config))
    analysis_config.setdefault("period", {})
    analysis_config["period"]["start"] = analysis_start
    analysis_config["period"]["end"] = analysis_end

    daily_long, daily_wide = load_or_build_core_daily_returns(
        analysis_config,
        root=project_root,
        cache_dir=cache_dir,
        rebuild=bool(options.rebuild_daily_cache),
    )
    if daily_wide.empty:
        raise RuntimeError("No daily frozen book returns were available after loading the core4 cache/build helpers.")

    ordered_books = [str(book["book"]).strip().lower() for book in config.get("books", [])]
    available_books = [book for book in ordered_books if book in daily_wide.columns]
    missing_books = [book for book in ordered_books if book not in daily_wide.columns]
    if missing_books:
        raise RuntimeError(
            f"Frozen country returns missing from the daily return matrix: {missing_books}. "
            f"Available={list(daily_wide.columns)}"
        )

    daily_wide = daily_wide.loc[:, available_books].copy()
    daily_wide.index = pd.to_datetime(daily_wide.index, errors="coerce").normalize()
    daily_wide = daily_wide[~daily_wide.index.isna()].sort_index()
    daily_wide = daily_wide.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    daily_long = daily_long.copy()
    daily_long["trade_date"] = pd.to_datetime(daily_long["trade_date"], errors="coerce").dt.normalize()
    daily_long = daily_long.dropna(subset=["trade_date"]).sort_values(["trade_date", "book"]).reset_index(drop=True)
    daily_long = daily_long[daily_long["book"].isin(available_books)].copy()

    monthly_long, monthly_wide, monthly_source_paths = load_frozen_monthly_returns(config, project_root=project_root)
    monthly_wide = monthly_wide.loc[:, available_books].copy()

    frozen_weights = compute_frozen_weights(monthly_wide, str(config.get("default_weight_scheme", "inverse_vol")))
    portfolio_returns = portfolio_returns_from_weights(daily_wide, frozen_weights)
    portfolio_equity = equity_from_returns(portfolio_returns, mode="compounded")

    return {
        "config": config,
        "config_path": config_path,
        "daily_cache_dir": cache_dir,
        "daily_long": daily_long,
        "country_returns": daily_wide,
        "monthly_long": monthly_long,
        "monthly_returns": monthly_wide,
        "frozen_weights": frozen_weights,
        "portfolio_returns": portfolio_returns,
        "portfolio_equity": portfolio_equity,
        "analysis_start": pd.Timestamp(analysis_start).normalize(),
        "analysis_end": pd.Timestamp(analysis_end).normalize(),
        "full_start": pd.Timestamp(full_start).normalize(),
        "full_end": pd.Timestamp(full_end).normalize(),
        "monthly_source_paths": monthly_source_paths,
    }


def load_frozen_monthly_returns(config: dict[str, Any], *, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    period = config.get("period", {})
    period_start = pd.Timestamp(str(period.get("start", "2018-01-01"))).to_period("M").to_timestamp()
    period_end = pd.Timestamp(str(period.get("end", "2025-12-31"))).to_period("M").to_timestamp()

    frames: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    for book_cfg in config.get("books", []):
        frame, source_path = _load_book_monthly_returns(book_cfg, project_root=project_root)
        frames.append(frame)
        source_paths.append(source_path)

    if not frames:
        raise RuntimeError("No monthly frozen source files were configured for the core4 reference.")

    monthly_long = pd.concat(frames, ignore_index=True, sort=False)
    monthly_long = monthly_long[
        (monthly_long["trade_month"] >= period_start) & (monthly_long["trade_month"] <= period_end)
    ].copy()
    monthly_wide = (
        monthly_long.pivot_table(index="trade_month", columns="book", values="month_return", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    if monthly_wide.empty:
        raise RuntimeError("Monthly frozen source returns are empty after filtering the configured period.")
    unique_source_paths = list(dict.fromkeys(source_paths))
    return monthly_long.sort_values(["trade_month", "book"]).reset_index(drop=True), monthly_wide, unique_source_paths


def _load_book_monthly_returns(book_cfg: dict[str, Any], *, project_root: Path) -> tuple[pd.DataFrame, Path]:
    source_dir = book_cfg.get("source_dir")
    source_file = book_cfg.get("source_file")
    if not source_dir or not source_file:
        raise ValueError(f"Missing source_dir/source_file for frozen book config: {book_cfg}")

    source_path = _resolve_path(project_root, Path(source_dir) / str(source_file))
    if not source_path.exists():
        raise FileNotFoundError(f"Frozen monthly source file not found: {source_path}")

    df = pd.read_csv(source_path)
    required = {"trade_month", "month_return"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required monthly columns {sorted(missing)} in {source_path}")

    out = df.copy()
    out["trade_month"] = normalize_month(out["trade_month"])
    out["month_return"] = pd.to_numeric(out["month_return"], errors="coerce")
    if "config_name" in out.columns and book_cfg.get("source_config_name") is not None:
        out = out[out["config_name"].astype(str).eq(str(book_cfg["source_config_name"]))].copy()
    if "book" in out.columns and book_cfg.get("source_book") is not None:
        out = out[out["book"].astype(str).eq(str(book_cfg["source_book"]))].copy()

    out["book"] = str(book_cfg["book"]).strip().lower()
    out["country"] = str(book_cfg["country"]).strip().lower()
    out["config_name"] = str(book_cfg["config_name"]).strip()
    out = out[["trade_month", "book", "country", "config_name", "month_return"]]
    out = out.dropna(subset=["trade_month", "month_return"]).sort_values("trade_month").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No monthly frozen rows survived filtering for book={book_cfg.get('book')} from {source_path}")
    return out, source_path


def normalize_month(value: Any) -> Any:
    dt = pd.to_datetime(value, errors="coerce")
    if isinstance(dt, pd.Series):
        return dt.dt.to_period("M").dt.to_timestamp()
    return dt.to_period("M").to_timestamp()


def compute_frozen_weights(monthly_returns: pd.DataFrame, scheme: str) -> pd.Series:
    clean = monthly_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weights = compute_allocation_weights(clean, str(scheme).strip().lower())
    weights = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(weights.sum()) <= 0.0:
        raise RuntimeError(f"Unable to compute frozen weights from monthly returns with scheme={scheme}")
    return (weights / float(weights.sum())).reindex(clean.columns).fillna(0.0)


def portfolio_returns_from_weights(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = pd.to_numeric(weights.reindex(returns.columns), errors="coerce").fillna(0.0)
    port = returns.mul(aligned_weights, axis=1).sum(axis=1)
    port.name = "portfolio_daily_return"
    return port.sort_index()


def compute_performance_metrics(returns: pd.Series, freq: int = TRADING_DAYS) -> dict[str, Any]:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return {}

    equity = equity_from_returns(series, mode="compounded")
    drawdown = drawdown_from_equity(equity)
    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float(equity.iloc[-1] ** (freq / len(series)) - 1.0) if float(equity.iloc[-1]) > 0 else np.nan
    vol = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    ann_vol = float(vol * math.sqrt(freq)) if np.isfinite(vol) else np.nan
    sharpe = float((series.mean() / vol) * math.sqrt(freq)) if np.isfinite(vol) and vol > 0 else np.nan
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan
    calmar = float(ann_return / abs(max_drawdown)) if np.isfinite(ann_return) and np.isfinite(max_drawdown) and max_drawdown < 0 else np.nan
    return {
        "start_date": _fmt_date(series.index.min()),
        "end_date": _fmt_date(series.index.max()),
        "n_days": int(len(series)),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_ratio_daily": float((series > 0.0).mean()),
        "best_day": float(series.max()),
        "worst_day": float(series.min()),
        "cumulative_return": total_return,
        "latest_equity": float(equity.iloc[-1]),
        "current_drawdown": float(drawdown.iloc[-1]) if not drawdown.empty else np.nan,
    }


def compute_drawdown(equity: pd.Series) -> pd.DataFrame:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    drawdown = drawdown_from_equity(eq)
    return pd.DataFrame({"date": eq.index, "equity": eq.values, "drawdown": drawdown.values})


def build_custom_periods() -> dict[str, tuple[str, str] | None]:
    return {
        "full_sample": None,
        "2018_2019": ("2018-01-01", "2019-12-31"),
        "2020": ("2020-01-01", "2020-12-31"),
        "2021": ("2021-01-01", "2021-12-31"),
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "pre_2022": ("2018-01-01", "2021-12-31"),
        "post_2022": ("2022-01-01", "2025-12-31"),
        "2023_2025": ("2023-01-01", "2025-12-31"),
    }


def compute_period_metrics(
    returns: pd.Series,
    periods: dict[str, tuple[str, str] | None],
    *,
    freq: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, list[str]]:
    series = pd.to_numeric(returns, errors="coerce").dropna().sort_index()
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    if series.empty:
        return pd.DataFrame(), list(periods.keys())

    for name, bounds in periods.items():
        if bounds is None:
            subset = series
        else:
            start, end = bounds
            mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
            subset = series.loc[mask]
            if subset.empty:
                LOGGER.info("Skipping custom period %s because no data are available in that range", name)
                skipped.append(name)
                continue
        metrics = compute_performance_metrics(subset, freq=freq)
        rows.append({"period": name, **metrics})

    return pd.DataFrame(rows), skipped


def compute_performance_by_year(returns: pd.Series, *, freq: int = TRADING_DAYS) -> pd.DataFrame:
    series = pd.to_numeric(returns, errors="coerce").dropna().sort_index()
    rows: list[dict[str, Any]] = []
    for year, group in series.groupby(series.index.year):
        rows.append({"year": int(year), **compute_performance_metrics(group, freq=freq)})
    return pd.DataFrame(rows)


def compute_leave_one_out(country_returns: pd.DataFrame, portfolio_returns: pd.Series, *, freq: int = TRADING_DAYS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frozen_metrics = compute_performance_metrics(portfolio_returns, freq=freq)
    frozen_ann = float(frozen_metrics.get("ann_return", np.nan))
    frozen_sharpe = float(frozen_metrics.get("sharpe", np.nan))
    frozen_cum = float(frozen_metrics.get("cumulative_return", np.nan))

    for country in country_returns.columns:
        remaining = [col for col in country_returns.columns if col != country]
        if not remaining:
            continue
        loo_returns = country_returns.loc[:, remaining].mean(axis=1)
        metrics = compute_performance_metrics(loo_returns, freq=freq)
        rows.append(
            {
                "portfolio": f"without_{country}",
                "excluded_country": str(country),
                "correlation_to_frozen_portfolio": float(loo_returns.corr(portfolio_returns.loc[loo_returns.index])),
                "delta_ann_return_vs_frozen": float(metrics.get("ann_return", np.nan) - frozen_ann) if np.isfinite(frozen_ann) else np.nan,
                "delta_sharpe_vs_frozen": float(metrics.get("sharpe", np.nan) - frozen_sharpe) if np.isfinite(frozen_sharpe) else np.nan,
                "delta_cumulative_return_vs_frozen": float(metrics.get("cumulative_return", np.nan) - frozen_cum)
                if np.isfinite(frozen_cum)
                else np.nan,
                **_select_metric_columns(metrics),
            }
        )
    return pd.DataFrame(rows)


def compute_country_standalone(
    country_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    *,
    frozen_weights: pd.Series | None = None,
    freq: int = TRADING_DAYS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for country in country_returns.columns:
        metrics = compute_performance_metrics(country_returns[country], freq=freq)
        rows.append(
            {
                "country": str(country),
                "correlation_to_portfolio": float(country_returns[country].corr(portfolio_returns.loc[country_returns.index])),
                "frozen_weight": float(frozen_weights.get(country, np.nan)) if frozen_weights is not None else np.nan,
                **_select_metric_columns(metrics),
            }
        )
    return pd.DataFrame(rows)


def compute_weight_sensitivity(
    country_returns: pd.DataFrame,
    *,
    frozen_weights: pd.Series | None = None,
    seed: int = 42,
    n_random: int = 100,
    freq: int = TRADING_DAYS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_row(name: str, weights: pd.Series, notes: str) -> None:
        weights = pd.to_numeric(weights.reindex(country_returns.columns), errors="coerce").fillna(0.0)
        if float(weights.sum()) <= 0.0:
            return
        weights = weights / float(weights.sum())
        port = portfolio_returns_from_weights(country_returns, weights)
        metrics = compute_performance_metrics(port, freq=freq)
        row = {"portfolio": name, **_select_metric_columns(metrics), "notes": notes}
        for book in country_returns.columns:
            row[f"weight_{book}"] = float(weights.get(book, 0.0))
        rows.append(row)

    equal_weight = pd.Series(1.0 / country_returns.shape[1], index=country_returns.columns)
    append_row("equal_weight", equal_weight, "Static equal-weight portfolio across the 4 frozen country books.")

    lookback = min(126, len(country_returns))
    if lookback > 0:
        hist = country_returns.iloc[:lookback].copy()
        inverse_vol_126d = compute_allocation_weights(hist, "inverse_vol")
        append_row(
            "inverse_vol_126d_static",
            inverse_vol_126d,
            f"Static inverse-vol weights estimated on the first {lookback} daily observations.",
        )

    if frozen_weights is not None:
        append_row(
            "frozen_reference",
            frozen_weights,
            "Static frozen reference weights derived from the configured monthly frozen source returns.",
        )
        append_row(
            "cap_25pct_renormalized",
            _one_pass_cap_and_renormalize(frozen_weights, cap=0.25),
            "One-pass clip at 25% followed by simple renormalization. The final weights can exceed 25% after renormalization.",
        )
        append_row(
            "cap_30pct_renormalized",
            _one_pass_cap_and_renormalize(frozen_weights, cap=0.30),
            "One-pass clip at 30% followed by simple renormalization.",
        )
        append_row(
            "no_country_above_30pct",
            apply_floor_cap(frozen_weights, floor=0.0, cap=0.30),
            "Strict 30% cap enforced with iterative renormalization so that no final weight exceeds 30%.",
        )

        rng = np.random.default_rng(int(seed))
        random_metrics: list[dict[str, float]] = []
        for idx in range(int(n_random)):
            noise = rng.lognormal(mean=0.0, sigma=0.15, size=len(frozen_weights))
            perturbed = pd.Series(frozen_weights.to_numpy(dtype=float) * noise, index=frozen_weights.index, dtype=float)
            perturbed = perturbed / float(perturbed.sum())
            name = f"random_{idx + 1:03d}"
            append_row(
                name,
                perturbed,
                f"Multiplicative lognormal perturbation around frozen weights | seed={seed} | sigma=0.15.",
            )
            metrics = compute_performance_metrics(portfolio_returns_from_weights(country_returns, perturbed), freq=freq)
            random_metrics.append(
                {
                    "ann_return": float(metrics.get("ann_return", np.nan)),
                    "sharpe": float(metrics.get("sharpe", np.nan)),
                    "max_drawdown": float(metrics.get("max_drawdown", np.nan)),
                }
            )

        if random_metrics:
            random_frame = pd.DataFrame(random_metrics)
            summary_notes = (
                "Random perturbation summary | "
                f"ann_return median={random_frame['ann_return'].median():.2%}, "
                f"p05={random_frame['ann_return'].quantile(0.05):.2%}, "
                f"p95={random_frame['ann_return'].quantile(0.95):.2%}; "
                f"sharpe median={random_frame['sharpe'].median():.2f}, "
                f"p05={random_frame['sharpe'].quantile(0.05):.2f}, "
                f"p95={random_frame['sharpe'].quantile(0.95):.2f}; "
                f"max_dd median={random_frame['max_drawdown'].median():.2%}."
            )
            rows.append(
                {
                    "portfolio": "random_summary",
                    "ann_return": float(random_frame["ann_return"].mean()),
                    "ann_vol": np.nan,
                    "sharpe": float(random_frame["sharpe"].mean()),
                    "max_drawdown": float(random_frame["max_drawdown"].mean()),
                    "calmar": np.nan,
                    "cumulative_return": np.nan,
                    "notes": summary_notes,
                }
            )

    return pd.DataFrame(rows)


def compute_cost_stress(
    portfolio_returns: pd.Series,
    *,
    turnover: pd.Series | None = None,
    freq: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, str | None]:
    scenarios = [("base", 0)]
    scenarios.extend([(f"cost_{bps}bps", bps) for bps in (1, 2, 5, 10, 20)])
    rows: list[dict[str, Any]] = []
    warning: str | None = None

    if turnover is not None and not turnover.empty:
        aligned_turnover = pd.to_numeric(turnover.reindex(portfolio_returns.index), errors="coerce").fillna(0.0)
        available_inputs = "portfolio_daily_returns, turnover_daily"
        for name, bps in scenarios:
            if bps == 0:
                stressed = portfolio_returns.copy()
                assumption = "Base scenario without additional execution-cost stress."
            else:
                stressed = portfolio_returns - aligned_turnover * (float(bps) / 10000.0)
                assumption = f"Daily execution-cost proxy computed as turnover_daily * {bps}bps."
            metrics = compute_performance_metrics(stressed, freq=freq)
            rows.append({"scenario": name, "assumption": assumption, "available_inputs": available_inputs, **_select_metric_columns(metrics)})
    else:
        warning = (
            "Cost stress uses a constant annual drag proxy because no portfolio-wide turnover series was identified for the frozen reference."
        )
        available_inputs = "portfolio_daily_returns only"
        for name, bps in scenarios:
            if bps == 0:
                stressed = portfolio_returns.copy()
                assumption = "Base scenario without additional execution-cost stress."
            else:
                daily_drag = (float(bps) / 10000.0) / float(freq)
                stressed = portfolio_returns - daily_drag
                assumption = (
                    f"Constant annual drag proxy of {bps}bps/year spread evenly across {freq} trading days; "
                    "used because turnover input was unavailable."
                )
            metrics = compute_performance_metrics(stressed, freq=freq)
            rows.append({"scenario": name, "assumption": assumption, "available_inputs": available_inputs, **_select_metric_columns(metrics)})

    return pd.DataFrame(rows), warning


def discover_trade_input_candidates(config: dict[str, Any], *, project_root: Path) -> list[dict[str, Any]]:
    search_dirs: list[Path] = []
    for book_cfg in config.get("books", []):
        source_dir = book_cfg.get("source_dir")
        if source_dir:
            search_dirs.append(_resolve_path(project_root, Path(str(source_dir))))
    for value in config.get("decision_sources", {}).values():
        search_dirs.append(_resolve_path(project_root, Path(str(value))))

    deduped_dirs: list[Path] = []
    seen: set[str] = set()
    for path in search_dirs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped_dirs.append(path)

    pattern = re.compile(r"(trade|trades|order|execution|fill)", flags=re.IGNORECASE)
    column_tokens = {"nb_trades", "engine_nb_trades", "total_pnl", "avg_pnl_per_trade", "avg_holding_days", "win_rate"}
    candidates: list[dict[str, Any]] = []
    for base in deduped_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.csv"):
            try:
                sample = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            columns = [str(col) for col in sample.columns]
            normalized_columns = {str(col).strip().lower() for col in columns}
            if not pattern.search(path.name) and not normalized_columns.intersection(column_tokens):
                continue
            candidates.append(
                {
                    "path": path,
                    "columns": columns,
                    "n_sample_rows": int(len(sample)),
                }
            )
    return candidates


def compute_trade_concentration(trade_candidates: list[dict[str, Any]]) -> tuple[pd.DataFrame, str | None]:
    trade_ledger = _try_load_trade_ledger(trade_candidates)
    if trade_ledger is None:
        available_inputs = ", ".join(str(candidate["path"].name) for candidate in trade_candidates[:5]) if trade_candidates else "none"
        warning = (
            "Trade-level concentration was not computed because no portfolio-wide trade ledger with per-trade PnL was identified. "
            "Only aggregate or country-specific trade summaries were found."
        )
        return (
            pd.DataFrame(
                [
                    {
                        "status": "not_available",
                        "reason": "No portfolio-wide trade ledger with per-trade PnL was identified.",
                        "available_inputs": available_inputs,
                    }
                ]
            ),
            warning,
        )

    pnl_column = trade_ledger.attrs["pnl_column"]
    pnl = pd.to_numeric(trade_ledger[pnl_column], errors="coerce").dropna().sort_values()
    total_pnl = float(pnl.sum()) if not pnl.empty else np.nan
    holding = _extract_holding_days(trade_ledger)
    row = {
        "status": "ok",
        "available_inputs": str(trade_ledger.attrs.get("source_path", "")),
        "n_trades": int(len(pnl)),
        "total_pnl": total_pnl,
        "mean_trade_pnl": float(pnl.mean()) if not pnl.empty else np.nan,
        "median_trade_pnl": float(pnl.median()) if not pnl.empty else np.nan,
        "hit_ratio_trade": float((pnl > 0.0).mean()) if not pnl.empty else np.nan,
        "top_1pct_pnl_share": _tail_share(pnl, pct=0.01, side="top", total=total_pnl),
        "top_5pct_pnl_share": _tail_share(pnl, pct=0.05, side="top", total=total_pnl),
        "top_10pct_pnl_share": _tail_share(pnl, pct=0.10, side="top", total=total_pnl),
        "worst_1pct_pnl_share": _tail_share(pnl, pct=0.01, side="bottom", total=total_pnl),
        "worst_5pct_pnl_share": _tail_share(pnl, pct=0.05, side="bottom", total=total_pnl),
        "largest_winner": float(pnl.max()) if not pnl.empty else np.nan,
        "largest_loser": float(pnl.min()) if not pnl.empty else np.nan,
        "avg_holding_days": float(holding.mean()) if holding is not None and not holding.empty else np.nan,
        "median_holding_days": float(holding.median()) if holding is not None and not holding.empty else np.nan,
    }
    return pd.DataFrame([row]), None


def _try_load_trade_ledger(trade_candidates: list[dict[str, Any]]) -> pd.DataFrame | None:
    preferred_pnl_cols = ("trade_pnl", "pnl", "net_pnl", "realized_pnl", "total_pnl")
    row_level_markers = {"trade_id", "entry_date", "exit_date", "open_date", "close_date"}

    for candidate in trade_candidates:
        columns = {str(col).strip().lower() for col in candidate["columns"]}
        if not any(col in columns for col in preferred_pnl_cols):
            continue
        if not any(col in columns for col in row_level_markers):
            continue
        path = Path(candidate["path"])
        try:
            ledger = pd.read_csv(path)
        except Exception:
            continue
        normalized = {str(col).strip().lower(): col for col in ledger.columns}
        pnl_key = next((col for col in preferred_pnl_cols if col in normalized), None)
        if pnl_key is None:
            continue
        pnl_column = normalized[pnl_key]
        ledger.attrs["pnl_column"] = pnl_column
        ledger.attrs["source_path"] = str(path)
        return ledger
    return None


def compute_rolling_metrics(returns: pd.Series, windows: tuple[int, ...] = (63, 126, 252), *, freq: int = TRADING_DAYS) -> pd.DataFrame:
    series = pd.to_numeric(returns, errors="coerce").dropna().sort_index()
    frames: list[pd.DataFrame] = []
    if series.empty:
        return pd.DataFrame()

    for window in windows:
        ann_return = series.rolling(window, min_periods=window).apply(
            lambda values: _annualized_return_from_window(values, freq=freq),
            raw=True,
        )
        ann_vol = series.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(freq)
        sharpe = (series.rolling(window, min_periods=window).mean() / series.rolling(window, min_periods=window).std(ddof=1)) * math.sqrt(freq)
        max_drawdown = _rolling_max_drawdown_from_returns(series, window=window)
        hit_ratio = (series > 0.0).astype(float).rolling(window, min_periods=window).mean()
        frame = pd.DataFrame(
            {
                "date": series.index,
                "window": int(window),
                "rolling_ann_return": ann_return.values,
                "rolling_ann_vol": ann_vol.values,
                "rolling_sharpe": sharpe.values,
                "rolling_max_drawdown": max_drawdown.values,
                "rolling_hit_ratio": hit_ratio.values,
            }
        )
        frames.append(frame.dropna(how="all", subset=["rolling_ann_return", "rolling_ann_vol", "rolling_sharpe", "rolling_max_drawdown", "rolling_hit_ratio"]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def compute_drawdown_events(equity: pd.Series) -> pd.DataFrame:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return pd.DataFrame(columns=["rank", "peak_date", "trough_date", "recovery_date", "drawdown", "duration_days", "recovery_days", "recovered_bool"])

    events: list[dict[str, Any]] = []
    peak_date = pd.Timestamp(eq.index[0])
    peak_value = float(eq.iloc[0])
    event_peak_date = peak_date
    trough_date = peak_date
    trough_drawdown = 0.0
    in_event = False
    tolerance = 1e-12

    for date, value in eq.items():
        current_date = pd.Timestamp(date)
        current_value = float(value)
        if current_value >= peak_value - tolerance:
            if in_event:
                events.append(
                    {
                        "peak_date": event_peak_date,
                        "trough_date": trough_date,
                        "recovery_date": current_date,
                        "drawdown": float(trough_drawdown),
                        "duration_days": int((trough_date - event_peak_date).days),
                        "recovery_days": int((current_date - trough_date).days),
                        "recovered_bool": True,
                    }
                )
                in_event = False
            peak_value = current_value
            event_peak_date = current_date
            trough_date = current_date
            trough_drawdown = 0.0
            continue

        current_drawdown = current_value / peak_value - 1.0
        if not in_event:
            in_event = True
            trough_date = current_date
            trough_drawdown = current_drawdown
        elif current_drawdown < trough_drawdown:
            trough_date = current_date
            trough_drawdown = current_drawdown

    if in_event:
        events.append(
            {
                "peak_date": event_peak_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "drawdown": float(trough_drawdown),
                "duration_days": int((trough_date - event_peak_date).days),
                "recovery_days": np.nan,
                "recovered_bool": False,
            }
        )

    if not events:
        return pd.DataFrame(columns=["rank", "peak_date", "trough_date", "recovery_date", "drawdown", "duration_days", "recovery_days", "recovered_bool"])

    out = pd.DataFrame(events).sort_values(["drawdown", "peak_date"], ascending=[True, True]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["peak_date"] = pd.to_datetime(out["peak_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["trough_date"] = pd.to_datetime(out["trough_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["recovery_date"] = pd.to_datetime(out["recovery_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def compute_monthly_returns(portfolio_returns: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = pd.to_numeric(portfolio_returns, errors="coerce").dropna().sort_index()
    monthly = series.groupby(series.index.to_period("M")).apply(lambda values: float((1.0 + values).prod() - 1.0))
    monthly.index = monthly.index.rename("trade_month")
    frame = monthly.reset_index(name="monthly_return")
    frame["trade_month"] = frame["trade_month"].astype("period[M]").dt.to_timestamp()
    frame["year"] = frame["trade_month"].dt.year.astype(int)
    frame["month"] = frame["trade_month"].dt.month.astype(int)
    frame["month_label"] = frame["month"].map(lambda value: f"{int(value):02d}")
    long = frame.loc[:, ["year", "month", "monthly_return"]].copy()
    pivot = frame.pivot(index="year", columns="month_label", values="monthly_return").sort_index()
    return long, pivot


def compute_country_contribution(
    country_returns: pd.DataFrame,
    frozen_weights: pd.Series,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    weighted = country_returns.mul(frozen_weights.reindex(country_returns.columns).fillna(0.0), axis=1)
    compounded = ((1.0 + weighted).prod() - 1.0).rename("weighted_compounded_return")
    additive = weighted.sum().rename("sum_weighted_daily_returns")
    contribution = pd.concat([frozen_weights.rename("weight"), compounded, additive], axis=1).reset_index()
    contribution = contribution.rename(columns={contribution.columns[0]: "country"})
    contribution["correlation_to_portfolio"] = contribution["country"].map(
        lambda country: float(country_returns[str(country)].corr(portfolio_returns.loc[country_returns.index]))
    )
    return contribution.sort_values("weighted_compounded_return", ascending=False).reset_index(drop=True)


def compute_daily_country_contribution(
    country_returns: pd.DataFrame,
    weights: pd.Series,
    *,
    portfolio_returns: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_returns = country_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0).sort_index()
    aligned_weights = pd.to_numeric(weights.reindex(clean_returns.columns), errors="coerce").fillna(0.0)
    weighted = clean_returns.mul(aligned_weights, axis=1)
    portfolio = (
        pd.to_numeric(portfolio_returns.reindex(clean_returns.index), errors="coerce").fillna(weighted.sum(axis=1))
        if portfolio_returns is not None
        else weighted.sum(axis=1)
    )
    portfolio = portfolio.sort_index()

    long = pd.concat(
        [
            weighted.stack(dropna=False).rename("weighted_return_contribution"),
            clean_returns.stack(dropna=False).rename("country_return"),
        ],
        axis=1,
    ).reset_index()
    long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "country"})
    long["country_weight"] = long["country"].map(aligned_weights.to_dict())
    long["portfolio_return"] = long["date"].map(portfolio.to_dict())
    long["contribution_share"] = np.where(
        long["portfolio_return"].abs() > 1e-12,
        long["weighted_return_contribution"] / long["portfolio_return"],
        np.nan,
    )
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    long = long.loc[
        :,
        [
            "date",
            "country",
            "country_return",
            "country_weight",
            "weighted_return_contribution",
            "portfolio_return",
            "contribution_share",
        ],
    ].sort_values(["date", "country"]).reset_index(drop=True)

    wide = weighted.copy()
    wide.index = pd.to_datetime(wide.index, errors="coerce")
    wide.index.name = "date"
    wide["portfolio_return"] = portfolio
    wide = wide.reset_index()
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return long, wide


def compute_annual_country_contribution(daily_contrib: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year",
        "country",
        "annual_country_return",
        "annual_weighted_contribution",
        "portfolio_annual_return",
        "contribution_rank",
        "contribution_pct_of_total_positive_years",
        "contribution_pct_of_total_absolute",
    ]
    if daily_contrib.empty:
        return pd.DataFrame(columns=columns)

    frame = daily_contrib.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["date", "country"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["year"] = frame["date"].dt.year.astype(int)

    country = (
        frame.groupby(["year", "country"], dropna=False)
        .agg(
            annual_country_return=("country_return", lambda values: float((1.0 + pd.Series(values).fillna(0.0)).prod() - 1.0)),
            annual_weighted_contribution=("weighted_return_contribution", "sum"),
        )
        .reset_index()
    )
    portfolio_daily = frame.loc[:, ["date", "year", "portfolio_return"]].drop_duplicates(subset=["date"]).copy()
    portfolio_year = (
        portfolio_daily.groupby("year", dropna=False)
        .agg(
            portfolio_annual_return=("portfolio_return", lambda values: float((1.0 + pd.Series(values).fillna(0.0)).prod() - 1.0)),
            portfolio_additive_return_proxy=("portfolio_return", "sum"),
        )
        .reset_index()
    )
    out = country.merge(portfolio_year, on="year", how="left")
    out["contribution_rank"] = (
        out.groupby("year")["annual_weighted_contribution"].rank(method="first", ascending=False).astype(int)
    )
    positive_denom = out.groupby("year")["annual_weighted_contribution"].transform(
        lambda values: float(pd.Series(values).clip(lower=0.0).sum())
    )
    out["contribution_pct_of_total_positive_years"] = np.where(
        (pd.to_numeric(out["portfolio_annual_return"], errors="coerce") > 0.0) & (positive_denom > 1e-12),
        out["annual_weighted_contribution"] / positive_denom,
        np.nan,
    )
    abs_total = out.groupby("year")["annual_weighted_contribution"].transform(lambda values: float(pd.Series(values).abs().sum()))
    out["contribution_pct_of_total_absolute"] = np.where(
        abs_total.abs() > 1e-12,
        out["annual_weighted_contribution"].abs() / abs_total,
        np.nan,
    )
    out = out.drop(columns=["portfolio_additive_return_proxy"])
    return out.loc[:, columns].sort_values(["year", "contribution_rank", "country"]).reset_index(drop=True)


def compute_annual_country_metrics(
    country_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    *,
    freq: int = TRADING_DAYS,
) -> pd.DataFrame:
    columns = [
        "year",
        "country",
        "n_days",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "hit_ratio_daily",
        "cumulative_return",
        "correlation_to_portfolio",
    ]
    if country_returns.empty:
        return pd.DataFrame(columns=columns)

    aligned_country = country_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0).sort_index()
    aligned_portfolio = pd.to_numeric(portfolio_returns.reindex(aligned_country.index), errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for year in sorted(pd.Index(aligned_country.index.year).unique()):
        mask = aligned_country.index.year == int(year)
        year_portfolio = aligned_portfolio.loc[mask]
        for country in aligned_country.columns:
            year_country = aligned_country.loc[mask, country]
            metrics = compute_performance_metrics(year_country, freq=freq)
            rows.append(
                {
                    "year": int(year),
                    "country": str(country),
                    "n_days": int(metrics.get("n_days", 0) or 0),
                    "ann_return": metrics.get("ann_return"),
                    "ann_vol": metrics.get("ann_vol"),
                    "sharpe": metrics.get("sharpe"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "hit_ratio_daily": metrics.get("hit_ratio_daily"),
                    "cumulative_return": metrics.get("cumulative_return"),
                    "correlation_to_portfolio": float(year_country.corr(year_portfolio)) if len(year_country) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def compute_leave_one_out_by_year(
    country_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    *,
    freq: int = TRADING_DAYS,
) -> pd.DataFrame:
    columns = [
        "year",
        "portfolio_variant",
        "excluded_country",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "cumulative_return",
        "correlation_to_frozen_portfolio",
    ]
    if country_returns.empty:
        return pd.DataFrame(columns=columns)

    aligned_country = country_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0).sort_index()
    aligned_portfolio = pd.to_numeric(portfolio_returns.reindex(aligned_country.index), errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for year in sorted(pd.Index(aligned_country.index.year).unique()):
        mask = aligned_country.index.year == int(year)
        year_portfolio = aligned_portfolio.loc[mask]
        for country in aligned_country.columns:
            remaining = [col for col in aligned_country.columns if col != country]
            if not remaining:
                continue
            loo = aligned_country.loc[mask, remaining].mean(axis=1)
            metrics = compute_performance_metrics(loo, freq=freq)
            rows.append(
                {
                    "year": int(year),
                    "portfolio_variant": f"without_{country}",
                    "excluded_country": str(country),
                    "ann_return": metrics.get("ann_return"),
                    "ann_vol": metrics.get("ann_vol"),
                    "sharpe": metrics.get("sharpe"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "calmar": metrics.get("calmar"),
                    "cumulative_return": metrics.get("cumulative_return"),
                    "correlation_to_frozen_portfolio": float(loo.corr(year_portfolio)) if len(loo) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["year", "portfolio_variant"]).reset_index(drop=True)


def compute_drawdown_attribution(drawdown_events: pd.DataFrame, daily_country_contribution: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "drawdown_rank",
        "peak_date",
        "trough_date",
        "recovery_date",
        "drawdown",
        "country",
        "country_cumulative_return_window",
        "weighted_contribution_window",
        "contribution_rank",
        "attribution_residual",
    ]
    if drawdown_events.empty or daily_country_contribution.empty:
        return pd.DataFrame(columns=columns)

    contrib = daily_country_contribution.copy()
    contrib["date"] = pd.to_datetime(contrib["date"], errors="coerce")
    contrib = contrib.dropna(subset=["date"]).sort_values(["date", "country"]).reset_index(drop=True)
    if contrib.empty:
        return pd.DataFrame(columns=columns)

    portfolio_daily = contrib.loc[:, ["date", "portfolio_return"]].drop_duplicates(subset=["date"]).set_index("date")["portfolio_return"]
    rows: list[dict[str, Any]] = []
    for row in drawdown_events.itertuples(index=False):
        peak_date = pd.Timestamp(row.peak_date)
        trough_date = pd.Timestamp(row.trough_date)
        recovery_date = pd.Timestamp(row.recovery_date) if pd.notna(row.recovery_date) and str(row.recovery_date) != "NaT" else pd.NaT
        window = contrib[(contrib["date"] >= peak_date) & (contrib["date"] <= trough_date)].copy()
        if window.empty:
            continue
        grouped = (
            window.groupby("country", dropna=False)
            .agg(
                country_cumulative_return_window=("country_return", lambda values: float((1.0 + pd.Series(values).fillna(0.0)).prod() - 1.0)),
                weighted_contribution_window=("weighted_return_contribution", "sum"),
            )
            .reset_index()
        )
        window_portfolio = portfolio_daily.loc[(portfolio_daily.index >= peak_date) & (portfolio_daily.index <= trough_date)]
        compounded_portfolio = float((1.0 + window_portfolio.fillna(0.0)).prod() - 1.0) if not window_portfolio.empty else np.nan
        residual = (
            float(compounded_portfolio - grouped["weighted_contribution_window"].sum())
            if np.isfinite(compounded_portfolio)
            else np.nan
        )
        grouped["contribution_rank"] = grouped["weighted_contribution_window"].rank(method="first", ascending=True).astype(int)
        grouped["drawdown_rank"] = int(row.rank)
        grouped["peak_date"] = peak_date.strftime("%Y-%m-%d")
        grouped["trough_date"] = trough_date.strftime("%Y-%m-%d")
        grouped["recovery_date"] = recovery_date.strftime("%Y-%m-%d") if pd.notna(recovery_date) else ""
        grouped["drawdown"] = float(row.drawdown)
        grouped["attribution_residual"] = residual
        rows.extend(grouped.loc[:, columns].to_dict(orient="records"))
    return pd.DataFrame(rows, columns=columns).sort_values(["drawdown_rank", "contribution_rank", "country"]).reset_index(drop=True)


def compute_worst_periods_attribution(
    portfolio_returns: pd.Series,
    daily_country_contribution: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (21, 63, 126),
    top_n: int = 10,
) -> pd.DataFrame:
    columns = [
        "window_days",
        "rank",
        "start_date",
        "end_date",
        "portfolio_return_window",
        "max_drawdown_inside_window",
        "country",
        "country_return_window",
        "weighted_contribution_window",
        "contribution_rank",
    ]
    if daily_country_contribution.empty:
        return pd.DataFrame(columns=columns)

    portfolio = pd.to_numeric(portfolio_returns, errors="coerce").dropna().sort_index()
    contrib = daily_country_contribution.copy()
    contrib["date"] = pd.to_datetime(contrib["date"], errors="coerce")
    contrib = contrib.dropna(subset=["date"]).sort_values(["date", "country"]).reset_index(drop=True)
    if portfolio.empty or contrib.empty:
        return pd.DataFrame(columns=columns)

    weighted_wide = (
        contrib.pivot_table(index="date", columns="country", values="weighted_return_contribution", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    return_wide = (
        contrib.pivot_table(index="date", columns="country", values="country_return", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        window_meta: list[dict[str, Any]] = []
        for idx in range(int(window) - 1, len(portfolio)):
            sample = portfolio.iloc[idx - int(window) + 1 : idx + 1]
            start_date = pd.Timestamp(sample.index[0])
            end_date = pd.Timestamp(sample.index[-1])
            metrics = compute_performance_metrics(sample)
            window_meta.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "portfolio_return_window": float((1.0 + sample.fillna(0.0)).prod() - 1.0),
                    "max_drawdown_inside_window": metrics.get("max_drawdown"),
                }
            )
        if not window_meta:
            continue
        worst = pd.DataFrame(window_meta).nsmallest(int(top_n), "portfolio_return_window").reset_index(drop=True)
        for rank, meta in enumerate(worst.itertuples(index=False), start=1):
            weighted_window = weighted_wide.loc[(weighted_wide.index >= meta.start_date) & (weighted_wide.index <= meta.end_date)]
            country_window = return_wide.loc[(return_wide.index >= meta.start_date) & (return_wide.index <= meta.end_date)]
            if weighted_window.empty or country_window.empty:
                continue
            country_summary = pd.DataFrame(
                {
                    "country": weighted_window.columns,
                    "country_return_window": [
                        float((1.0 + country_window[col].fillna(0.0)).prod() - 1.0) for col in weighted_window.columns
                    ],
                    "weighted_contribution_window": [float(weighted_window[col].sum()) for col in weighted_window.columns],
                }
            )
            country_summary["contribution_rank"] = (
                country_summary["weighted_contribution_window"].rank(method="first", ascending=True).astype(int)
            )
            country_summary["window_days"] = int(window)
            country_summary["rank"] = int(rank)
            country_summary["start_date"] = meta.start_date.strftime("%Y-%m-%d")
            country_summary["end_date"] = meta.end_date.strftime("%Y-%m-%d")
            country_summary["portfolio_return_window"] = float(meta.portfolio_return_window)
            country_summary["max_drawdown_inside_window"] = meta.max_drawdown_inside_window
            rows.extend(country_summary.loc[:, columns].to_dict(orient="records"))
    return pd.DataFrame(rows, columns=columns).sort_values(["window_days", "rank", "contribution_rank"]).reset_index(drop=True)


def compute_daily_exposure(daily_long: pd.DataFrame, frozen_weights: pd.Series) -> pd.DataFrame:
    columns = [
        "date",
        "country",
        "gross_exposure",
        "net_exposure",
        "long_exposure",
        "short_exposure",
        "number_of_positions",
        "number_of_pairs",
        "country_weight",
        "capacity_utilization",
        "max_pairs_observed",
        "exposure_source",
        "exposure_quality",
    ]
    if daily_long.empty or "trade_date" not in daily_long.columns or "book" not in daily_long.columns:
        return pd.DataFrame(
            [
                {
                    "date": "",
                    "country": "not_available",
                    "gross_exposure": np.nan,
                    "net_exposure": np.nan,
                    "long_exposure": np.nan,
                    "short_exposure": np.nan,
                    "number_of_positions": np.nan,
                    "number_of_pairs": np.nan,
                    "country_weight": np.nan,
                    "capacity_utilization": np.nan,
                    "max_pairs_observed": np.nan,
                    "exposure_source": "not_available",
                    "exposure_quality": "not_available",
                }
            ],
            columns=columns,
        )

    frame = daily_long.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["book"] = frame["book"].astype(str).str.strip().str.lower()
    open_positions = frame["n_open_positions"] if "n_open_positions" in frame.columns else pd.Series(0.0, index=frame.index, dtype=float)
    frame["n_open_positions"] = pd.to_numeric(open_positions, errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["trade_date"]).sort_values(["trade_date", "book"]).reset_index(drop=True)
    countries = [str(col) for col in frozen_weights.index]
    counts = (
        frame.pivot_table(index="trade_date", columns="book", values="n_open_positions", aggfunc="max")
        .reindex(columns=countries)
        .sort_index()
        .fillna(0.0)
    )
    if counts.empty:
        return pd.DataFrame(
            [
                {
                    "date": "",
                    "country": "not_available",
                    "gross_exposure": np.nan,
                    "net_exposure": np.nan,
                    "long_exposure": np.nan,
                    "short_exposure": np.nan,
                    "number_of_positions": np.nan,
                    "number_of_pairs": np.nan,
                    "country_weight": np.nan,
                    "capacity_utilization": np.nan,
                    "max_pairs_observed": np.nan,
                    "exposure_source": "daily_long missing n_open_positions",
                    "exposure_quality": "not_available",
                }
            ],
            columns=columns,
        )

    weights = pd.to_numeric(frozen_weights.reindex(counts.columns), errors="coerce").fillna(0.0)
    max_pairs = counts.max().replace(0.0, 1.0)
    capacity = counts.div(max_pairs, axis=1).clip(lower=0.0)
    gross = capacity.mul(weights, axis=1)
    long_exp = gross / 2.0
    short_exp = gross / 2.0
    net_exp = gross * 0.0
    positions = counts * 2.0

    long_rows: list[dict[str, Any]] = []
    source_note = "Proxy from frozen country weights scaled by n_open_positions / max_observed_pairs_per_country."
    for country in counts.columns:
        country_frame = pd.DataFrame(
            {
                "date": counts.index.strftime("%Y-%m-%d"),
                "country": str(country),
                "gross_exposure": gross[country].to_numpy(dtype=float),
                "net_exposure": net_exp[country].to_numpy(dtype=float),
                "long_exposure": long_exp[country].to_numpy(dtype=float),
                "short_exposure": short_exp[country].to_numpy(dtype=float),
                "number_of_positions": positions[country].to_numpy(dtype=float),
                "number_of_pairs": counts[country].to_numpy(dtype=float),
                "country_weight": float(weights.get(country, np.nan)),
                "capacity_utilization": capacity[country].to_numpy(dtype=float),
                "max_pairs_observed": float(max_pairs.get(country, np.nan)),
                "exposure_source": source_note,
                "exposure_quality": "n_open_positions_proxy",
            }
        )
        long_rows.extend(country_frame.to_dict(orient="records"))

    exposure = pd.DataFrame(long_rows, columns=columns)
    portfolio = (
        exposure.groupby("date", dropna=False)
        .agg(
            gross_exposure=("gross_exposure", "sum"),
            net_exposure=("net_exposure", "sum"),
            long_exposure=("long_exposure", "sum"),
            short_exposure=("short_exposure", "sum"),
            number_of_positions=("number_of_positions", "sum"),
            number_of_pairs=("number_of_pairs", "sum"),
        )
        .reset_index()
    )
    portfolio["country"] = "portfolio"
    portfolio["country_weight"] = float(weights.sum())
    portfolio["capacity_utilization"] = np.where(
        portfolio["country_weight"].abs() > 1e-12,
        portfolio["gross_exposure"] / portfolio["country_weight"],
        np.nan,
    )
    portfolio["max_pairs_observed"] = float(max_pairs.sum())
    portfolio["exposure_source"] = source_note
    portfolio["exposure_quality"] = "n_open_positions_proxy"
    exposure = pd.concat([exposure, portfolio.loc[:, columns]], ignore_index=True, sort=False)
    return exposure.sort_values(["date", "country"]).reset_index(drop=True)


def compute_daily_turnover_from_positions(daily_exposure: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "country",
        "gross_turnover",
        "net_turnover",
        "long_turnover",
        "short_turnover",
        "turnover_source",
        "turnover_quality",
        "assumption",
    ]
    if daily_exposure.empty or "gross_exposure" not in daily_exposure.columns or "country" not in daily_exposure.columns:
        return pd.DataFrame(
            [
                {
                    "date": "",
                    "country": "not_available",
                    "gross_turnover": np.nan,
                    "net_turnover": np.nan,
                    "long_turnover": np.nan,
                    "short_turnover": np.nan,
                    "turnover_source": "not_available",
                    "turnover_quality": "not_available",
                    "assumption": "Daily exposure inputs were not available.",
                }
            ],
            columns=columns,
        )

    exposure = daily_exposure.copy()
    exposure = exposure[exposure["country"].astype(str).ne("portfolio")].copy()
    exposure["date"] = pd.to_datetime(exposure["date"], errors="coerce")
    exposure = exposure.dropna(subset=["date"]).sort_values(["country", "date"]).reset_index(drop=True)
    if exposure.empty:
        return pd.DataFrame(
            [
                {
                    "date": "",
                    "country": "not_available",
                    "gross_turnover": np.nan,
                    "net_turnover": np.nan,
                    "long_turnover": np.nan,
                    "short_turnover": np.nan,
                    "turnover_source": "not_available",
                    "turnover_quality": "not_available",
                    "assumption": "Country-level exposure rows were not available.",
                }
            ],
            columns=columns,
        )

    assumption = (
        "Turnover proxy computed as abs(exposure_t - exposure_t-1) on exposure reconstructed from "
        "frozen_weight * n_open_positions / max_observed_pairs."
    )
    rows: list[dict[str, Any]] = []
    for country, group in exposure.groupby("country", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        gross_turnover = pd.to_numeric(group["gross_exposure"], errors="coerce").fillna(0.0).diff().fillna(group["gross_exposure"]).abs()
        net_turnover = pd.to_numeric(group["net_exposure"], errors="coerce").fillna(0.0).diff().fillna(group["net_exposure"]).abs()
        long_turnover = pd.to_numeric(group["long_exposure"], errors="coerce").fillna(0.0).diff().fillna(group["long_exposure"]).abs()
        short_turnover = pd.to_numeric(group["short_exposure"], errors="coerce").fillna(0.0).diff().fillna(group["short_exposure"]).abs()
        source = str(group["exposure_source"].iloc[0]) if "exposure_source" in group.columns else "daily_exposure proxy"
        rows.extend(
            pd.DataFrame(
                {
                    "date": group["date"].dt.strftime("%Y-%m-%d"),
                    "country": str(country),
                    "gross_turnover": gross_turnover.to_numpy(dtype=float),
                    "net_turnover": net_turnover.to_numpy(dtype=float),
                    "long_turnover": long_turnover.to_numpy(dtype=float),
                    "short_turnover": short_turnover.to_numpy(dtype=float),
                    "turnover_source": source,
                    "turnover_quality": "rebalance_proxy",
                    "assumption": assumption,
                }
            ).to_dict(orient="records")
        )

    turnover = pd.DataFrame(rows, columns=columns)
    portfolio = (
        turnover.groupby("date", dropna=False)
        .agg(
            gross_turnover=("gross_turnover", "sum"),
            net_turnover=("net_turnover", "sum"),
            long_turnover=("long_turnover", "sum"),
            short_turnover=("short_turnover", "sum"),
        )
        .reset_index()
    )
    portfolio["country"] = "portfolio"
    portfolio["turnover_source"] = assumption
    portfolio["turnover_quality"] = "rebalance_proxy"
    portfolio["assumption"] = assumption
    turnover = pd.concat([turnover, portfolio.loc[:, columns]], ignore_index=True, sort=False)
    return turnover.sort_values(["date", "country"]).reset_index(drop=True)


def extract_portfolio_turnover_series(daily_turnover: pd.DataFrame) -> pd.Series:
    if daily_turnover.empty or "gross_turnover" not in daily_turnover.columns:
        return pd.Series(dtype=float, name="gross_turnover")
    frame = daily_turnover.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return pd.Series(dtype=float, name="gross_turnover")
    if "country" in frame.columns and frame["country"].astype(str).eq("portfolio").any():
        out = frame[frame["country"].astype(str).eq("portfolio")].copy()
        series = pd.to_numeric(out["gross_turnover"], errors="coerce").fillna(0.0)
        series.index = pd.to_datetime(out["date"], errors="coerce")
        series.name = "gross_turnover"
        return series.sort_index()
    grouped = frame.groupby("date", dropna=False)["gross_turnover"].sum().sort_index()
    grouped.name = "gross_turnover"
    return pd.to_numeric(grouped, errors="coerce").fillna(0.0)


def summarize_turnover_input(daily_turnover: pd.DataFrame, *, freq: int = TRADING_DAYS) -> dict[str, Any]:
    if daily_turnover.empty or "gross_turnover" not in daily_turnover.columns:
        return {
            "available": False,
            "quality": "not_available",
            "source": "not_available",
            "avg_daily_turnover": np.nan,
            "ann_turnover": np.nan,
            "warning": "Daily turnover could not be reconstructed because no usable exposure or position path was identified.",
        }
    frame = daily_turnover.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "country" in frame.columns and frame["country"].astype(str).eq("portfolio").any():
        frame = frame[frame["country"].astype(str).eq("portfolio")].copy()
    if frame.empty or frame["turnover_quality"].astype(str).eq("not_available").all():
        return {
            "available": False,
            "quality": "not_available",
            "source": "not_available",
            "avg_daily_turnover": np.nan,
            "ann_turnover": np.nan,
            "warning": "Daily turnover export exists, but it does not contain an exploitable portfolio turnover series.",
        }
    series = pd.to_numeric(frame["gross_turnover"], errors="coerce").fillna(0.0)
    quality = str(frame["turnover_quality"].iloc[0])
    source = str(frame["turnover_source"].iloc[0])
    warning = None
    if quality != "actual_positions":
        warning = (
            f"Turnover is a `{quality}` diagnostic, not true order-level turnover. "
            "It is derived from the best available frozen reference proxy."
        )
    return {
        "available": True,
        "quality": quality,
        "source": source,
        "avg_daily_turnover": float(series.mean()),
        "ann_turnover": float(series.mean() * float(freq)),
        "warning": warning,
    }


def compute_turnover_based_cost_stress(
    portfolio_returns: pd.Series,
    daily_turnover: pd.DataFrame,
    *,
    freq: int = TRADING_DAYS,
) -> pd.DataFrame:
    columns = [
        "scenario",
        "cost_bps",
        "turnover_source",
        "turnover_quality",
        "avg_daily_turnover",
        "ann_turnover",
        "total_cost_drag",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "cumulative_return",
        "assumption",
    ]
    info = summarize_turnover_input(daily_turnover, freq=freq)
    turnover = extract_portfolio_turnover_series(daily_turnover)
    if turnover.empty or not bool(info.get("available")):
        return pd.DataFrame(
            [
                {
                    "scenario": "not_available",
                    "cost_bps": np.nan,
                    "turnover_source": info.get("source", "not_available"),
                    "turnover_quality": info.get("quality", "not_available"),
                    "avg_daily_turnover": info.get("avg_daily_turnover"),
                    "ann_turnover": info.get("ann_turnover"),
                    "total_cost_drag": np.nan,
                    "ann_return": np.nan,
                    "ann_vol": np.nan,
                    "sharpe": np.nan,
                    "max_drawdown": np.nan,
                    "calmar": np.nan,
                    "cumulative_return": np.nan,
                    "assumption": str(info.get("warning") or "Turnover input was not available."),
                }
            ],
            columns=columns,
        )

    turnover = pd.to_numeric(turnover.reindex(portfolio_returns.index), errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for bps in (0, 1, 2, 5, 10, 20, 50):
        scenario = "base" if bps == 0 else f"cost_{bps}bps"
        cost_daily = turnover * (float(bps) / 10000.0)
        stressed = portfolio_returns - cost_daily
        metrics = compute_performance_metrics(stressed, freq=freq)
        assumption = (
            "Base scenario without additional cost drag."
            if bps == 0
            else f"Daily execution cost = portfolio gross_turnover * {bps}bps using the exported turnover diagnostic."
        )
        rows.append(
            {
                "scenario": scenario,
                "cost_bps": float(bps),
                "turnover_source": str(info.get("source", "")),
                "turnover_quality": str(info.get("quality", "")),
                "avg_daily_turnover": info.get("avg_daily_turnover"),
                "ann_turnover": info.get("ann_turnover"),
                "total_cost_drag": float(cost_daily.sum()),
                "ann_return": metrics.get("ann_return"),
                "ann_vol": metrics.get("ann_vol"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "calmar": metrics.get("calmar"),
                "cumulative_return": metrics.get("cumulative_return"),
                "assumption": assumption,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_execution_delay_stress(
    *,
    country_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    daily_exposure: pd.DataFrame,
) -> pd.DataFrame:
    _ = (country_returns, portfolio_returns, daily_exposure)
    return pd.DataFrame(
        [
            {
                "scenario": "not_available",
                "delay_days": np.nan,
                "ann_return": np.nan,
                "ann_vol": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "calmar": np.nan,
                "cumulative_return": np.nan,
                "assumption": (
                    "Execution-delay stress was not computed because the frozen reference exposes only realized daily returns "
                    "plus proxy exposure diagnostics, not the true daily signal or position state required for a defensible shift test."
                ),
                "available_inputs": "portfolio_returns, country_returns, exposure_proxy_only",
            }
        ]
    )


def discover_trade_ledgers(config: dict[str, Any], *, project_root: Path) -> pd.DataFrame:
    columns = [
        "country",
        "book",
        "config_name",
        "status",
        "path",
        "selected_rows",
        "total_rows",
        "source_columns",
        "notes",
        "search_roots",
    ]
    rows: list[dict[str, Any]] = []
    for book_cfg in config.get("books", []):
        country = str(book_cfg.get("country", book_cfg.get("book", ""))).strip().lower()
        book = str(book_cfg.get("book", country)).strip().lower()
        config_name = str(book_cfg.get("config_name", "")).strip()
        roots = _resolve_trade_search_roots(book_cfg, config, project_root=project_root)
        search_roots_str = "; ".join(str(path) for path in roots)
        exact_match: dict[str, Any] | None = None
        near_match: dict[str, Any] | None = None

        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.csv")):
                lower = path.name.lower()
                if "trade" not in lower:
                    continue
                try:
                    sample = pd.read_csv(path, nrows=5)
                except Exception:
                    continue
                normalized_cols = {str(col).strip().lower() for col in sample.columns}
                if not _looks_like_trade_ledger(normalized_cols):
                    continue
                try:
                    frame = pd.read_csv(path)
                except Exception:
                    continue
                if frame.empty:
                    continue
                normalized = {str(col).strip().lower(): col for col in frame.columns}
                selected = frame.copy()
                notes: list[str] = []
                status = "candidate"
                if "config_name" in normalized:
                    config_col = normalized["config_name"]
                    exact_rows = frame[frame[config_col].astype(str).eq(config_name)].copy()
                    if not exact_rows.empty:
                        selected = exact_rows
                        status = "exact"
                    else:
                        unique_cfg = sorted(frame[config_col].astype(str).dropna().unique().tolist())
                        similar = [value for value in unique_cfg if value in config_name or config_name in value]
                        status = "near_match" if similar else "mismatch"
                        if unique_cfg:
                            notes.append(f"available configs={unique_cfg}")
                        if similar:
                            notes.append(f"closest configs={similar}")
                else:
                    status = "unknown_config"
                    notes.append("file has no config_name column")
                if not selected.empty and status == "exact":
                    candidate = {
                        "country": country,
                        "book": book,
                        "config_name": config_name,
                        "status": "exact",
                        "path": str(path),
                        "selected_rows": int(len(selected)),
                        "total_rows": int(len(frame)),
                        "source_columns": "; ".join(str(col) for col in frame.columns),
                        "notes": "; ".join(notes) if notes else "Exact config rows found.",
                        "search_roots": search_roots_str,
                    }
                    if exact_match is None or int(candidate["selected_rows"]) > int(exact_match["selected_rows"]):
                        exact_match = candidate
                elif status in {"near_match", "unknown_config"}:
                    candidate = {
                        "country": country,
                        "book": book,
                        "config_name": config_name,
                        "status": "near_match" if status == "near_match" else "candidate_no_config",
                        "path": str(path),
                        "selected_rows": 0,
                        "total_rows": int(len(frame)),
                        "source_columns": "; ".join(str(col) for col in frame.columns),
                        "notes": "; ".join(notes) if notes else "No exact config rows found.",
                        "search_roots": search_roots_str,
                    }
                    if near_match is None:
                        near_match = candidate

        if exact_match is not None:
            rows.append(exact_match)
        elif near_match is not None:
            rows.append(near_match)
        else:
            rows.append(
                {
                    "country": country,
                    "book": book,
                    "config_name": config_name,
                    "status": "missing",
                    "path": "",
                    "selected_rows": 0,
                    "total_rows": 0,
                    "source_columns": "",
                    "notes": "No row-level trade ledger with the frozen config was identified in the inspected roots.",
                    "search_roots": search_roots_str,
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["country", "status"]).reset_index(drop=True)


def harmonize_trade_ledger(trade_sources: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None]:
    columns = [
        "trade_id",
        "country",
        "pair_id",
        "long_leg",
        "short_leg",
        "entry_date",
        "exit_date",
        "holding_days",
        "entry_zscore",
        "exit_zscore",
        "pnl",
        "return",
        "gross_exposure",
        "net_exposure",
        "capital_at_entry",
        "capital_at_exit",
        "status",
        "config_name",
        "source_file",
    ]
    if trade_sources.empty:
        schema = "# Trade Ledger Schema\n\n- No trade sources were discovered.\n"
        return (
            pd.DataFrame([{"trade_id": "not_available", "status": "not_available"}]),
            schema,
            "No trade sources were discovered while building the execution diagnostics.",
        )

    exact = trade_sources[trade_sources["status"].astype(str).eq("exact")].copy()
    frames: list[pd.DataFrame] = []
    schema_lines = [
        "# Trade Ledger Schema",
        "",
        "## Source Discovery",
    ]
    for row in trade_sources.itertuples(index=False):
        schema_lines.append(
            f"- `{row.country}` | status=`{row.status}` | config=`{row.config_name}` | path=`{row.path or 'n/a'}` | selected_rows={row.selected_rows} | notes={row.notes}"
        )

    if exact.empty:
        schema_lines.extend(
            [
                "",
                "## Outcome",
                "- No exact frozen trade ledger was identified, so `trade_ledger_portfolio.csv` is exported as `not_available`.",
            ]
        )
        warning = "No exact frozen trade ledger was identified for the portfolio-level execution diagnostics."
        not_available = pd.DataFrame(
            [
                {
                    "trade_id": "not_available",
                    "country": "portfolio",
                    "pair_id": "",
                    "long_leg": "",
                    "short_leg": "",
                    "entry_date": "",
                    "exit_date": "",
                    "holding_days": np.nan,
                    "entry_zscore": np.nan,
                    "exit_zscore": np.nan,
                    "pnl": np.nan,
                    "return": np.nan,
                    "gross_exposure": np.nan,
                    "net_exposure": np.nan,
                    "capital_at_entry": np.nan,
                    "capital_at_exit": np.nan,
                    "status": "not_available",
                    "config_name": "",
                    "source_file": "",
                }
            ],
            columns=columns,
        )
        return not_available, "\n".join(schema_lines) + "\n", warning

    for row in exact.itertuples(index=False):
        path = Path(str(row.path))
        frame = pd.read_csv(path)
        if "config_name" in frame.columns:
            frame = frame[frame["config_name"].astype(str).eq(str(row.config_name))].copy()
        if frame.empty:
            continue
        frames.append(_harmonize_trade_frame(frame, country=str(row.country), config_name=str(row.config_name), source_file=str(path)))

    if not frames:
        schema_lines.extend(
            [
                "",
                "## Outcome",
                "- Exact trade source files were found but no rows survived the frozen-config filter.",
            ]
        )
        warning = "Exact trade source files were found, but no row survived the frozen-config filter during harmonization."
        not_available = pd.DataFrame([{"trade_id": "not_available", "status": "not_available"}])
        return not_available, "\n".join(schema_lines) + "\n", warning

    ledger = pd.concat(frames, ignore_index=True, sort=False)
    ledger["entry_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce")
    ledger["exit_date"] = pd.to_datetime(ledger["exit_date"], errors="coerce")
    ledger = ledger.sort_values(["entry_date", "exit_date", "country", "pair_id"]).reset_index(drop=True)
    ledger["trade_id"] = [f"core4_{idx + 1:05d}" for idx in range(len(ledger))]
    ledger["entry_date"] = ledger["entry_date"].dt.strftime("%Y-%m-%d")
    ledger["exit_date"] = ledger["exit_date"].dt.strftime("%Y-%m-%d")
    ledger = ledger.loc[:, [col for col in columns if col in ledger.columns]]

    missing_books = trade_sources[trade_sources["status"].astype(str).ne("exact")]["country"].astype(str).tolist()
    warning = None
    if missing_books:
        warning = (
            "Trade ledger coverage is partial. Exact frozen trade rows were found for "
            f"{', '.join(sorted(ledger['country'].dropna().astype(str).unique().tolist()))}, "
            f"but not for {', '.join(sorted(missing_books))}."
        )

    schema_lines.extend(
        [
            "",
            "## Column Mapping",
            "- `trade_id`: generated sequentially for the consolidated ledger.",
            "- `long_leg` / `short_leg`: derived from `side` plus `asset_left`/`asset_right` when available.",
            "- `pnl`: mapped from the best available per-trade PnL column such as `pnl`, `trade_pnl`, or `net_pnl`.",
            "- `return`: mapped from `return_pct`, `trade_return`, or `trade_return_isolated` when available.",
            "- `gross_exposure` / `net_exposure`: left as NaN unless explicitly present in the source ledger.",
            "",
            "## Known Gaps",
            "- The annual and concentration diagnostics on this ledger are exact only for the books with `status=exact` above.",
            "- The Germany frozen mitigated config did not expose an exact row-level trade ledger during discovery, so it is excluded rather than approximated from a near-match.",
            "- Year-based trade concentration uses `exit_date` when available, otherwise `entry_date`.",
            "- `pnl_gini` in the extended concentration file is computed on absolute trade PnL magnitudes.",
        ]
    )
    return ledger.reset_index(drop=True), "\n".join(schema_lines) + "\n", warning


def compute_trade_concentration_summary(
    trade_ledger: pd.DataFrame,
    trade_sources: pd.DataFrame,
) -> pd.DataFrame:
    if trade_ledger.empty or "status" in trade_ledger.columns and trade_ledger["status"].astype(str).eq("not_available").all():
        available_inputs = "; ".join(trade_sources["path"].astype(str).tolist()) if not trade_sources.empty else "none"
        return pd.DataFrame(
            [
                {
                    "status": "not_available",
                    "reason": "No exact portfolio-wide trade ledger was available.",
                    "available_inputs": available_inputs,
                }
            ]
        )

    pnl_source = trade_ledger["pnl"] if "pnl" in trade_ledger.columns else pd.Series(dtype=float)
    pnl = pd.to_numeric(pnl_source, errors="coerce").dropna()
    holding = _extract_holding_days(trade_ledger)
    included = sorted(trade_ledger["country"].dropna().astype(str).unique().tolist()) if "country" in trade_ledger.columns else []
    status = "partial" if not trade_sources.empty and trade_sources["status"].astype(str).ne("exact").any() else "ok"
    total_pnl = float(pnl.sum()) if not pnl.empty else np.nan
    return pd.DataFrame(
        [
            {
                "status": status,
                "reason": "Partial exact coverage only." if status == "partial" else "Exact trade ledger coverage found for every book.",
                "available_inputs": "; ".join(sorted(set(trade_ledger["source_file"].dropna().astype(str).tolist()))),
                "countries_included": ", ".join(included),
                "n_trades": int(len(pnl)),
                "total_pnl": total_pnl,
                "mean_trade_pnl": float(pnl.mean()) if not pnl.empty else np.nan,
                "median_trade_pnl": float(pnl.median()) if not pnl.empty else np.nan,
                "hit_ratio_trade": float((pnl > 0.0).mean()) if not pnl.empty else np.nan,
                "top_1pct_pnl_share": _tail_share(pnl, pct=0.01, side="top", total=total_pnl),
                "top_5pct_pnl_share": _tail_share(pnl, pct=0.05, side="top", total=total_pnl),
                "top_10pct_pnl_share": _tail_share(pnl, pct=0.10, side="top", total=total_pnl),
                "worst_1pct_pnl_share": _tail_share(pnl, pct=0.01, side="bottom", total=total_pnl),
                "worst_5pct_pnl_share": _tail_share(pnl, pct=0.05, side="bottom", total=total_pnl),
                "largest_winner": float(pnl.max()) if not pnl.empty else np.nan,
                "largest_loser": float(pnl.min()) if not pnl.empty else np.nan,
                "avg_holding_days": float(holding.mean()) if holding is not None and not holding.empty else np.nan,
                "median_holding_days": float(holding.median()) if holding is not None and not holding.empty else np.nan,
            }
        ]
    )


def compute_trade_concentration_extended(
    trade_ledger: pd.DataFrame,
    trade_sources: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "scope",
        "country",
        "year",
        "n_trades",
        "total_pnl",
        "mean_trade_pnl",
        "median_trade_pnl",
        "hit_ratio_trade",
        "avg_holding_days",
        "median_holding_days",
        "largest_winner",
        "largest_loser",
        "top_1pct_pnl_share",
        "top_5pct_pnl_share",
        "top_10pct_pnl_share",
        "worst_1pct_pnl_share",
        "worst_5pct_pnl_share",
        "pnl_gini",
        "status",
        "coverage_notes",
    ]
    if trade_ledger.empty or "status" in trade_ledger.columns and trade_ledger["status"].astype(str).eq("not_available").all():
        return pd.DataFrame(
            [
                {
                    "scope": "global",
                    "country": "all",
                    "year": "all",
                    "n_trades": 0,
                    "total_pnl": np.nan,
                    "mean_trade_pnl": np.nan,
                    "median_trade_pnl": np.nan,
                    "hit_ratio_trade": np.nan,
                    "avg_holding_days": np.nan,
                    "median_holding_days": np.nan,
                    "largest_winner": np.nan,
                    "largest_loser": np.nan,
                    "top_1pct_pnl_share": np.nan,
                    "top_5pct_pnl_share": np.nan,
                    "top_10pct_pnl_share": np.nan,
                    "worst_1pct_pnl_share": np.nan,
                    "worst_5pct_pnl_share": np.nan,
                    "pnl_gini": np.nan,
                    "status": "not_available",
                    "coverage_notes": "No exact consolidated trade ledger was available.",
                }
            ],
            columns=columns,
        )

    frame = trade_ledger.copy()
    frame["pnl"] = pd.to_numeric(frame["pnl"] if "pnl" in frame.columns else pd.Series(np.nan, index=frame.index), errors="coerce")
    frame["holding_days"] = pd.to_numeric(
        frame["holding_days"] if "holding_days" in frame.columns else pd.Series(np.nan, index=frame.index),
        errors="coerce",
    )
    exit_dates = pd.to_datetime(frame["exit_date"] if "exit_date" in frame.columns else pd.Series(pd.NaT, index=frame.index), errors="coerce")
    entry_dates = pd.to_datetime(frame["entry_date"] if "entry_date" in frame.columns else pd.Series(pd.NaT, index=frame.index), errors="coerce")
    trade_year = exit_dates.dt.year.where(exit_dates.notna(), entry_dates.dt.year)
    frame["trade_year"] = pd.to_numeric(trade_year, errors="coerce")
    status = "partial" if not trade_sources.empty and trade_sources["status"].astype(str).ne("exact").any() else "ok"
    notes = (
        "Exact coverage is partial because at least one frozen book had no exact row-level trade ledger."
        if status == "partial"
        else "Exact coverage found for every frozen book."
    )

    rows: list[dict[str, Any]] = []
    rows.append(_summarize_trade_group(frame, scope="global", country="all", year="all", status=status, coverage_notes=notes))
    for country, group in frame.groupby("country", sort=True):
        rows.append(_summarize_trade_group(group, scope="by_country", country=str(country), year="all", status=status, coverage_notes=notes))
    for year, group in frame.dropna(subset=["trade_year"]).groupby("trade_year", sort=True):
        rows.append(_summarize_trade_group(group, scope="by_year", country="all", year=str(int(year)), status=status, coverage_notes=notes))
    for (country, year), group in frame.dropna(subset=["trade_year"]).groupby(["country", "trade_year"], sort=True):
        rows.append(
            _summarize_trade_group(
                group,
                scope="by_country_year",
                country=str(country),
                year=str(int(year)),
                status=status,
                coverage_notes=notes,
            )
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["scope", "country", "year"]).reset_index(drop=True)


def write_drawdown_2023_deep_dive(
    *,
    drawdown_events: pd.DataFrame,
    drawdown_attribution_by_country: pd.DataFrame,
    country_returns: pd.DataFrame,
    daily_turnover: pd.DataFrame,
    trade_ledger: pd.DataFrame,
    output_path: Path | None = None,
) -> str:
    _ = output_path
    event = _find_2023_drawdown_event(drawdown_events)
    lines = [
        "# 2023 Drawdown Deep Dive",
        "",
    ]
    if event is None:
        lines.extend(
            [
                "## Event",
                "- No drawdown event overlapping calendar year 2023 was identified in `drawdown_events.csv`.",
                "",
                "## Conclusion",
                "- 2023-specific drawdown deep dive is not available because no overlapping drawdown event was found.",
            ]
        )
        return "\n".join(lines) + "\n"

    peak_date = pd.Timestamp(event["peak_date"])
    trough_date = pd.Timestamp(event["trough_date"])
    recovery_date = pd.Timestamp(event["recovery_date"]) if pd.notna(event["recovery_date"]) and str(event["recovery_date"]) != "NaT" else pd.NaT
    rank = int(event["rank"])
    attrib = drawdown_attribution_by_country[
        pd.to_numeric(drawdown_attribution_by_country["drawdown_rank"], errors="coerce").eq(rank)
    ].copy()
    attrib = attrib.sort_values("weighted_contribution_window").reset_index(drop=True)
    worst_countries = attrib.head(2)["country"].astype(str).tolist() if not attrib.empty else []
    concentration_note = _drawdown_concentration_note(attrib)

    lines.extend(
        [
            "## Event",
            f"- Drawdown rank: `{rank}`",
            f"- Peak date: `{peak_date.strftime('%Y-%m-%d')}`",
            f"- Trough date: `{trough_date.strftime('%Y-%m-%d')}`",
            f"- Recovery date: `{recovery_date.strftime('%Y-%m-%d') if pd.notna(recovery_date) else 'not recovered in sample'}`",
            f"- Portfolio drawdown: {_fmt_pct(event['drawdown'])}",
            "",
            "## Country Attribution",
        ]
    )
    if attrib.empty:
        lines.append("- Country attribution was not available for this drawdown window.")
    else:
        for row in attrib.itertuples(index=False):
            lines.append(
                f"- `{row.country}`: window return {_fmt_pct(row.country_cumulative_return_window)}, "
                f"weighted contribution {_fmt_pct(row.weighted_contribution_window)}."
            )

    window_returns = country_returns.loc[(country_returns.index >= peak_date) & (country_returns.index <= trough_date)].copy()
    lines.extend(["", "## Country Metrics On Peak-To-Trough Window"])
    if window_returns.empty:
        lines.append("- Country window metrics were not available.")
    else:
        for country in window_returns.columns:
            metrics = compute_performance_metrics(window_returns[country])
            lines.append(
                f"- `{country}`: cumulative return {_fmt_pct(metrics.get('cumulative_return'))}, "
                f"Sharpe {_fmt_number(metrics.get('sharpe'))}, max drawdown {_fmt_pct(metrics.get('max_drawdown'))}."
            )

    lines.extend(["", "## Trade-Level View"])
    if trade_ledger.empty or "status" in trade_ledger.columns and trade_ledger["status"].astype(str).eq("not_available").all():
        lines.append("- Trade-level unavailable: no exact consolidated trade ledger was found.")
    else:
        trades = trade_ledger.copy()
        trades["entry_date"] = pd.to_datetime(
            trades["entry_date"] if "entry_date" in trades.columns else pd.Series(pd.NaT, index=trades.index),
            errors="coerce",
        )
        trades["exit_date"] = pd.to_datetime(
            trades["exit_date"] if "exit_date" in trades.columns else pd.Series(pd.NaT, index=trades.index),
            errors="coerce",
        )
        overlap = trades[(trades["entry_date"] <= trough_date) & (trades["exit_date"] >= peak_date)].copy()
        overlap["pnl"] = pd.to_numeric(
            overlap["pnl"] if "pnl" in overlap.columns else pd.Series(np.nan, index=overlap.index),
            errors="coerce",
        )
        if overlap.empty:
            lines.append("- No exact trade row overlapped the 2023 drawdown window in the available consolidated ledger.")
        else:
            worst_trades = overlap.nsmallest(5, "pnl")[["country", "pair_id", "entry_date", "exit_date", "pnl"]]
            top_losing_pairs = overlap.groupby("pair_id", dropna=False)["pnl"].sum().sort_values().head(5)
            holding = _extract_holding_days(overlap)
            lines.append(
                f"- Trade coverage is partial. Exact rows are available for countries `{', '.join(sorted(overlap['country'].dropna().astype(str).unique().tolist()))}` only."
            )
            if holding is not None and not holding.empty:
                lines.append(
                    f"- Average holding period on overlapping trades: {_fmt_number(holding.mean())} days "
                    f"(median {_fmt_number(holding.median())})."
                )
            lines.append("- Worst overlapping trades:")
            for row in worst_trades.itertuples(index=False):
                lines.append(
                    f"  - `{row.country}` | `{row.pair_id}` | {pd.Timestamp(row.entry_date).strftime('%Y-%m-%d')} -> "
                    f"{pd.Timestamp(row.exit_date).strftime('%Y-%m-%d')} | pnl {_fmt_number(row.pnl)}"
                )
            lines.append("- Worst pair aggregates on overlapping trades:")
            for pair_id, value in top_losing_pairs.items():
                lines.append(f"  - `{pair_id}`: {_fmt_number(value)}")

    lines.extend(["", "## Turnover And Cost"])
    turnover = extract_portfolio_turnover_series(daily_turnover)
    turnover_info = summarize_turnover_input(daily_turnover)
    window_turnover = turnover.loc[(turnover.index >= peak_date) & (turnover.index <= trough_date)] if not turnover.empty else pd.Series(dtype=float)
    if window_turnover.empty or not bool(turnover_info.get("available")):
        lines.append("- Turnover unavailable: no exploitable portfolio turnover series was identified for this window.")
    else:
        cost_10bps = float((window_turnover * (10.0 / 10000.0)).sum())
        lines.append(
            f"- Turnover quality: `{turnover_info.get('quality')}` from `{turnover_info.get('source')}`."
        )
        lines.append(
            f"- Average daily gross turnover during the window: {_fmt_pct(window_turnover.mean())}; "
            f"10bps cumulative cost drag proxy on the window: {_fmt_pct(cost_10bps)}."
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            f"- {concentration_note}",
            (
                f"- Worst country contributors on peak->trough are {', '.join(worst_countries)}."
                if worst_countries
                else "- Country attribution on peak->trough is unavailable."
            ),
            (
                "- Trade-level coverage is partial because at least one frozen book does not expose an exact row-level ledger."
                if not trade_ledger.empty and not trade_ledger["status"].astype(str).eq("not_available").all()
                else "- Trade-level unavailable."
            ),
            (
                f"- Turnover quality is `{turnover_info.get('quality')}`."
                if bool(turnover_info.get("available"))
                else "- Turnover unavailable."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def export_validation_pack(output_dir: Path, bundle: dict[str, pd.DataFrame]) -> None:
    file_map = {
        "performance_by_year": "performance_by_year.csv",
        "performance_by_period": "performance_by_period.csv",
        "daily_country_contribution": "daily_country_contribution.csv",
        "daily_country_contribution_wide": "daily_country_contribution_wide.csv",
        "annual_country_contribution": "annual_country_contribution.csv",
        "annual_country_metrics": "annual_country_metrics.csv",
        "leave_one_country_out": "leave_one_country_out.csv",
        "leave_one_country_out_by_year": "leave_one_country_out_by_year.csv",
        "country_standalone": "country_standalone.csv",
        "weight_sensitivity": "weight_sensitivity.csv",
        "cost_stress": "cost_stress.csv",
        "cost_stress_turnover_based": "cost_stress_turnover_based.csv",
        "trade_concentration": "trade_concentration.csv",
        "trade_concentration_extended": "trade_concentration_extended.csv",
        "trade_ledger_portfolio": "trade_ledger_portfolio.csv",
        "rolling_metrics": "rolling_metrics.csv",
        "drawdown_events": "drawdown_events.csv",
        "drawdown_attribution_by_country": "drawdown_attribution_by_country.csv",
        "worst_periods_attribution": "worst_periods_attribution.csv",
        "correlation_matrix": "correlation_matrix.csv",
        "monthly_returns": "monthly_returns.csv",
        "monthly_returns_pivot": "monthly_returns_pivot.csv",
        "daily_exposure": "daily_exposure.csv",
        "daily_turnover": "daily_turnover.csv",
        "execution_delay_stress": "execution_delay_stress.csv",
    }
    indexed_exports = {"correlation_matrix", "monthly_returns_pivot"}
    for key, filename in file_map.items():
        frame = bundle.get(key, pd.DataFrame())
        if not isinstance(frame, pd.DataFrame):
            continue
        frame.to_csv(output_dir / filename, index=key in indexed_exports)


def build_validation_summary(
    *,
    generated_at: datetime,
    context: dict[str, Any],
    performance_by_year: pd.DataFrame,
    performance_by_period: pd.DataFrame,
    annual_country_contribution: pd.DataFrame,
    annual_country_metrics: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    leave_one_out_by_year: pd.DataFrame,
    country_standalone: pd.DataFrame,
    weight_sensitivity: pd.DataFrame,
    cost_stress: pd.DataFrame,
    cost_stress_turnover_based: pd.DataFrame,
    trade_concentration: pd.DataFrame,
    trade_concentration_extended: pd.DataFrame,
    trade_ledger_portfolio: pd.DataFrame,
    drawdown_events: pd.DataFrame,
    drawdown_attribution_by_country: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    country_contribution: pd.DataFrame,
    daily_turnover: pd.DataFrame,
    execution_delay_stress: pd.DataFrame,
    drawdown_2023_md: str,
    warnings_list: list[str],
) -> str:
    portfolio_returns = context["portfolio_returns"]
    config = context["config"]
    full_metrics = compute_performance_metrics(portfolio_returns)
    best_year = performance_by_year.sort_values("cumulative_return", ascending=False).head(1)
    worst_year = performance_by_year.sort_values("cumulative_return", ascending=True).head(1)
    worst_event = drawdown_events.head(1)
    dependency = leave_one_out.sort_values("delta_cumulative_return_vs_frozen", ascending=True).head(1)
    top_country = country_contribution.head(1)
    corr_stats = _correlation_stats(correlation_matrix)
    primary_cost_stress = (
        cost_stress_turnover_based
        if not cost_stress_turnover_based.empty and str(cost_stress_turnover_based.iloc[0].get("scenario", "")) != "not_available"
        else cost_stress
    )
    cost_note = _cost_summary(primary_cost_stress)
    random_note = _random_sensitivity_note(weight_sensitivity)
    turnover_info = summarize_turnover_input(daily_turnover)
    event_2023 = _find_2023_drawdown_event(drawdown_events)
    drawdown_2023_lines = _drawdown_2023_summary_lines(
        event_2023=event_2023,
        drawdown_attribution_by_country=drawdown_attribution_by_country,
        daily_turnover=daily_turnover,
        trade_ledger_portfolio=trade_ledger_portfolio,
    )
    remaining_blockers = _remaining_blockers_lines(
        trade_ledger_portfolio=trade_ledger_portfolio,
        trade_concentration_extended=trade_concentration_extended,
        turnover_info=turnover_info,
        execution_delay_stress=execution_delay_stress,
    )

    lines = [
        "# Core4 Validation Pack",
        "",
        "## Run Context",
        f"- Run timestamp: {generated_at.isoformat(sep=' ')}",
        f"- Portfolio id: {config.get('portfolio_id', 'unknown')}",
        f"- Portfolio status: {config.get('status', 'unknown')}",
        f"- Frozen at: {config.get('frozen_at', 'unknown')}",
        f"- Analysis window: {_fmt_date(portfolio_returns.index.min())} to {_fmt_date(portfolio_returns.index.max())}",
        f"- Number of trading days: {len(portfolio_returns)}",
        f"- Frozen weight scheme: {config.get('default_weight_scheme', 'unknown')}",
        "- Reminder: this pack validates the existing frozen reference only. It does not retune countries, re-optimize allocation, or modify the trading engine.",
        "",
        "## Inputs Loaded",
        f"- Config: {context['config_path']}",
        f"- Daily frozen source via helper `load_or_build_core_daily_returns`: {context['daily_cache_dir']}",
        f"- Monthly frozen source files: {', '.join(str(path) for path in context['monthly_source_paths'])}",
        f"- Frozen weights: {_format_weight_summary(context['frozen_weights'])}",
        "",
        "## Full-Sample Metrics",
        f"- Annualized return: {_fmt_pct(full_metrics.get('ann_return'))}",
        f"- Annualized volatility: {_fmt_pct(full_metrics.get('ann_vol'))}",
        f"- Sharpe: {_fmt_number(full_metrics.get('sharpe'))}",
        f"- Max drawdown: {_fmt_pct(full_metrics.get('max_drawdown'))}",
        f"- Calmar: {_fmt_number(full_metrics.get('calmar'))}",
        f"- Cumulative return: {_fmt_pct(full_metrics.get('cumulative_return'))}",
        f"- Daily hit ratio: {_fmt_pct(full_metrics.get('hit_ratio_daily'))}",
        "",
        "## Automatic Findings",
        _best_year_line(best_year),
        _worst_year_line(worst_year),
        _drawdown_line(worst_event),
        _dependency_line(dependency),
        _top_country_line(top_country),
        _correlation_line(corr_stats),
        cost_note,
        random_note,
        "",
        "## Warnings",
    ]

    if warnings_list:
        lines.extend(f"- {warning}" for warning in warnings_list)
    else:
        lines.append("- No material warning was raised by the validation pack.")

    lines.extend(
        [
            "",
            "## Diagnostics Availability",
            f"- Custom periods exported: {len(performance_by_period)} rows",
            f"- Trade concentration status: {trade_concentration.iloc[0].get('status', 'unknown') if not trade_concentration.empty else 'unknown'}",
            "",
            "## Execution-readiness diagnostics",
            f"- Turnover availability: {'available' if bool(turnover_info.get('available')) else 'not available'}",
            f"- Turnover quality: `{turnover_info.get('quality', 'unknown')}`",
            (
                f"- Cost stress quality: turnover-based (`{turnover_info.get('quality', 'unknown')}`)"
                if not cost_stress_turnover_based.empty and str(cost_stress_turnover_based.iloc[0].get("scenario", "")) != "not_available"
                else "- Cost stress quality: fallback constant-drag proxy or unavailable"
            ),
            f"- Trade ledger availability: {_trade_ledger_availability_text(trade_ledger_portfolio, trade_concentration_extended)}",
            f"- Trade concentration availability: {trade_concentration_extended.iloc[0].get('status', 'unknown') if not trade_concentration_extended.empty else 'unknown'}",
            f"- Drawdown attribution availability: {'available' if not drawdown_attribution_by_country.empty else 'not available'}",
            (
                f"- Execution delay stress availability: {execution_delay_stress.iloc[0].get('scenario', 'unknown')}"
                if not execution_delay_stress.empty
                else "- Execution delay stress availability: unknown"
            ),
            "",
            "## 2023 drawdown diagnostic",
        ]
    )
    lines.extend(drawdown_2023_lines)

    lines.extend(
        [
            "",
            "## Remaining blockers before IBKR paper",
        ]
    )
    lines.extend(remaining_blockers)

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- Leave-one-country-out uses naive equal-weight three-country portfolios. It is a dependency check, not an optimizer.",
            "- Weight sensitivity is deterministic. Random perturbations use a fixed seed and moderate multiplicative noise around the frozen weights.",
            "- Country attribution files use additive weighted daily-return sums for contribution accounting. They are directionally informative but do not force an exact compounded decomposition.",
            "- Cost stress should be interpreted through the quality flag. In this extension, turnover can remain a proxy even when the file is available.",
            "- Trade-level concentration can be partial when one frozen book does not expose an exact row-level ledger.",
            f"- Detailed 2023 deep dive is also exported separately in `drawdown_2023_deep_dive.md` ({len(drawdown_2023_md.splitlines())} lines).",
            f"- Annual country contribution rows exported: {len(annual_country_contribution)} | annual country metrics rows: {len(annual_country_metrics)} | leave-one-out-by-year rows: {len(leave_one_out_by_year)}.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_equity_curve(path: Path, equity: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(equity.index, equity.values, linewidth=1.8, color="#0f4c5c")
    ax.set_title("Frozen core4 equity curve")
    ax.set_ylabel("Equity (base 1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdown(path: Path, drawdown_frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(drawdown_frame["date"], drawdown_frame["drawdown"], 0.0, alpha=0.30, color="#8f2d56")
    ax.axhline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Frozen core4 drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_yearly_returns(path: Path, performance_by_year: pd.DataFrame) -> None:
    if performance_by_year.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#2d6a4f" if value >= 0 else "#8f2d56" for value in performance_by_year["cumulative_return"]]
    positions = np.arange(len(performance_by_year))
    ax.bar(positions, performance_by_year["cumulative_return"], color=colors)
    ax.set_xticks(positions)
    ax.set_xticklabels(performance_by_year["year"].astype(str))
    ax.axhline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Frozen core4 yearly cumulative returns")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(path: Path, rolling_metrics: pd.DataFrame) -> None:
    if rolling_metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    color_map = {63: "#1f4e79", 126: "#2d6a4f", 252: "#8d6a00"}
    for window, group in rolling_metrics.groupby("window", sort=True):
        ax.plot(group["date"], group["rolling_sharpe"], label=f"{int(window)}d", linewidth=1.6, color=color_map.get(int(window), "#4c566a"))
    ax.axhline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Rolling Sharpe by window")
    ax.set_ylabel("Rolling Sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_country_contribution(path: Path, country_contribution: pd.DataFrame) -> None:
    if country_contribution.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(country_contribution["country"], country_contribution["weighted_compounded_return"], color="#2d6a4f")
    ax.axhline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Frozen weighted contribution by country")
    ax.set_ylabel("Weighted compounded return")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(path: Path, correlation_matrix: pd.DataFrame) -> None:
    if correlation_matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(correlation_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.index)
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Daily correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annual_country_contribution(path: Path, annual_country_contribution: pd.DataFrame) -> None:
    if annual_country_contribution.empty:
        return
    pivot = (
        annual_country_contribution.pivot_table(
            index="year",
            columns="country",
            values="annual_weighted_contribution",
            aggfunc="sum",
        )
        .sort_index()
        .fillna(0.0)
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Annual country contribution")
    ax.set_ylabel("Additive weighted contribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Country", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdown_attribution_by_country(path: Path, drawdown_attribution_by_country: pd.DataFrame) -> None:
    if drawdown_attribution_by_country.empty:
        return
    subset = drawdown_attribution_by_country[
        pd.to_numeric(drawdown_attribution_by_country["drawdown_rank"], errors="coerce") <= 5
    ].copy()
    if subset.empty:
        subset = drawdown_attribution_by_country.copy()
    pivot = (
        subset.pivot_table(
            index="drawdown_rank",
            columns="country",
            values="weighted_contribution_window",
            aggfunc="sum",
        )
        .sort_index()
        .fillna(0.0)
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="barh", stacked=True, ax=ax, width=0.80)
    ax.axvline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Drawdown attribution by country")
    ax.set_xlabel("Additive weighted contribution on peak-to-trough window")
    ax.set_ylabel("Drawdown rank")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(title="Country", loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_daily_turnover(path: Path, daily_turnover: pd.DataFrame) -> None:
    if daily_turnover.empty or "gross_turnover" not in daily_turnover.columns:
        return
    frame = daily_turnover.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "country" in frame.columns and frame["country"].astype(str).eq("portfolio").any():
        frame = frame[frame["country"].astype(str).eq("portfolio")].copy()
    if frame.empty or frame["turnover_quality"].astype(str).eq("not_available").all():
        return
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(frame["date"], frame["gross_turnover"], linewidth=1.4, color="#6a4c93")
    ax.set_title("Daily gross turnover")
    ax.set_ylabel("Gross turnover")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_stress_turnover_based(path: Path, cost_stress_turnover_based: pd.DataFrame) -> None:
    if cost_stress_turnover_based.empty or str(cost_stress_turnover_based.iloc[0].get("scenario", "")) == "not_available":
        return
    frame = cost_stress_turnover_based.copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(frame["scenario"], frame["ann_return"], color="#bc6c25")
    ax.axhline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Turnover-based cost stress")
    ax.set_ylabel("Annualized return")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_worst_periods_by_country(path: Path, worst_periods_attribution: pd.DataFrame) -> None:
    if worst_periods_attribution.empty:
        return
    summary = (
        worst_periods_attribution.groupby(["window_days", "country"], dropna=False)["weighted_contribution_window"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot_table(index="window_days", columns="country", values="weighted_contribution_window", aggfunc="sum")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0.0, color="#808080", linewidth=0.8)
    ax.set_title("Average country contribution across worst windows")
    ax.set_ylabel("Mean additive weighted contribution")
    ax.set_xlabel("Window length")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Country", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_output_dir(output_root: Path, generated_at: datetime, *, smoke: bool) -> Path:
    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    leaf = f"{stamp}_smoke" if smoke else stamp
    out = output_root / leaf
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_path(project_root: Path, path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _resolve_trade_search_roots(book_cfg: dict[str, Any], config: dict[str, Any], *, project_root: Path) -> list[Path]:
    roots: list[Path] = []
    source_dir = book_cfg.get("source_dir")
    if source_dir:
        roots.append(_resolve_path(project_root, Path(str(source_dir))))
    for value in config.get("decision_sources", {}).values():
        roots.append(_resolve_path(project_root, Path(str(value))))

    expanded: list[Path] = []
    for root in roots:
        expanded.append(root)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in metadata.get("books", []):
            if str(item.get("book", "")).strip().lower() != str(book_cfg.get("book", "")).strip().lower():
                continue
            meta_source = item.get("source_dir")
            if meta_source:
                expanded.append(_resolve_path(project_root, Path(str(meta_source))))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in expanded:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _looks_like_trade_ledger(columns: set[str]) -> bool:
    pnl_tokens = {"pnl", "trade_pnl", "net_pnl", "realized_pnl"}
    entry_tokens = {"entry_datetime", "entry_date", "open_date"}
    exit_tokens = {"exit_datetime", "exit_date", "close_date"}
    return bool(columns.intersection(pnl_tokens)) and bool(columns.intersection(entry_tokens)) and bool(columns.intersection(exit_tokens))


def _harmonize_trade_frame(frame: pd.DataFrame, *, country: str, config_name: str, source_file: str) -> pd.DataFrame:
    normalized = {str(col).strip().lower(): col for col in frame.columns}
    pnl_col = _select_first_existing(normalized, ("pnl", "trade_pnl", "net_pnl", "realized_pnl", "total_pnl"))
    return_col = _select_first_existing(normalized, ("return_pct", "trade_return", "trade_return_isolated", "return"))
    entry_col = _select_first_existing(normalized, ("entry_datetime", "entry_date", "open_date"))
    exit_col = _select_first_existing(normalized, ("exit_datetime", "exit_date", "close_date"))
    pair_col = _select_first_existing(normalized, ("pair_id",))
    holding_col = _select_first_existing(normalized, ("holding_days", "duration_days"))
    entry_z_col = _select_first_existing(normalized, ("entry_z", "z_entry_observed"))
    exit_z_col = _select_first_existing(normalized, ("exit_z",))
    gross_col = _select_first_existing(normalized, ("gross_exposure",))
    net_col = _select_first_existing(normalized, ("net_exposure",))
    capital_entry_col = _select_first_existing(normalized, ("capital_at_entry",))
    capital_exit_col = _select_first_existing(normalized, ("capital_at_exit",))

    out = pd.DataFrame(index=frame.index)
    out["country"] = str(country)
    out["pair_id"] = frame[pair_col].astype(str) if pair_col is not None else ""
    out["entry_date"] = pd.to_datetime(frame[entry_col], errors="coerce") if entry_col is not None else pd.NaT
    out["exit_date"] = pd.to_datetime(frame[exit_col], errors="coerce") if exit_col is not None else pd.NaT
    out["holding_days"] = pd.to_numeric(frame[holding_col], errors="coerce") if holding_col is not None else np.nan
    if holding_col is None and entry_col is not None and exit_col is not None:
        out["holding_days"] = (out["exit_date"] - out["entry_date"]).dt.days
    out["entry_zscore"] = pd.to_numeric(frame[entry_z_col], errors="coerce") if entry_z_col is not None else np.nan
    out["exit_zscore"] = pd.to_numeric(frame[exit_z_col], errors="coerce") if exit_z_col is not None else np.nan
    out["pnl"] = pd.to_numeric(frame[pnl_col], errors="coerce") if pnl_col is not None else np.nan
    out["return"] = pd.to_numeric(frame[return_col], errors="coerce") if return_col is not None else np.nan
    out["gross_exposure"] = pd.to_numeric(frame[gross_col], errors="coerce") if gross_col is not None else np.nan
    out["net_exposure"] = pd.to_numeric(frame[net_col], errors="coerce") if net_col is not None else np.nan
    out["capital_at_entry"] = pd.to_numeric(frame[capital_entry_col], errors="coerce") if capital_entry_col is not None else np.nan
    out["capital_at_exit"] = pd.to_numeric(frame[capital_exit_col], errors="coerce") if capital_exit_col is not None else np.nan
    out["config_name"] = str(config_name)
    out["status"] = "exact_source_match"
    out["source_file"] = str(source_file)
    long_leg, short_leg = _derive_trade_legs(frame, normalized)
    out["long_leg"] = long_leg
    out["short_leg"] = short_leg
    return out.reset_index(drop=True)


def _select_first_existing(normalized: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    return next((normalized[key] for key in candidates if key in normalized), None)


def _derive_trade_legs(frame: pd.DataFrame, normalized: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    left_col = _select_first_existing(normalized, ("asset_left", "asset_1"))
    right_col = _select_first_existing(normalized, ("asset_right", "asset_2"))
    side_col = _select_first_existing(normalized, ("side", "pair_direction_at_entry"))
    left = frame[left_col].astype(str) if left_col is not None else pd.Series("", index=frame.index, dtype=object)
    right = frame[right_col].astype(str) if right_col is not None else pd.Series("", index=frame.index, dtype=object)
    if side_col is None:
        return left, right
    side = frame[side_col].astype(str).str.upper()
    long_leg = pd.Series(np.where(side.str.contains("SHORT_SPREAD"), right, left), index=frame.index, dtype=object)
    short_leg = pd.Series(np.where(side.str.contains("SHORT_SPREAD"), left, right), index=frame.index, dtype=object)
    return long_leg, short_leg


def _summarize_trade_group(
    frame: pd.DataFrame,
    *,
    scope: str,
    country: str,
    year: str,
    status: str,
    coverage_notes: str,
) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["pnl"] if "pnl" in frame.columns else pd.Series(dtype=float), errors="coerce").dropna()
    total_pnl = float(pnl.sum()) if not pnl.empty else np.nan
    holding = pd.to_numeric(
        frame["holding_days"] if "holding_days" in frame.columns else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()
    return {
        "scope": scope,
        "country": country,
        "year": year,
        "n_trades": int(len(frame)),
        "total_pnl": total_pnl,
        "mean_trade_pnl": float(pnl.mean()) if not pnl.empty else np.nan,
        "median_trade_pnl": float(pnl.median()) if not pnl.empty else np.nan,
        "hit_ratio_trade": float((pnl > 0.0).mean()) if not pnl.empty else np.nan,
        "avg_holding_days": float(holding.mean()) if not holding.empty else np.nan,
        "median_holding_days": float(holding.median()) if not holding.empty else np.nan,
        "largest_winner": float(pnl.max()) if not pnl.empty else np.nan,
        "largest_loser": float(pnl.min()) if not pnl.empty else np.nan,
        "top_1pct_pnl_share": _tail_share(pnl, pct=0.01, side="top", total=total_pnl),
        "top_5pct_pnl_share": _tail_share(pnl, pct=0.05, side="top", total=total_pnl),
        "top_10pct_pnl_share": _tail_share(pnl, pct=0.10, side="top", total=total_pnl),
        "worst_1pct_pnl_share": _tail_share(pnl, pct=0.01, side="bottom", total=total_pnl),
        "worst_5pct_pnl_share": _tail_share(pnl, pct=0.05, side="bottom", total=total_pnl),
        "pnl_gini": _gini(pnl.abs()),
        "status": status,
        "coverage_notes": coverage_notes,
    }


def _gini(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return np.nan
    x = np.sort(series.to_numpy(dtype=float))
    if np.allclose(x, 0.0):
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n)


def _select_metric_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "ann_return": metrics.get("ann_return"),
        "ann_vol": metrics.get("ann_vol"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "calmar": metrics.get("calmar"),
        "cumulative_return": metrics.get("cumulative_return"),
    }


def _one_pass_cap_and_renormalize(weights: pd.Series, *, cap: float) -> pd.Series:
    clipped = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0, upper=float(cap))
    if float(clipped.sum()) <= 0.0:
        return pd.Series(1.0 / len(clipped), index=clipped.index, dtype=float)
    return clipped / float(clipped.sum())


def _annualized_return_from_window(values: np.ndarray, *, freq: int) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan
    total = float(np.prod(1.0 + clean))
    return float(total ** (freq / clean.size) - 1.0) if total > 0.0 else np.nan


def _rolling_max_drawdown_from_returns(returns: pd.Series, *, window: int) -> pd.Series:
    equity = equity_from_returns(returns, mode="compounded")
    out = pd.Series(np.nan, index=equity.index, dtype=float)
    values = equity.to_numpy(dtype=float)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        span = values[left : idx + 1]
        if len(span) < window:
            continue
        local_peak = np.maximum.accumulate(span)
        drawdown = span / local_peak - 1.0
        out.iloc[idx] = float(np.min(drawdown))
    return out


def _extract_holding_days(trade_ledger: pd.DataFrame) -> pd.Series | None:
    normalized = {str(col).strip().lower(): col for col in trade_ledger.columns}
    if "holding_days" in normalized:
        return pd.to_numeric(trade_ledger[normalized["holding_days"]], errors="coerce").dropna()
    entry_key = next((key for key in ("entry_date", "open_date") if key in normalized), None)
    exit_key = next((key for key in ("exit_date", "close_date") if key in normalized), None)
    if entry_key is None or exit_key is None:
        return None
    entry = pd.to_datetime(trade_ledger[normalized[entry_key]], errors="coerce")
    exit_ = pd.to_datetime(trade_ledger[normalized[exit_key]], errors="coerce")
    holding = (exit_ - entry).dt.days
    return pd.to_numeric(holding, errors="coerce").dropna()


def _tail_share(pnl: pd.Series, *, pct: float, side: str, total: float) -> float:
    if pnl.empty or not np.isfinite(total) or abs(total) <= 1e-12:
        return np.nan
    k = max(1, int(math.ceil(len(pnl) * float(pct))))
    if side == "top":
        return float(pnl.sort_values(ascending=False).head(k).sum() / total)
    return float(pnl.sort_values(ascending=True).head(k).sum() / total)


def _correlation_stats(correlation_matrix: pd.DataFrame) -> dict[str, Any]:
    if correlation_matrix.empty or len(correlation_matrix.columns) < 2:
        return {"mean_corr": np.nan, "max_corr": np.nan, "max_pair": "n/a"}
    pairs: list[tuple[str, str, float]] = []
    cols = list(correlation_matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            pairs.append((left, right, float(correlation_matrix.loc[left, right])))
    if not pairs:
        return {"mean_corr": np.nan, "max_corr": np.nan, "max_pair": "n/a"}
    mean_corr = float(np.mean([value for _, _, value in pairs]))
    left, right, max_corr = max(pairs, key=lambda item: item[2])
    return {"mean_corr": mean_corr, "max_corr": float(max_corr), "max_pair": f"{left}/{right}"}


def _cost_summary(cost_stress: pd.DataFrame) -> str:
    if cost_stress.empty:
        return "- Cost stress was not computed."
    if str(cost_stress.iloc[0].get("scenario", "")) == "not_available":
        return f"- Cost stress was not available. Assumption: {cost_stress.iloc[0].get('assumption', 'n/a')}"
    stressed = cost_stress[cost_stress["scenario"].ne("base")].copy()
    if stressed.empty:
        return "- Cost stress only contains the base scenario."
    positive = stressed[stressed["cumulative_return"] > 0.0]
    max_available = stressed["scenario"].iloc[-1]
    if len(positive) == len(stressed):
        return (
            f"- The frozen portfolio remains positive under the available cost stress grid through `{max_available}`. "
            f"Interpretation remains cautious because the assumption is `{stressed.iloc[-1]['assumption']}`"
        )
    first_negative = stressed[stressed["cumulative_return"] <= 0.0].head(1)
    if first_negative.empty:
        return "- Cost stress was computed but produced no interpretable scenario ordering."
    return (
        f"- The frozen portfolio turns non-positive under `{first_negative.iloc[0]['scenario']}`. "
        f"Assumption used: {first_negative.iloc[0]['assumption']}"
    )


def _random_sensitivity_note(weight_sensitivity: pd.DataFrame) -> str:
    if weight_sensitivity.empty or "portfolio" not in weight_sensitivity.columns:
        return "- Random weight perturbation diagnostics were not produced."
    random_rows = weight_sensitivity[weight_sensitivity["portfolio"].astype(str).str.startswith("random_")].copy()
    if random_rows.empty:
        return "- Random weight perturbation diagnostics were not produced."
    ann = pd.to_numeric(random_rows["ann_return"], errors="coerce").dropna()
    sharpe = pd.to_numeric(random_rows["sharpe"], errors="coerce").dropna()
    if ann.empty or sharpe.empty:
        return "- Random weight perturbation diagnostics were exported, but the summary percentiles are unavailable."
    return (
        f"- Random weight perturbations stay within an annualized return range of {_fmt_pct(ann.min())} to {_fmt_pct(ann.max())} "
        f"and a Sharpe range of {_fmt_number(sharpe.min())} to {_fmt_number(sharpe.max())}."
    )


def _best_year_line(best_year: pd.DataFrame) -> str:
    if best_year.empty:
        return "- Best-year summary is unavailable."
    row = best_year.iloc[0]
    return (
        f"- Best year by cumulative return: `{int(row['year'])}` with cumulative return {_fmt_pct(row['cumulative_return'])} "
        f"and Sharpe {_fmt_number(row['sharpe'])}."
    )


def _worst_year_line(worst_year: pd.DataFrame) -> str:
    if worst_year.empty:
        return "- Worst-year summary is unavailable."
    row = worst_year.iloc[0]
    return (
        f"- Worst year by cumulative return: `{int(row['year'])}` with cumulative return {_fmt_pct(row['cumulative_return'])} "
        f"and max drawdown {_fmt_pct(row['max_drawdown'])}."
    )


def _drawdown_line(worst_event: pd.DataFrame) -> str:
    if worst_event.empty:
        return "- Drawdown-event summary is unavailable."
    row = worst_event.iloc[0]
    recovery = row["recovery_date"] if pd.notna(row["recovery_date"]) and str(row["recovery_date"]) != "NaT" else "not yet recovered"
    return (
        f"- The largest drawdown event peaked on `{row['peak_date']}`, troughed on `{row['trough_date']}` at {_fmt_pct(row['drawdown'])}, "
        f"and recovered on `{recovery}`."
    )


def _dependency_line(dependency: pd.DataFrame) -> str:
    if dependency.empty:
        return "- Leave-one-country-out dependency summary is unavailable."
    row = dependency.iloc[0]
    return (
        f"- Leave-one-country-out indicates the strongest dependency on `{row['excluded_country']}`: removing it changes cumulative return by "
        f"{_fmt_pct(row['delta_cumulative_return_vs_frozen'])} and Sharpe by {_fmt_number(row['delta_sharpe_vs_frozen'])} versus the frozen reference."
    )


def _top_country_line(top_country: pd.DataFrame) -> str:
    if top_country.empty:
        return "- Country contribution summary is unavailable."
    row = top_country.iloc[0]
    return (
        f"- The most contributive frozen country is `{row['country']}` with weighted compounded contribution {_fmt_pct(row['weighted_compounded_return'])} "
        f"at frozen weight {_fmt_pct(row['weight'])}."
    )


def _correlation_line(stats: dict[str, Any]) -> str:
    return (
        f"- Average inter-book daily correlation is {_fmt_number(stats.get('mean_corr'))}; "
        f"the highest pair is `{stats.get('max_pair', 'n/a')}` at {_fmt_number(stats.get('max_corr'))}."
    )


def _trade_ledger_availability_text(trade_ledger: pd.DataFrame, trade_concentration_extended: pd.DataFrame) -> str:
    if trade_ledger.empty or "status" in trade_ledger.columns and trade_ledger["status"].astype(str).eq("not_available").all():
        return "not available"
    countries = sorted(trade_ledger["country"].dropna().astype(str).unique().tolist()) if "country" in trade_ledger.columns else []
    if not trade_concentration_extended.empty:
        status = str(trade_concentration_extended.iloc[0].get("status", "unknown"))
    else:
        status = "partial" if len(countries) < 4 else "ok"
    return f"{status} ({', '.join(countries)})"


def _find_2023_drawdown_event(drawdown_events: pd.DataFrame) -> pd.Series | None:
    if drawdown_events.empty:
        return None
    events = drawdown_events.copy()
    events["peak_date_dt"] = pd.to_datetime(events["peak_date"], errors="coerce")
    events["trough_date_dt"] = pd.to_datetime(events["trough_date"], errors="coerce")
    events["recovery_date_dt"] = pd.to_datetime(events["recovery_date"], errors="coerce")
    if events.empty:
        return None
    mask = events["trough_date_dt"].dt.year.eq(2023)
    if not mask.any():
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-12-31")
        mask = (events["peak_date_dt"] <= end) & (
            events["recovery_date_dt"].fillna(pd.Timestamp.max) >= start
        )
    subset = events.loc[mask].copy()
    if subset.empty:
        return None
    return subset.sort_values("drawdown", ascending=True).iloc[0]


def _drawdown_concentration_note(attrib: pd.DataFrame) -> str:
    if attrib.empty:
        return "Country attribution is unavailable for the selected drawdown."
    negative = attrib[pd.to_numeric(attrib["weighted_contribution_window"], errors="coerce") < 0.0].copy()
    if negative.empty:
        return "The selected drawdown window does not show a negative country attribution breakdown."
    abs_total = float(negative["weighted_contribution_window"].abs().sum())
    worst = negative.iloc[0]
    share = abs(float(worst["weighted_contribution_window"])) / abs_total if abs_total > 1e-12 else np.nan
    if np.isfinite(share) and share >= 0.50:
        return f"Drawdown concentrated in `{worst['country']}` with about {_fmt_pct(share)} of the negative additive attribution."
    return "Drawdown appears broad-based across countries rather than concentrated in a single book."


def _drawdown_2023_summary_lines(
    *,
    event_2023: pd.Series | None,
    drawdown_attribution_by_country: pd.DataFrame,
    daily_turnover: pd.DataFrame,
    trade_ledger_portfolio: pd.DataFrame,
) -> list[str]:
    if event_2023 is None:
        return ["- No drawdown event overlapping 2023 was found."]
    rank = int(event_2023["rank"])
    attrib = drawdown_attribution_by_country[
        pd.to_numeric(drawdown_attribution_by_country["drawdown_rank"], errors="coerce").eq(rank)
    ].copy()
    attrib = attrib.sort_values("weighted_contribution_window").reset_index(drop=True)
    top_bits = ", ".join(
        f"{row.country} {_fmt_pct(row.weighted_contribution_window)}"
        for row in attrib.head(2).itertuples(index=False)
    )
    turnover_info = summarize_turnover_input(daily_turnover)
    trade_status = _trade_ledger_availability_text(trade_ledger_portfolio, pd.DataFrame())
    return [
        f"- Event `{rank}` peaks on `{event_2023['peak_date']}`, troughs on `{event_2023['trough_date']}`, and draws down {_fmt_pct(event_2023['drawdown'])}.",
        f"- Peak-to-trough country attribution: {top_bits or 'n/a'}.",
        f"- {_drawdown_concentration_note(attrib)}",
        f"- Trade-level coverage: {trade_status}. Turnover quality: `{turnover_info.get('quality', 'unknown')}`.",
        "- Limits: country attribution is additive, trade coverage can be partial, and execution-delay stress remains unavailable without true signal/position state.",
    ]


def _remaining_blockers_lines(
    *,
    trade_ledger_portfolio: pd.DataFrame,
    trade_concentration_extended: pd.DataFrame,
    turnover_info: dict[str, Any],
    execution_delay_stress: pd.DataFrame,
) -> list[str]:
    blockers: list[str] = []
    if turnover_info.get("quality") != "actual_positions":
        blockers.append("- No actual daily positions were found. Turnover remains a proxy reconstructed from `n_open_positions` capacity utilization.")
    if trade_ledger_portfolio.empty or "status" in trade_ledger_portfolio.columns and trade_ledger_portfolio["status"].astype(str).eq("not_available").all():
        blockers.append("- No exact consolidated trade ledger was found.")
    elif not trade_concentration_extended.empty and str(trade_concentration_extended.iloc[0].get("status", "")) == "partial":
        blockers.append("- Trade ledger coverage is partial, so trade concentration is not fully portfolio-complete.")
    if not execution_delay_stress.empty and str(execution_delay_stress.iloc[0].get("scenario", "")) == "not_available":
        blockers.append("- Execution delay stress is unavailable because true daily signal or position state was not found.")
    blockers.append("- Borrow costs are not modeled in this pack.")
    return blockers


def _format_weight_summary(weights: pd.Series) -> str:
    return ", ".join(f"{book}={value:.2%}" for book, value in weights.items())


def _fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.2%}"


def _fmt_number(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.2f}"
