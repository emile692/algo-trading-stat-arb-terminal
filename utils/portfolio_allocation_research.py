from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils.core_portfolio_daily import load_or_build_core_daily_returns


LOGGER = logging.getLogger("portfolio_allocation_research")
TRADING_DAYS = 252


@dataclass(frozen=True)
class AllocationResearchOptions:
    start: str
    end: str
    books: tuple[str, ...]
    allocation_methods: tuple[str, ...]
    rebalance_frequencies: tuple[str, ...]
    lookback_days: tuple[int, ...]
    weight_floor: float = 0.10
    weight_cap: float = 0.40
    compounding_mode: str = "both"
    lag_days: int = 1
    output_root: Path = Path("data/experiments")
    output_suffix: str | None = None
    smoke: bool = False
    config_path: Path = Path("config/core_portfolio_reference.json")
    daily_cache_dir: Path = Path("data/experiments/core_portfolio_reference_daily_cache")


@dataclass(frozen=True)
class AllocationConfig:
    config_id: str
    method: str
    lookback_days: int
    rebalance_frequency: str
    weight_floor: float
    weight_cap: float
    constraint_label: str


def build_output_dir(options: AllocationResearchOptions) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    books_label = "_".join(options.books)
    name = f"portfolio_allocation_research_{books_label}_{stamp}"
    if options.smoke:
        name += "_smoke"
    if options.output_suffix:
        name += f"_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_book_daily_returns(options: AllocationResearchOptions, *, project_root: Path) -> pd.DataFrame:
    config_path = _resolve_path(project_root, options.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Core portfolio config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    daily_cache_dir = _resolve_path(project_root, options.daily_cache_dir)
    _daily_long, wide = load_or_build_core_daily_returns(
        config,
        root=project_root,
        cache_dir=daily_cache_dir,
        rebuild=False,
    )
    if wide.empty:
        raise RuntimeError("No daily book returns could be loaded or built.")

    requested = list(options.books)
    missing = [book for book in requested if book not in wide.columns]
    if missing:
        raise ValueError(f"Requested books not available in daily returns: {missing}. Available={list(wide.columns)}")

    out = wide.loc[:, requested].copy()
    out.index = pd.to_datetime(out.index, errors="coerce").normalize()
    out = out[~out.index.isna()].sort_index()
    out = out.loc[(out.index >= pd.Timestamp(options.start)) & (out.index <= pd.Timestamp(options.end))]
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if out.empty:
        raise RuntimeError("Daily book return matrix is empty after date/book filtering.")
    return out


def build_allocation_configs(options: AllocationResearchOptions) -> list[AllocationConfig]:
    constraints = [("unconstrained", 0.0, 1.0)]
    if options.weight_floor > 0.0 or options.weight_cap < 1.0:
        constraints.append(("floor_cap", float(options.weight_floor), float(options.weight_cap)))

    configs: list[AllocationConfig] = []
    for method in options.allocation_methods:
        for lookback in options.lookback_days:
            for freq in options.rebalance_frequencies:
                for label, floor, cap in constraints:
                    config_id = f"{method}__lb{int(lookback)}__{freq}__{label}"
                    configs.append(
                        AllocationConfig(
                            config_id=config_id,
                            method=method,
                            lookback_days=int(lookback),
                            rebalance_frequency=freq,
                            weight_floor=float(floor),
                            weight_cap=float(cap),
                            constraint_label=label,
                        )
                    )
    return configs


def walk_forward_backtest_allocations(
    returns: pd.DataFrame,
    configs: list[AllocationConfig],
    *,
    lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    portfolio_returns: list[pd.DataFrame] = []
    weights_frames: list[pd.DataFrame] = []
    turnover_frames: list[pd.DataFrame] = []
    rebalance_log_frames: list[pd.DataFrame] = []

    for cfg in configs:
        LOGGER.info("Running allocation config=%s", cfg.config_id)
        result = _walk_forward_single(returns, cfg, lag_days=lag_days)
        portfolio_returns.append(result["portfolio_daily_returns"])
        weights_frames.append(result["weights_timeseries"])
        turnover_frames.append(result["turnover"])
        rebalance_log_frames.append(result["rebalance_log"])

    return {
        "portfolio_daily_returns": pd.concat(portfolio_returns, ignore_index=True, sort=False),
        "weights_timeseries": pd.concat(weights_frames, ignore_index=True, sort=False),
        "turnover": pd.concat(turnover_frames, ignore_index=True, sort=False),
        "rebalance_log": pd.concat(rebalance_log_frames, ignore_index=True, sort=False),
    }


def _walk_forward_single(returns: pd.DataFrame, cfg: AllocationConfig, *, lag_days: int) -> dict[str, pd.DataFrame]:
    dates = pd.DatetimeIndex(returns.index).sort_values()
    rebalance_dates = _eligible_rebalance_dates(dates, cfg.rebalance_frequency)
    date_pos = {dt: i for i, dt in enumerate(dates)}
    lag = max(0, int(lag_days))

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    log_rows: list[dict[str, Any]] = []
    prev_weights: pd.Series | None = None
    turnover_by_rebalance: dict[pd.Timestamp, float] = {}

    eligible: list[pd.Timestamp] = []
    for dt in rebalance_dates:
        pos = date_pos[pd.Timestamp(dt)]
        hist_end_pos = pos - lag
        hist_start_pos = hist_end_pos - int(cfg.lookback_days) + 1
        if hist_start_pos < 0 or hist_end_pos < 0:
            continue
        eligible.append(pd.Timestamp(dt))

    if not eligible:
        raise RuntimeError(f"No eligible rebalance dates for {cfg.config_id}; lower lookback or extend date range.")

    for i, dt in enumerate(eligible):
        pos = date_pos[dt]
        hist_end_pos = pos - lag
        hist_start_pos = hist_end_pos - int(cfg.lookback_days) + 1
        hist = returns.iloc[hist_start_pos : hist_end_pos + 1].copy()
        raw = compute_allocation_weights(hist, cfg.method)
        weights = apply_floor_cap(raw, floor=cfg.weight_floor, cap=cfg.weight_cap)
        weights = weights.reindex(returns.columns).fillna(0.0)
        if prev_weights is None:
            turnover = 0.0
        else:
            turnover = float(0.5 * (weights - prev_weights).abs().sum())
        prev_weights = weights.copy()
        turnover_by_rebalance[dt] = turnover

        next_dt = eligible[i + 1] if i + 1 < len(eligible) else None
        apply_end_pos = date_pos[next_dt] - 1 if next_dt is not None else len(dates) - 1
        for apply_dt in dates[pos : apply_end_pos + 1]:
            weights_by_date[pd.Timestamp(apply_dt)] = weights

        log_rows.append(
            {
                "config_id": cfg.config_id,
                "method": cfg.method,
                "lookback_days": cfg.lookback_days,
                "rebalance_frequency": cfg.rebalance_frequency,
                "constraint_label": cfg.constraint_label,
                "weight_floor": cfg.weight_floor,
                "weight_cap": cfg.weight_cap,
                "rebalance_date": dt,
                "estimation_start": dates[hist_start_pos],
                "estimation_end": dates[hist_end_pos],
                "application_start": dt,
                "application_end": dates[apply_end_pos],
                "turnover": turnover,
                **{f"weight_{book}": float(weights[book]) for book in returns.columns},
            }
        )

    applied_dates = pd.DatetimeIndex(sorted(weights_by_date))
    weights_matrix = pd.DataFrame([weights_by_date[dt] for dt in applied_dates], index=applied_dates)
    weights_matrix = weights_matrix.reindex(columns=returns.columns).fillna(0.0)
    aligned_returns = returns.loc[applied_dates]
    port = aligned_returns.mul(weights_matrix, axis=0).sum(axis=1)

    weights_out = weights_matrix.reset_index().rename(columns={"index": "date"})
    weights_out.insert(0, "config_id", cfg.config_id)
    weights_out.insert(1, "method", cfg.method)
    weights_out.insert(2, "lookback_days", cfg.lookback_days)
    weights_out.insert(3, "rebalance_frequency", cfg.rebalance_frequency)
    weights_out.insert(4, "constraint_label", cfg.constraint_label)
    weights_out.insert(5, "weight_floor", cfg.weight_floor)
    weights_out.insert(6, "weight_cap", cfg.weight_cap)
    weights_out = weights_out.rename(columns={book: f"weight_{book}" for book in returns.columns})

    turnover_series = pd.Series(0.0, index=applied_dates, name="turnover")
    for dt, value in turnover_by_rebalance.items():
        if dt in turnover_series.index:
            turnover_series.loc[dt] = value
    turnover_out = turnover_series.reset_index().rename(columns={"index": "date"})
    turnover_out.insert(0, "config_id", cfg.config_id)
    turnover_out.insert(1, "method", cfg.method)
    turnover_out.insert(2, "lookback_days", cfg.lookback_days)
    turnover_out.insert(3, "rebalance_frequency", cfg.rebalance_frequency)
    turnover_out.insert(4, "constraint_label", cfg.constraint_label)

    port_out = port.reset_index().rename(columns={"index": "date", 0: "portfolio_daily_return"})
    port_out.insert(0, "config_id", cfg.config_id)
    port_out.insert(1, "method", cfg.method)
    port_out.insert(2, "lookback_days", cfg.lookback_days)
    port_out.insert(3, "rebalance_frequency", cfg.rebalance_frequency)
    port_out.insert(4, "constraint_label", cfg.constraint_label)
    port_out.insert(5, "weight_floor", cfg.weight_floor)
    port_out.insert(6, "weight_cap", cfg.weight_cap)
    if "portfolio_daily_return" not in port_out.columns:
        port_out = port_out.rename(columns={port_out.columns[-1]: "portfolio_daily_return"})

    return {
        "portfolio_daily_returns": port_out,
        "weights_timeseries": weights_out,
        "turnover": turnover_out,
        "rebalance_log": pd.DataFrame(log_rows),
    }


def compute_allocation_weights(hist: pd.DataFrame, method: str) -> pd.Series:
    clean = hist.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    method = str(method).strip().lower()
    if method == "equal_weight":
        return _equal_weights(clean.columns)
    if method == "inverse_vol":
        return _inverse_vol_weights(clean)
    if method == "risk_parity":
        return _risk_parity_weights(clean)
    if method == "mean_variance_shrunk":
        return _mean_variance_shrunk_weights(clean)
    if method == "reward_to_risk":
        return _reward_to_risk_weights(clean)
    if method == "contribution_based":
        return _contribution_based_weights(clean)
    raise ValueError(f"Unknown allocation method: {method}")


def apply_floor_cap(weights: pd.Series, *, floor: float = 0.0, cap: float = 1.0) -> pd.Series:
    w = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = w.clip(lower=0.0)
    if float(w.sum()) <= 0.0:
        w = _equal_weights(w.index)
    else:
        w = w / float(w.sum())

    floor = max(0.0, float(floor))
    cap = min(1.0, float(cap))
    n = len(w)
    if n == 0:
        return w
    if floor * n > 1.0 + 1e-12:
        raise ValueError(f"weight_floor={floor} infeasible for n={n}")
    if cap * n < 1.0 - 1e-12:
        raise ValueError(f"weight_cap={cap} infeasible for n={n}")
    if floor <= 0.0 and cap >= 1.0:
        return w

    out = pd.Series(floor, index=w.index, dtype=float)
    remaining = 1.0 - floor * n
    active = list(w.index)
    scores = w.copy()

    while active and remaining > 1e-12:
        active_scores = scores.loc[active].clip(lower=0.0)
        if float(active_scores.sum()) <= 0.0:
            active_scores = pd.Series(1.0, index=active)
        alloc = remaining * active_scores / float(active_scores.sum())
        proposed = out.loc[active] + alloc
        over = proposed > cap + 1e-12
        if not bool(over.any()):
            out.loc[active] = proposed
            remaining = 0.0
            break
        capped = list(proposed.index[over])
        out.loc[capped] = cap
        remaining = 1.0 - float(out.sum())
        active = [idx for idx in active if idx not in capped]

    if remaining > 1e-10 and active:
        out.loc[active] += remaining / len(active)
    out = out.clip(lower=floor, upper=cap)
    return out / float(out.sum())


def compute_portfolio_outputs(
    returns: pd.DataFrame,
    portfolio_daily_returns: pd.DataFrame,
    weights_timeseries: pd.DataFrame,
    turnover: pd.DataFrame,
    configs: list[AllocationConfig],
    *,
    compounding_modes: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    daily = portfolio_daily_returns.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()

    weights = weights_timeseries.copy()
    weights["date"] = pd.to_datetime(weights["date"], errors="coerce").dt.normalize()
    book_cols = list(returns.columns)
    weight_cols = [f"weight_{book}" for book in book_cols]

    config_meta = pd.DataFrame([asdict(cfg) for cfg in configs])
    equity_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    drawdown_frames: list[pd.DataFrame] = []

    turnover_summary = _turnover_summary(turnover)
    weight_summary = _weight_summary(weights, book_cols)
    contribution_summary = compute_book_contributions(returns, daily, weights, configs)

    for config_id, group in daily.groupby("config_id", sort=False):
        r = group.set_index("date")["portfolio_daily_return"].astype(float).sort_index()
        books_oos = returns.loc[r.index, book_cols]
        for mode in compounding_modes:
            equity = equity_from_returns(r, mode)
            drawdown = drawdown_from_equity(equity)
            eq_frame = pd.DataFrame(
                {
                    "date": equity.index,
                    "config_id": config_id,
                    "compounding_mode": mode,
                    "equity": equity.values,
                    "drawdown": drawdown.values,
                }
            )
            equity_frames.append(eq_frame)
            metrics = compute_portfolio_metrics(r, mode=mode)
            metrics.update(_portfolio_book_correlations(r, books_oos))
            run_rows.append({"config_id": config_id, "compounding_mode": mode, **metrics})
            yearly_frames.append(_period_metrics(r, mode=mode, freq="Y", label_col="year", config_id=config_id))
            monthly_frames.append(_monthly_returns(r, mode=mode, config_id=config_id))
            drawdown_frames.append(_drawdown_by_period(r, mode=mode, config_id=config_id))

    run_level = pd.DataFrame(run_rows)
    run_level = run_level.merge(config_meta, on="config_id", how="left")
    run_level = run_level.merge(turnover_summary, on="config_id", how="left")
    run_level = run_level.merge(weight_summary, on="config_id", how="left")
    run_level = _add_relative_vs_inverse_vol(run_level)

    summary = _allocation_summary(run_level)
    ranking = _allocation_ranking(run_level)

    return {
        "allocation_run_level": run_level,
        "allocation_summary": summary,
        "allocation_ranking": ranking,
        "portfolio_equity_curves": pd.concat(equity_frames, ignore_index=True, sort=False),
        "book_contribution_summary": contribution_summary,
        "yearly_metrics": pd.concat(yearly_frames, ignore_index=True, sort=False) if yearly_frames else pd.DataFrame(),
        "monthly_returns": pd.concat(monthly_frames, ignore_index=True, sort=False) if monthly_frames else pd.DataFrame(),
        "drawdown_table": pd.concat(drawdown_frames, ignore_index=True, sort=False) if drawdown_frames else pd.DataFrame(),
    }


def compute_portfolio_metrics(returns: pd.Series, *, mode: str) -> dict[str, Any]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {}
    equity = equity_from_returns(r, mode)
    drawdown = drawdown_from_equity(equity)
    total_return = float(equity.iloc[-1] - 1.0)
    if mode == "compounded":
        ann_return = float(equity.iloc[-1] ** (TRADING_DAYS / len(r)) - 1.0) if equity.iloc[-1] > 0 else np.nan
    else:
        ann_return = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = float((r.mean() / downside_std) * np.sqrt(TRADING_DAYS)) if np.isfinite(downside_std) and downside_std > 0 else np.nan
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    return {
        "n_days": int(len(r)),
        "start": r.index.min(),
        "end": r.index.max(),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if np.isfinite(ann_return) and np.isfinite(max_dd) and max_dd < 0 else np.nan,
        "hit_ratio_daily": float((r > 0.0).mean()),
        "skew": float(r.skew()) if len(r) > 2 else np.nan,
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }


def compute_book_contributions(
    returns: pd.DataFrame,
    portfolio_daily_returns: pd.DataFrame,
    weights_timeseries: pd.DataFrame,
    configs: list[AllocationConfig],
) -> pd.DataFrame:
    book_cols = list(returns.columns)
    cfg_meta = pd.DataFrame([asdict(cfg) for cfg in configs])
    port = portfolio_daily_returns.copy()
    port["date"] = pd.to_datetime(port["date"], errors="coerce").dt.normalize()
    weights = weights_timeseries.copy()
    weights["date"] = pd.to_datetime(weights["date"], errors="coerce").dt.normalize()

    rows: list[dict[str, Any]] = []
    for config_id, w_group in weights.groupby("config_id", sort=False):
        idx = pd.DatetimeIndex(w_group["date"])
        book_rets = returns.loc[idx, book_cols]
        total_contrib = 0.0
        contrib_by_book: dict[str, float] = {}
        for book in book_cols:
            w = pd.to_numeric(w_group[f"weight_{book}"], errors="coerce").to_numpy(dtype=float)
            r = book_rets[book].to_numpy(dtype=float)
            contrib = pd.Series(w * r, index=idx)
            contrib_by_book[book] = float(contrib.sum())
            total_contrib += contrib_by_book[book]
        for book in book_cols:
            w_series = pd.to_numeric(w_group[f"weight_{book}"], errors="coerce")
            rows.append(
                {
                    "config_id": config_id,
                    "book": book,
                    "avg_weight": float(w_series.mean()),
                    "median_weight": float(w_series.median()),
                    "min_weight": float(w_series.min()),
                    "max_weight": float(w_series.max()),
                    "weight_std": float(w_series.std(ddof=1)) if len(w_series) > 1 else np.nan,
                    "cumulative_contribution": contrib_by_book[book],
                    "average_daily_contribution": contrib_by_book[book] / max(1, len(w_group)),
                    "contribution_share": contrib_by_book[book] / total_contrib if abs(total_contrib) > 1e-12 else np.nan,
                    "standalone_total_return": float((1.0 + returns.loc[idx, book]).prod() - 1.0),
                }
            )
    out = pd.DataFrame(rows)
    return out.merge(cfg_meta, on="config_id", how="left")


def make_allocation_plots(
    out_dir: Path,
    returns: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    *,
    top_n: int = 8,
) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ranking = outputs["allocation_ranking"].copy()
    top_configs = (
        ranking[ranking["compounding_mode"].eq("compounded")]
        .sort_values("robust_rank_score")
        .head(top_n)["config_id"]
        .tolist()
    )
    baseline_ids = [
        cid
        for cid in ranking["config_id"].astype(str).unique()
        if cid.startswith("inverse_vol") and cid not in top_configs
    ][:2]
    plot_configs = list(dict.fromkeys(top_configs + baseline_ids))
    if not plot_configs:
        return

    equity = outputs["portfolio_equity_curves"]
    eq_plot = equity[equity["config_id"].isin(plot_configs) & equity["compounding_mode"].eq("compounded")]
    _plot_lines(eq_plot, "date", "equity", "config_id", plots_dir / "equity_curves_compounded.png", "Compounded equity curves")
    _plot_lines(eq_plot, "date", "drawdown", "config_id", plots_dir / "drawdowns_compounded.png", "Compounded drawdowns")

    additive = equity[equity["config_id"].isin(plot_configs) & equity["compounding_mode"].eq("additive")]
    if not additive.empty:
        _plot_lines(additive, "date", "equity", "config_id", plots_dir / "equity_curves_additive.png", "Additive equity curves")

    weights = outputs.get("portfolio_weights_timeseries", pd.DataFrame())
    book_cols = list(returns.columns)
    for config_id in plot_configs[:3]:
        sub = weights[weights["config_id"].eq(config_id)].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 5))
        for book in book_cols:
            ax.plot(pd.to_datetime(sub["date"]), sub[f"weight_{book}"], label=book, linewidth=1.2)
        ax.set_title(f"Weights over time: {config_id}")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"weights_{_safe_filename(config_id)}.png", dpi=140)
        plt.close(fig)

    contrib = outputs["book_contribution_summary"]
    best_config = plot_configs[0]
    contrib_best = contrib[contrib["config_id"].eq(best_config)].copy()
    if not contrib_best.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        contrib_best.sort_values("cumulative_contribution").plot(
            x="book",
            y="cumulative_contribution",
            kind="barh",
            ax=ax,
            legend=False,
            title=f"Cumulative contribution by book: {best_config}",
        )
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "contribution_by_book_best.png", dpi=140)
        plt.close(fig)

    run = outputs["allocation_run_level"]
    turn = run[run["compounding_mode"].eq("compounded")].copy()
    if not turn.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        turn.sort_values("avg_rebalance_turnover").plot(
            x="config_id",
            y="avg_rebalance_turnover",
            kind="bar",
            ax=ax,
            legend=False,
            title="Average rebalance turnover by allocator",
        )
        ax.set_ylabel("Average turnover")
        ax.tick_params(axis="x", labelrotation=80)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "turnover_by_allocator.png", dpi=140)
        plt.close(fig)

    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Book daily return correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / "book_correlation_heatmap.png", dpi=140)
    plt.close(fig)

    if not contrib_best.empty:
        comp = contrib_best[["book", "avg_weight", "contribution_share"]].set_index("book")
        fig, ax = plt.subplots(figsize=(9, 4))
        comp.plot(kind="bar", ax=ax, title=f"Average weights vs contribution share: {best_config}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "avg_weights_vs_contributions_best.png", dpi=140)
        plt.close(fig)

    daily = outputs["portfolio_daily_returns"]
    rolling_rows = []
    for config_id in plot_configs:
        r = (
            daily[daily["config_id"].eq(config_id)]
            .set_index("date")["portfolio_daily_return"]
            .astype(float)
            .sort_index()
        )
        roll = rolling_sharpe(r, window=126)
        rolling_rows.append(pd.DataFrame({"date": roll.index, "config_id": config_id, "rolling_sharpe_126d": roll.values}))
    rolling = pd.concat(rolling_rows, ignore_index=True, sort=False)
    _plot_lines(rolling, "date", "rolling_sharpe_126d", "config_id", plots_dir / "rolling_sharpe_126d.png", "Rolling Sharpe 126d")


def write_outputs(
    out_dir: Path,
    returns: pd.DataFrame,
    wf_outputs: dict[str, pd.DataFrame],
    portfolio_outputs: dict[str, pd.DataFrame],
    options: AllocationResearchOptions,
    configs: list[AllocationConfig],
    *,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outputs = {
        **wf_outputs,
        **portfolio_outputs,
        "correlation_matrix": returns.corr(),
    }
    outputs["portfolio_weights_timeseries"] = wf_outputs["weights_timeseries"]
    outputs["portfolio_turnover"] = wf_outputs["turnover"]

    file_map = {
        "allocation_run_level": "allocation_run_level.csv",
        "allocation_summary": "allocation_summary.csv",
        "allocation_ranking": "allocation_ranking.csv",
        "portfolio_daily_returns": "portfolio_daily_returns.csv",
        "portfolio_equity_curves": "portfolio_equity_curves.csv",
        "portfolio_weights_timeseries": "portfolio_weights_timeseries.csv",
        "portfolio_turnover": "portfolio_turnover.csv",
        "book_contribution_summary": "book_contribution_summary.csv",
        "yearly_metrics": "yearly_metrics.csv",
        "monthly_returns": "monthly_returns.csv",
        "correlation_matrix": "correlation_matrix.csv",
        "drawdown_table": "drawdown_table.csv",
    }
    for key, filename in file_map.items():
        frame = outputs.get(key, pd.DataFrame())
        path = out_dir / filename
        frame.to_csv(path, index=(key == "correlation_matrix"))

    best = best_config_payload(outputs["allocation_ranking"], outputs["allocation_run_level"], outputs["book_contribution_summary"])
    (out_dir / "best_config.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    metadata = {
        "options": _options_to_jsonable(options),
        "configs": [asdict(cfg) for cfg in configs],
        "input_books": list(returns.columns),
        "input_start": str(returns.index.min().date()),
        "input_end": str(returns.index.max().date()),
        "nb_input_days": int(len(returns)),
        "assumptions": [
            "Book inputs are daily returns from the frozen core country books.",
            "At each rebalance date, weights use only the trailing lookback window ending lag_days before application.",
            "No whole-history optimization is used for walk-forward portfolio returns.",
            "Missing daily book returns are treated as zero only after calendar alignment.",
            "Mean-variance uses strong expected-return and covariance shrinkage and is long-only.",
            "Contribution-based weights are shrunk toward inverse-vol to avoid pure recent-winner chasing.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    conclusion = write_conclusion(out_dir, outputs, best)
    make_allocation_plots(out_dir, returns, outputs)
    return {"best_config": best, "conclusion": conclusion, "outputs": outputs}


def write_conclusion(out_dir: Path, outputs: dict[str, pd.DataFrame], best: dict[str, Any]) -> str:
    run = outputs["allocation_run_level"].copy()
    ranking = outputs["allocation_ranking"].copy()
    contrib = outputs["book_contribution_summary"].copy()
    compounded = run[run["compounding_mode"].eq("compounded")].copy()
    baseline = _select_inverse_vol_baseline(compounded)

    best_sharpe = compounded.sort_values("sharpe", ascending=False).iloc[0]
    best_return = compounded.sort_values("ann_return", ascending=False).iloc[0]
    best_calmar = compounded.sort_values("calmar", ascending=False).iloc[0]
    best_dd = compounded.sort_values("max_drawdown", ascending=False).iloc[0]
    stable = compounded.sort_values(["avg_rebalance_turnover", "avg_weight_std"], ascending=True).iloc[0]
    production = ranking[ranking["compounding_mode"].eq("compounded")].sort_values("robust_rank_score").iloc[0]
    conservative = _select_conservative_candidate(compounded, baseline)

    baseline_id = str(baseline["config_id"]) if baseline is not None else ""
    prod_id = str(production["config_id"])
    conservative_id = str(conservative["config_id"]) if conservative is not None else ""
    base_contrib = contrib[contrib["config_id"].eq(baseline_id)]
    prod_contrib = contrib[contrib["config_id"].eq(prod_id)]

    germany_msg = _book_reallocation_message("germany", base_contrib, prod_contrib)
    netherlands_msg = _book_reallocation_message("netherlands", base_contrib, prod_contrib, reduce_bias=True)
    overfit_msg = _overfit_diagnostic(compounded)
    diversification_msg = _diversification_diagnostic(baseline, production, base_contrib, prod_contrib)

    if baseline is not None:
        delta_sharpe = float(production["sharpe"]) - float(baseline["sharpe"])
        delta_return = float(production["ann_return"]) - float(baseline["ann_return"])
        delta_dd = float(production["max_drawdown"]) - float(baseline["max_drawdown"])
    else:
        delta_sharpe = delta_return = delta_dd = np.nan

    prod_worsens_dd = np.isfinite(delta_dd) and delta_dd < -0.005
    if conservative is not None and (
        prod_worsens_dd
        or not str(prod_id).startswith("inverse_vol")
        or float(production.get("avg_rebalance_turnover", np.inf)) > float(conservative.get("avg_rebalance_turnover", np.inf)) + 0.05
    ):
        recommendation = (
            f"Do not switch directly to the headline winner {prod_id}. "
            f"Use {conservative_id} as the conservative shadow-production candidate, with inverse_vol kept as the control benchmark."
        )
    elif prod_id.startswith("inverse_vol") and abs(delta_sharpe) < 0.10:
        recommendation = "Keep inverse_vol as production baseline; caps/floors are useful monitoring guardrails but not a mandatory allocator switch."
    elif "floor_cap" in prod_id and np.isfinite(delta_sharpe) and delta_sharpe > 0.05:
        recommendation = "Prefer the capped/floored allocator candidate for shadow production, with inverse_vol kept as the control benchmark."
    elif np.isfinite(delta_sharpe) and delta_sharpe > 0.10 and float(production.get("avg_rebalance_turnover", np.inf)) < 0.25:
        recommendation = "Promote the winning allocator to a shadow run; do not replace inverse_vol until it survives a fresh out-of-sample window."
    else:
        recommendation = "Keep inverse_vol; alternatives are not convincing enough after turnover, stability and robustness penalties."

    lines = [
        "Portfolio allocation research conclusion",
        "",
        "Decision summary",
        f"- Inverse-vol baseline: {baseline_id or 'not found'}",
        f"- Best Sharpe: {best_sharpe['config_id']} | Sharpe={best_sharpe['sharpe']:.3f} | ann_return={best_sharpe['ann_return']:.3%} | maxDD={best_sharpe['max_drawdown']:.3%}",
        f"- Best annual return: {best_return['config_id']} | ann_return={best_return['ann_return']:.3%} | Sharpe={best_return['sharpe']:.3f}",
        f"- Best Calmar: {best_calmar['config_id']} | Calmar={best_calmar['calmar']:.3f} | maxDD={best_calmar['max_drawdown']:.3%}",
        f"- Best max drawdown: {best_dd['config_id']} | maxDD={best_dd['max_drawdown']:.3%} | ann_return={best_dd['ann_return']:.3%}",
        f"- Most stable weights: {stable['config_id']} | avg turnover={stable['avg_rebalance_turnover']:.3f} | avg weight std={stable['avg_weight_std']:.3f}",
        f"- Production-credible ranked candidate: {prod_id} | robust_score={production['robust_rank_score']:.2f}",
        f"- Conservative guardrail candidate: {conservative_id or 'not found'}",
        "",
        "Relative to inverse_vol",
        f"- Delta Sharpe vs baseline: {delta_sharpe:.3f}",
        f"- Delta ann_return vs baseline: {delta_return:.3%}",
        f"- Delta maxDD vs baseline: {delta_dd:.3%} (positive means less severe drawdown)",
        "",
        "Book allocation read-through",
        f"- Germany: {germany_msg}",
        f"- Netherlands: {netherlands_msg}",
        "",
        "Robustness and overfit checks",
        f"- {overfit_msg}",
        f"- {diversification_msg}",
        "",
        "Final recommendation",
        f"- {recommendation}",
        "",
        "Files to inspect first",
        "- allocation_ranking.csv",
        "- allocation_run_level.csv",
        "- book_contribution_summary.csv",
        "- plots/equity_curves_compounded.png",
        "- plots/weights_*.png",
    ]
    text = "\n".join(lines) + "\n"
    (out_dir / "conclusion.txt").write_text(text, encoding="utf-8")
    return text


def best_config_payload(ranking: pd.DataFrame, run: pd.DataFrame, contrib: pd.DataFrame) -> dict[str, Any]:
    compounded = ranking[ranking["compounding_mode"].eq("compounded")].sort_values("robust_rank_score")
    if compounded.empty:
        return {}
    best_row = compounded.iloc[0].to_dict()
    config_id = str(best_row["config_id"])
    run_row = run[(run["config_id"].eq(config_id)) & (run["compounding_mode"].eq("compounded"))]
    payload = {"selected_by": "lowest_robust_rank_score", "ranking_row": best_row}
    if not run_row.empty:
        payload["run_level_row"] = run_row.iloc[0].to_dict()
    payload["book_contributions"] = contrib[contrib["config_id"].eq(config_id)].to_dict(orient="records")
    return payload


def equity_from_returns(returns: pd.Series, mode: str) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if mode == "compounded":
        return (1.0 + r).cumprod()
    if mode == "additive":
        return 1.0 + r.cumsum()
    raise ValueError(f"Unsupported compounding mode: {mode}")


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    return eq / eq.cummax() - 1.0


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    mu = r.rolling(window, min_periods=max(20, window // 3)).mean()
    vol = r.rolling(window, min_periods=max(20, window // 3)).std(ddof=1)
    return (mu / vol.replace(0.0, np.nan)) * np.sqrt(TRADING_DAYS)


def _eligible_rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    frequency = str(frequency).strip().lower()
    if frequency == "daily":
        return [pd.Timestamp(dt) for dt in dates]
    if frequency == "weekly":
        return [pd.Timestamp(group[0]) for _, group in pd.Series(dates, index=dates).groupby(dates.to_period("W-MON"))]
    if frequency == "monthly":
        return [pd.Timestamp(group[0]) for _, group in pd.Series(dates, index=dates).groupby(dates.to_period("M"))]
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def _equal_weights(columns: Iterable[str]) -> pd.Series:
    cols = list(columns)
    if not cols:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(cols), index=cols, dtype=float)


def _inverse_vol_weights(hist: pd.DataFrame) -> pd.Series:
    vol = hist.std(ddof=1).replace(0.0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if float(inv.sum()) <= 0.0:
        return _equal_weights(hist.columns)
    return inv / float(inv.sum())


def _risk_parity_weights(hist: pd.DataFrame) -> pd.Series:
    cov = _shrunk_covariance(hist, shrink_to_diag=0.65)
    cols = list(hist.columns)
    n = len(cols)
    x0 = _inverse_vol_weights(hist).reindex(cols).fillna(1.0 / n).to_numpy(dtype=float)
    try:
        from scipy.optimize import minimize

        def objective(w: np.ndarray) -> float:
            sigma_w = cov @ w
            port_var = float(w @ sigma_w)
            if port_var <= 1e-14:
                return 1e6
            rc = w * sigma_w / port_var
            return float(((rc - 1.0 / n) ** 2).sum())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = minimize(
                objective,
                x0=x0,
                method="SLSQP",
                bounds=[(1e-8, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
                options={"maxiter": 200, "ftol": 1e-12, "disp": False},
            )
        if result.success and np.all(np.isfinite(result.x)):
            w = pd.Series(result.x, index=cols)
            return w / float(w.sum())
    except Exception as exc:
        LOGGER.debug("Risk parity optimizer fallback to inverse_vol: %s", exc)
    return _inverse_vol_weights(hist)


def _mean_variance_shrunk_weights(hist: pd.DataFrame) -> pd.Series:
    cols = list(hist.columns)
    vol = hist.std(ddof=1).reindex(cols).fillna(0.0).to_numpy(dtype=float)
    active = np.isfinite(vol) & (vol > 1e-10)
    if not bool(active.any()):
        return _equal_weights(cols)
    cov = _shrunk_covariance(hist, shrink_to_diag=0.70)
    mu = hist.mean().reindex(cols).fillna(0.0).to_numpy(dtype=float)
    mu_anchor = float(np.nanmedian(mu)) if np.isfinite(mu).any() else 0.0
    mu_shrunk = 0.20 * mu + 0.80 * mu_anchor
    try:
        raw = np.linalg.pinv(cov) @ mu_shrunk
    except np.linalg.LinAlgError:
        return _inverse_vol_weights(hist)
    raw = np.asarray(raw, dtype=float)
    raw[~active] = 0.0
    raw = np.clip(raw, 0.0, np.nanpercentile(raw[np.isfinite(raw)], 90) if np.isfinite(raw).any() else 0.0)
    if not np.isfinite(raw).all() or float(raw.sum()) <= 1e-12:
        return _inverse_vol_weights(hist)
    mv = pd.Series(raw / float(raw.sum()), index=cols)
    # Strongly shrink the optimizer toward inverse vol; expected returns are noisy with four books.
    return 0.50 * _inverse_vol_weights(hist).reindex(cols).fillna(0.0) + 0.50 * mv


def _reward_to_risk_weights(hist: pd.DataFrame) -> pd.Series:
    vol = hist.std(ddof=1).replace(0.0, np.nan)
    score = hist.mean() / vol
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if float(score.sum()) <= 0.0:
        return _inverse_vol_weights(hist)
    target = score / float(score.sum())
    return 0.50 * _equal_weights(hist.columns) + 0.50 * target


def _contribution_based_weights(hist: pd.DataFrame) -> pd.Series:
    base = _inverse_vol_weights(hist)
    contribution = hist.mul(base, axis=1).sum(axis=0)
    score = contribution.clip(lower=0.0)
    if float(score.sum()) <= 0.0:
        return base
    target = score / float(score.sum())
    return 0.55 * base + 0.45 * target


def _shrunk_covariance(hist: pd.DataFrame, *, shrink_to_diag: float = 0.60) -> np.ndarray:
    data = hist.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sample = data.cov().to_numpy(dtype=float)
    diag = np.diag(np.diag(sample))
    shrink = float(np.clip(shrink_to_diag, 0.0, 1.0))
    cov = (1.0 - shrink) * sample + shrink * diag
    avg_var = float(np.nanmean(np.diag(cov))) if cov.size else 1.0
    ridge = max(avg_var, 1e-8) * 1e-4
    return cov + np.eye(cov.shape[0]) * ridge


def _turnover_summary(turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, group in turnover.groupby("config_id", sort=False):
        t = pd.to_numeric(group["turnover"], errors="coerce").fillna(0.0)
        rebal = t[t > 0.0]
        rows.append(
            {
                "config_id": config_id,
                "avg_daily_turnover": float(t.mean()),
                "avg_rebalance_turnover": float(rebal.mean()) if not rebal.empty else 0.0,
                "cumulative_turnover": float(t.sum()),
                "nb_rebalances_with_turnover": int((t > 0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _weight_summary(weights: pd.DataFrame, books: list[str]) -> pd.DataFrame:
    rows = []
    for config_id, group in weights.groupby("config_id", sort=False):
        hhi = np.zeros(len(group), dtype=float)
        book_stds = []
        max_weights = []
        for book in books:
            w = pd.to_numeric(group[f"weight_{book}"], errors="coerce").fillna(0.0)
            hhi += w.to_numpy(dtype=float) ** 2
            book_stds.append(float(w.std(ddof=1)) if len(w) > 1 else 0.0)
            max_weights.append(float(w.mean()))
        rows.append(
            {
                "config_id": config_id,
                "avg_weight_hhi": float(np.mean(hhi)),
                "avg_effective_n_books": float(np.mean(1.0 / np.maximum(hhi, 1e-12))),
                "avg_weight_std": float(np.mean(book_stds)),
                "avg_max_book_weight": float(max(max_weights)) if max_weights else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _portfolio_book_correlations(portfolio_returns: pd.Series, book_returns: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    values = []
    for book in book_returns.columns:
        corr = float(portfolio_returns.corr(book_returns[book])) if len(portfolio_returns) > 2 else np.nan
        out[f"corr_portfolio_{book}"] = corr
        if np.isfinite(corr):
            values.append(corr)
    out["avg_corr_portfolio_books"] = float(np.mean(values)) if values else np.nan
    return out


def _period_metrics(returns: pd.Series, *, mode: str, freq: str, label_col: str, config_id: str) -> pd.DataFrame:
    rows = []
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    for period, group in r.groupby(r.index.to_period(freq)):
        metrics = compute_portfolio_metrics(group, mode=mode)
        rows.append({"config_id": config_id, "compounding_mode": mode, label_col: str(period), **metrics})
    return pd.DataFrame(rows)


def _monthly_returns(returns: pd.Series, *, mode: str, config_id: str) -> pd.DataFrame:
    rows = []
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    for period, group in r.groupby(r.index.to_period("M")):
        if mode == "compounded":
            value = float((1.0 + group).prod() - 1.0)
        else:
            value = float(group.sum())
        rows.append({"config_id": config_id, "compounding_mode": mode, "trade_month": period.to_timestamp(), "month_return": value})
    return pd.DataFrame(rows)


def _drawdown_by_period(returns: pd.Series, *, mode: str, config_id: str) -> pd.DataFrame:
    rows = []
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    full_eq = equity_from_returns(r, mode)
    full_dd = drawdown_from_equity(full_eq)
    rows.append(
        {
            "config_id": config_id,
            "compounding_mode": mode,
            "period": "full",
            "max_drawdown": float(full_dd.min()),
            "dd_end": full_dd.idxmin(),
        }
    )
    for year, group in r.groupby(r.index.year):
        eq = equity_from_returns(group, mode)
        dd = drawdown_from_equity(eq)
        rows.append(
            {
                "config_id": config_id,
                "compounding_mode": mode,
                "period": str(year),
                "max_drawdown": float(dd.min()),
                "dd_end": dd.idxmin(),
            }
        )
    return pd.DataFrame(rows)


def _add_relative_vs_inverse_vol(run: pd.DataFrame) -> pd.DataFrame:
    out = run.copy()
    for col in ["delta_sharpe_vs_inverse_vol", "delta_ann_return_vs_inverse_vol", "delta_max_dd_vs_inverse_vol"]:
        out[col] = np.nan
    for mode, group in out.groupby("compounding_mode"):
        baseline = _select_inverse_vol_baseline(group)
        if baseline is None:
            continue
        idx = out["compounding_mode"].eq(mode)
        out.loc[idx, "delta_sharpe_vs_inverse_vol"] = pd.to_numeric(out.loc[idx, "sharpe"], errors="coerce") - float(baseline["sharpe"])
        out.loc[idx, "delta_ann_return_vs_inverse_vol"] = pd.to_numeric(out.loc[idx, "ann_return"], errors="coerce") - float(baseline["ann_return"])
        out.loc[idx, "delta_max_dd_vs_inverse_vol"] = pd.to_numeric(out.loc[idx, "max_drawdown"], errors="coerce") - float(baseline["max_drawdown"])
    return out


def _select_inverse_vol_baseline(run: pd.DataFrame) -> pd.Series | None:
    candidates = run[run["method"].astype(str).eq("inverse_vol")].copy()
    if candidates.empty:
        return None
    unconstrained = candidates[candidates["constraint_label"].astype(str).eq("unconstrained")]
    if not unconstrained.empty:
        candidates = unconstrained
    return candidates.sort_values(["lookback_days", "rebalance_frequency"]).iloc[0]


def _select_conservative_candidate(run: pd.DataFrame, baseline: pd.Series | None) -> pd.Series | None:
    candidates = run[run["constraint_label"].astype(str).eq("floor_cap")].copy()
    if candidates.empty:
        return None
    if baseline is not None:
        candidates = candidates[
            (pd.to_numeric(candidates["sharpe"], errors="coerce") >= float(baseline["sharpe"]) - 0.02)
            & (pd.to_numeric(candidates["max_drawdown"], errors="coerce") >= float(baseline["max_drawdown"]) - 0.0025)
        ].copy()
    if candidates.empty:
        candidates = run[run["constraint_label"].astype(str).eq("floor_cap")].copy()
    candidates["conservative_score"] = (
        candidates["sharpe"].rank(ascending=False, method="min") * 0.35
        + candidates["calmar"].rank(ascending=False, method="min") * 0.25
        + candidates["max_drawdown"].rank(ascending=False, method="min") * 0.20
        + candidates["avg_rebalance_turnover"].rank(ascending=True, method="min") * 0.20
    )
    return candidates.sort_values("conservative_score").iloc[0]


def _allocation_summary(run: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run.groupby(["method", "constraint_label", "compounding_mode"], dropna=False)
        .agg(
            n_configs=("config_id", "nunique"),
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            min_sharpe=("sharpe", "min"),
            max_sharpe=("sharpe", "max"),
            mean_ann_return=("ann_return", "mean"),
            mean_ann_vol=("ann_vol", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_calmar=("calmar", "mean"),
            mean_turnover=("avg_rebalance_turnover", "mean"),
            mean_weight_hhi=("avg_weight_hhi", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["compounding_mode", "mean_sharpe"], ascending=[True, False]).reset_index(drop=True)


def _allocation_ranking(run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in run.groupby("compounding_mode", sort=False):
        out = group.copy()
        out["rank_sharpe"] = out["sharpe"].rank(ascending=False, method="min")
        out["rank_ann_return"] = out["ann_return"].rank(ascending=False, method="min")
        out["rank_calmar"] = out["calmar"].rank(ascending=False, method="min")
        out["rank_max_drawdown"] = out["max_drawdown"].rank(ascending=False, method="min")
        out["rank_turnover"] = out["avg_rebalance_turnover"].rank(ascending=True, method="min")
        out["rank_weight_stability"] = out["avg_weight_std"].rank(ascending=True, method="min")
        out["robust_rank_score"] = (
            0.30 * out["rank_sharpe"]
            + 0.20 * out["rank_calmar"]
            + 0.15 * out["rank_ann_return"]
            + 0.15 * out["rank_max_drawdown"]
            + 0.10 * out["rank_turnover"]
            + 0.10 * out["rank_weight_stability"]
        )
        rows.append(out)
    ranked = pd.concat(rows, ignore_index=True, sort=False)
    return ranked.sort_values(["compounding_mode", "robust_rank_score"]).reset_index(drop=True)


def _plot_lines(df: pd.DataFrame, x: str, y: str, hue: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    for key, group in df.groupby(hue, sort=False):
        ax.plot(pd.to_datetime(group[x]), group[y], linewidth=1.2, label=str(key))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _book_reallocation_message(book: str, baseline: pd.DataFrame, candidate: pd.DataFrame, *, reduce_bias: bool = False) -> str:
    if baseline.empty or candidate.empty:
        return "not enough contribution data."
    b = baseline[baseline["book"].astype(str).eq(book)]
    c = candidate[candidate["book"].astype(str).eq(book)]
    if b.empty or c.empty:
        return "book not present in comparison."
    b_row = b.iloc[0]
    c_row = c.iloc[0]
    delta_weight = float(c_row["avg_weight"]) - float(b_row["avg_weight"])
    delta_contrib = float(c_row["contribution_share"]) - float(b_row["contribution_share"])
    if reduce_bias:
        if delta_weight < -0.03:
            return f"candidate reduces average weight by {delta_weight:.1%}; cap/shrink looks supported."
        if float(b_row["avg_weight"]) > float(b_row["contribution_share"]) + 0.05:
            return "baseline weight is above contribution share; keep cap/shrink under review."
        return "no strong evidence for an aggressive reduction."
    if delta_weight > 0.03 and delta_contrib >= -0.02:
        return f"candidate increases average weight by {delta_weight:.1%}; higher allocation is supported but should be shadow-tested."
    if float(b_row["contribution_share"]) > float(b_row["avg_weight"]) + 0.05:
        return "baseline contribution share exceeds average weight; a moderate increase deserves testing."
    return "no decisive evidence for a larger structural allocation."


def _overfit_diagnostic(compounded: pd.DataFrame) -> str:
    method_stats = (
        compounded.groupby("method")
        .agg(mean_sharpe=("sharpe", "mean"), std_sharpe=("sharpe", "std"), min_sharpe=("sharpe", "min"), max_sharpe=("sharpe", "max"))
        .reset_index()
    )
    unstable = method_stats[(method_stats["std_sharpe"] > 0.25) & ((method_stats["max_sharpe"] - method_stats["min_sharpe"]) > 0.50)]
    if unstable.empty:
        return "No allocator shows an extreme Sharpe dispersion across the tested lookbacks/frequencies."
    names = ", ".join(unstable["method"].astype(str).tolist())
    return f"Potential overfit sensitivity detected for: {names}; prefer conservative/shrunk variants over single-window winners."


def _diversification_diagnostic(
    baseline: pd.Series | None,
    production: pd.Series,
    baseline_contrib: pd.DataFrame,
    production_contrib: pd.DataFrame,
) -> str:
    if baseline is None or production_contrib.empty:
        return "Diversification read-through unavailable."
    hhi_delta = float(production["avg_weight_hhi"]) - float(baseline["avg_weight_hhi"])
    top_share = float(production_contrib["contribution_share"].abs().max()) if "contribution_share" in production_contrib else np.nan
    if hhi_delta > 0.04 or (np.isfinite(top_share) and top_share > 0.65):
        return "Gains appear partly driven by concentration; inspect contribution plots before production use."
    return "Gains do not appear to come solely from a single-book concentration jump."


def _safe_filename(value: str) -> str:
    keep = [c if c.isalnum() or c in {"-", "_"} else "_" for c in str(value)]
    return "".join(keep)[:180]


def _resolve_path(project_root: Path, path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root / p


def _options_to_jsonable(options: AllocationResearchOptions) -> dict[str, Any]:
    out = asdict(options)
    for key in ["output_root", "config_path", "daily_cache_dir"]:
        out[key] = str(out[key])
    out["books"] = list(options.books)
    out["allocation_methods"] = list(options.allocation_methods)
    out["rebalance_frequencies"] = list(options.rebalance_frequencies)
    out["lookback_days"] = list(options.lookback_days)
    return out


def compounding_modes_from_arg(value: str) -> tuple[str, ...]:
    v = str(value).strip().lower()
    if v == "both":
        return ("compounded", "additive")
    if v in {"compounded", "additive"}:
        return (v,)
    raise ValueError(f"Unsupported compounding mode: {value}")


def run_portfolio_allocation_research(options: AllocationResearchOptions, *, project_root: Path) -> Path:
    out_dir = build_output_dir(options)
    returns = load_book_daily_returns(options, project_root=project_root)
    configs = build_allocation_configs(options)
    LOGGER.info("Loaded daily returns shape=%s books=%s", returns.shape, list(returns.columns))
    LOGGER.info("Running %d allocation configs", len(configs))
    wf = walk_forward_backtest_allocations(returns, configs, lag_days=options.lag_days)
    modes = compounding_modes_from_arg(options.compounding_mode)
    portfolio_outputs = compute_portfolio_outputs(
        returns,
        wf["portfolio_daily_returns"],
        wf["weights_timeseries"],
        wf["turnover"],
        configs,
        compounding_modes=modes,
    )
    write_outputs(out_dir, returns, wf, portfolio_outputs, options, configs)
    LOGGER.info("Allocation research complete: %s", out_dir)
    return out_dir
