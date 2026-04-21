from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
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
from utils.germany_phase3 import fixed_pair_filter, run_phase3_specs_for_period


LOGGER = logging.getLogger("multibook_portfolio")
EXPERIMENTS_ROOT = PROJECT_ROOT / "data" / "experiments"


@dataclass(frozen=True)
class MultibookOptions:
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    output_root: Path = EXPERIMENTS_ROOT
    output_suffix: str | None = None
    smoke: bool = False


@dataclass(frozen=True)
class BookDefinition:
    book: str
    country: str
    config_name: str
    source_dir: Path
    monthly_file: str
    rationale: str
    simplicity: str
    maturity_status: str
    needs_germany_rerun: bool = False


def build_output_dir(options: MultibookOptions) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"multibook_portfolio_sweden_germany_france_netherlands_{stamp}"
    if options.smoke:
        name += "_smoke"
    if options.output_suffix:
        name += f"_{options.output_suffix}"
    out = Path(options.output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def default_books() -> list[BookDefinition]:
    return [
        BookDefinition(
            book="sweden",
            country="sweden",
            config_name="best_plus_regime_filter",
            source_dir=EXPERIMENTS_ROOT / "sweden_filter_robustness_20180101_20251231_20260419_095523",
            monthly_file="split_monthly_returns.csv",
            rationale="Advanced Sweden local C regime_filter: best Sharpe/DD/breadth compromise in robustness campaign.",
            simplicity="local simple regime filter",
            maturity_status="advanced_local_validated",
        ),
        BookDefinition(
            book="germany",
            country="germany",
            config_name="pair_filter_corr_abs_le_0p75",
            source_dir=EXPERIMENTS_ROOT / "germany_phase3_20180101_20251231_20260420_225845",
            monthly_file="generated_germany_monthly_returns.csv",
            rationale="Simple Germany local pair filter: exclude abs(6m_corr)>0.75. Mitigation is not used because it is not fully promoted.",
            simplicity="local simple pair filter",
            maturity_status="advanced_needs_stress_trending_validation",
            needs_germany_rerun=True,
        ),
        BookDefinition(
            book="france",
            country="france",
            config_name="reference",
            source_dir=EXPERIMENTS_ROOT / "country_research_france_20180101_20251231_20260419_210305",
            monthly_file="monthly_returns.csv",
            rationale="France qualified as top-pf baseline; transferred filters degraded portfolio metrics.",
            simplicity="baseline local reference",
            maturity_status="top_pf_qualified",
        ),
        BookDefinition(
            book="netherlands",
            country="netherlands",
            config_name="reference",
            source_dir=EXPERIMENTS_ROOT / "country_research_netherlands_20180101_20251231_20260420_192536",
            monthly_file="monthly_returns.csv",
            rationale="Netherlands qualified as simple baseline book; no forced ablation.",
            simplicity="baseline local reference",
            maturity_status="top_pf_qualified",
        ),
    ]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except Exception:
        return default


def max_drawdown_from_returns(returns: pd.Series) -> tuple[float, pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return np.nan, pd.NaT, pd.NaT
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    if pd.isna(trough):
        return np.nan, pd.NaT, pd.NaT
    peak_date = equity.loc[:trough].idxmax()
    return float(dd.loc[trough]), pd.Timestamp(peak_date), pd.Timestamp(trough)


def return_metrics(returns: pd.Series, periods_per_year: int = 12) -> dict[str, Any]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "n_months": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "positive_month_rate": np.nan,
            "worst_month": np.nan,
            "best_month": np.nan,
            "dd_start": pd.NaT,
            "dd_end": pd.NaT,
        }
    total = float((1.0 + r).prod() - 1.0)
    ann_return = float((1.0 + total) ** (periods_per_year / len(r)) - 1.0) if total > -1 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    dd, dd_start, dd_end = max_drawdown_from_returns(r)
    return {
        "n_months": int(len(r)),
        "total_return": total,
        "annualized_return": ann_return,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "positive_month_rate": float((r > 0).mean()),
        "worst_month": float(r.min()),
        "best_month": float(r.max()),
        "dd_start": dd_start,
        "dd_end": dd_end,
    }


def normalize_monthly_frame(df: pd.DataFrame, book: BookDefinition) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "trade_month" not in out.columns or "month_return" not in out.columns:
        return pd.DataFrame()
    out["trade_month"] = pd.to_datetime(out["trade_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["config_name"] = out.get("config_name", "").astype(str)
    out = out[out["config_name"] == book.config_name].copy()
    if out.empty:
        return pd.DataFrame()
    out = out[["trade_month", "month_return", "config_name"]].copy()
    out["month_return"] = pd.to_numeric(out["month_return"], errors="coerce")
    out["book"] = book.book
    out["country"] = book.country
    return out.dropna(subset=["trade_month", "month_return"]).sort_values("trade_month").reset_index(drop=True)


def load_book_monthly(book: BookDefinition, out_dir: Path, options: MultibookOptions) -> pd.DataFrame:
    if book.needs_germany_rerun:
        return build_germany_monthly(out_dir=out_dir, options=options, book=book)
    monthly_path = book.source_dir / book.monthly_file
    if not monthly_path.exists():
        raise FileNotFoundError(f"Monthly returns missing for {book.book}: {monthly_path}")
    return normalize_monthly_frame(pd.read_csv(monthly_path), book)


def build_germany_monthly(out_dir: Path, options: MultibookOptions, book: BookDefinition) -> pd.DataFrame:
    cache = out_dir / book.monthly_file
    if cache.exists():
        return normalize_monthly_frame(pd.read_csv(cache), book)

    reference = select_country_reference("germany", None)
    scans = load_or_build_country_scans(reference, start=options.start, end=options.end, rebuild=False)
    assets = build_country_assets("germany", scans)
    price_panel = load_price_panel(assets, BASE_DATA_PATH, start=options.start, end=options.end, buffer_days=520)
    market_features = compute_market_regime_features(price_panel)
    asset_metadata = pd.read_csv(ASSET_REGISTRY_PATH) if ASSET_REGISTRY_PATH.exists() else pd.DataFrame()
    thresholds = FilterThresholds(
        DEFAULT_ABS_Z_THRESHOLD,
        DEFAULT_ZSPEED_EWMA_THRESHOLD,
        GERMANY_BETA_THRESHOLD,
        PROJECT_ROOT,
    )
    frames = run_phase3_specs_for_period(
        period=PeriodSpec("full_period", "2018_2025", options.start, options.end, "full"),
        specs=[fixed_pair_filter()],
        reference=reference,
        scans=scans,
        thresholds=thresholds,
        market_features=market_features,
        price_panel=price_panel,
        asset_metadata=asset_metadata,
    )
    monthly = frames.get("monthly_returns", pd.DataFrame()).copy()
    monthly.to_csv(cache, index=False)
    return normalize_monthly_frame(monthly, book)


def align_book_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    wide = monthly.pivot_table(index="trade_month", columns="book", values="month_return", aggfunc="sum")
    wide = wide.sort_index()
    # Missing book returns are treated as zero allocation returns for that book/month.
    # This keeps the common calendar while avoiding look-ahead interpolation.
    return wide.fillna(0.0)


def book_definitions_frame(books: list[BookDefinition]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "book": b.book,
                "country": b.country,
                "config_name": b.config_name,
                "source_dir": str(b.source_dir),
                "rationale": b.rationale,
                "simplicity": b.simplicity,
                "maturity_status": b.maturity_status,
            }
            for b in books
        ]
    )


def weight_schemes_for_combo(returns: pd.DataFrame, combo: tuple[str, ...]) -> dict[str, dict[str, float]]:
    n = len(combo)
    schemes: dict[str, dict[str, float]] = {
        "equal_weight": {book: 1.0 / n for book in combo},
    }
    vol = returns.loc[:, list(combo)].std(ddof=1)
    inv = 1.0 / vol.replace(0, np.nan)
    if inv.notna().all() and inv.sum() > 0:
        w = inv / inv.sum()
        schemes["inverse_vol"] = {book: float(w[book]) for book in combo}
    return schemes


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = list(weights)
    w = pd.Series(weights, dtype=float)
    return returns.loc[:, cols].mul(w, axis=1).sum(axis=1)


def drawdown_contribution(returns: pd.DataFrame, weights: dict[str, float], dd_start: pd.Timestamp, dd_end: pd.Timestamp) -> dict[str, float]:
    if pd.isna(dd_start) or pd.isna(dd_end):
        return {book: np.nan for book in weights}
    window = returns.loc[(returns.index >= dd_start) & (returns.index <= dd_end), list(weights)].copy()
    out: dict[str, float] = {}
    for book, weight in weights.items():
        out[book] = float((window[book] * weight).sum()) if book in window else np.nan
    return out


def build_combination_outputs(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    books = list(returns.columns)
    performance_rows: list[dict[str, Any]] = []
    dd_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    detail_rows: list[dict[str, Any]] = []

    for k in range(2, len(books) + 1):
        for combo in itertools.combinations(books, k):
            schemes = weight_schemes_for_combo(returns, combo)
            for scheme_name, weights in schemes.items():
                pr = portfolio_returns(returns, weights)
                metrics = return_metrics(pr)
                combo_name = "+".join(combo)
                row = {
                    "combo_name": combo_name,
                    "n_books": k,
                    "weight_scheme": scheme_name,
                    **{f"weight_{book}": weights.get(book, 0.0) for book in books},
                    **metrics,
                }
                performance_rows.append(row)
                dd_rows.append(
                    {
                        "combo_name": combo_name,
                        "n_books": k,
                        "weight_scheme": scheme_name,
                        "max_drawdown": metrics["max_drawdown"],
                        "dd_start": metrics["dd_start"],
                        "dd_end": metrics["dd_end"],
                        "dd_months": int(len(pr.loc[(pr.index >= metrics["dd_start"]) & (pr.index <= metrics["dd_end"])]))
                        if pd.notna(metrics["dd_start"]) and pd.notna(metrics["dd_end"])
                        else 0,
                    }
                )
                contrib = drawdown_contribution(returns, weights, metrics["dd_start"], metrics["dd_end"])
                for book, value in contrib.items():
                    detail_rows.append(
                        {
                            "combo_name": combo_name,
                            "weight_scheme": scheme_name,
                            "book": book,
                            "weighted_return_during_max_dd": value,
                            "weight": weights.get(book, 0.0),
                            "dd_start": metrics["dd_start"],
                            "dd_end": metrics["dd_end"],
                        }
                    )
                m = pd.DataFrame(
                    {
                        "trade_month": pr.index,
                        "combo_name": combo_name,
                        "n_books": k,
                        "weight_scheme": scheme_name,
                        "portfolio_month_return": pr.values,
                    }
                )
                monthly_frames.append(m)

    dd = pd.DataFrame(dd_rows)
    details = pd.DataFrame(detail_rows)
    if not dd.empty and not details.empty:
        pivot = details.pivot_table(
            index=["combo_name", "weight_scheme"],
            columns="book",
            values="weighted_return_during_max_dd",
            aggfunc="sum",
        ).reset_index()
        pivot.columns = [f"dd_contrib_{c}" if c not in {"combo_name", "weight_scheme"} else c for c in pivot.columns]
        dd = dd.merge(pivot, on=["combo_name", "weight_scheme"], how="left")

    return (
        pd.DataFrame(performance_rows),
        dd,
        pd.concat(monthly_frames, ignore_index=True, sort=False) if monthly_frames else pd.DataFrame(),
    )


def build_book_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for book, group in monthly.groupby("book"):
        metrics = return_metrics(group.set_index("trade_month")["month_return"])
        rows.append({"book": book, **metrics})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def build_correlation_outputs(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = returns.corr()
    rolling_rows = []
    for a, b in itertools.combinations(returns.columns, 2):
        s = returns[a].rolling(12, min_periods=6).corr(returns[b]).dropna()
        rolling_rows.append(
            {
                "book_a": a,
                "book_b": b,
                "mean_rolling_12m_corr": float(s.mean()) if not s.empty else np.nan,
                "min_rolling_12m_corr": float(s.min()) if not s.empty else np.nan,
                "max_rolling_12m_corr": float(s.max()) if not s.empty else np.nan,
                "latest_rolling_12m_corr": float(s.iloc[-1]) if not s.empty else np.nan,
                "n_obs": int(len(s)),
            }
        )
    return corr, pd.DataFrame(rolling_rows)


def select_best_combos(perf: pd.DataFrame) -> pd.DataFrame:
    out = perf.copy()
    out["score"] = (
        out["sharpe"].fillna(-99) * 10
        + out["annualized_return"].fillna(-99) * 5
        + out["max_drawdown"].fillna(-1) * 3
        + out["positive_month_rate"].fillna(0) * 2
    )
    out = out.sort_values(["n_books", "score", "sharpe"], ascending=[True, False, False])
    best = out.groupby("n_books", as_index=False).head(1).copy()
    best["decision_label"] = best["n_books"].map({2: "best_2_book_core", 3: "best_3_book_core", 4: "best_4_book_core"})
    return best.sort_values("n_books").reset_index(drop=True)


def decide_book_roles(book_summary: pd.DataFrame, best_combos: pd.DataFrame) -> pd.DataFrame:
    best3 = ""
    best4 = ""
    if not best_combos[best_combos["n_books"] == 3].empty:
        best3 = str(best_combos[best_combos["n_books"] == 3].iloc[0]["combo_name"])
    if not best_combos[best_combos["n_books"] == 4].empty:
        best4 = str(best_combos[best_combos["n_books"] == 4].iloc[0]["combo_name"])
    rows = []
    for _, row in book_summary.iterrows():
        book = str(row["book"])
        if book in {"france", "sweden"} and book in best3:
            role = "core_book"
        elif book == "netherlands" and book in best3:
            role = "supporting_book"
        elif book == "germany":
            role = "needs_validation"
        elif book in best4:
            role = "supporting_book"
        else:
            role = "not_selected_for_current_core"
        rows.append(
            {
                "book": book,
                "role": role,
                "book_sharpe": row["sharpe"],
                "book_total_return": row["total_return"],
                "book_max_drawdown": row["max_drawdown"],
                "reason": role_reason(book, role),
            }
        )
    return pd.DataFrame(rows)


def role_reason(book: str, role: str) -> str:
    if book == "germany":
        return "Strong standalone candidate but phase-3 stress_trending risk remains unresolved."
    if role == "core_book":
        return "Included in best simple core and has strong standalone/robust evidence."
    if role == "supporting_book":
        return "Adds diversification/stability but is not the primary driver."
    return "Not selected in current simple core ranking."


def run_multibook_portfolio(options: MultibookOptions) -> Path:
    out_dir = build_output_dir(options)
    books = default_books()
    monthly_frames = [load_book_monthly(book, out_dir, options) for book in books]
    monthly = pd.concat(monthly_frames, ignore_index=True, sort=False)
    returns = align_book_returns(monthly)

    book_defs = book_definitions_frame(books)
    book_summary = build_book_summary(monthly)
    corr, rolling_corr = build_correlation_outputs(returns)
    combo_perf, combo_dd, combo_monthly = build_combination_outputs(returns)
    best_combos = select_best_combos(combo_perf)
    book_roles = decide_book_roles(book_summary, best_combos)

    book_defs.to_csv(out_dir / "book_definitions.csv", index=False)
    book_summary.to_csv(out_dir / "book_return_summary.csv", index=False)
    corr.to_csv(out_dir / "book_correlation_matrix.csv")
    rolling_corr.to_csv(out_dir / "rolling_correlation_summary.csv", index=False)
    monthly.to_csv(out_dir / "book_monthly_returns.csv", index=False)
    returns.to_csv(out_dir / "book_monthly_returns_wide.csv")
    combo_perf.to_csv(out_dir / "combination_performance.csv", index=False)
    combo_dd.to_csv(out_dir / "combination_drawdown_summary.csv", index=False)
    combo_monthly.to_csv(out_dir / "combination_monthly_returns.csv", index=False)
    book_roles.to_csv(out_dir / "book_role_decisions.csv", index=False)
    best_combos.to_csv(out_dir / "portfolio_core_ranking.csv", index=False)
    build_drawdown_details(combo_dd).to_csv(out_dir / "drawdown_contribution_details.csv", index=False)

    metadata = {
        "start": options.start,
        "end": options.end,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "books": [book.__dict__ | {"source_dir": str(book.source_dir)} for book in books],
        "weight_schemes": ["equal_weight", "inverse_vol"],
        "methodology": "Monthly-return portfolio construction over fixed local books. No Markowitz, no retuning, no cross-country rule transfer.",
        "germany_note": "Germany uses fixed corr<=0.75, not the unpromoted stress_trending mitigation.",
        "monthly_return_note": "Sweden monthly series comes from robustness split outputs concatenated across contiguous splits; Germany monthly series is regenerated from the fixed phase-3 helper.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_text_outputs(out_dir, book_summary, corr, combo_perf, combo_dd, best_combos, book_roles)
    return out_dir


def build_drawdown_details(combo_dd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in combo_dd.iterrows():
        for col in combo_dd.columns:
            if col.startswith("dd_contrib_"):
                rows.append(
                    {
                        "combo_name": row["combo_name"],
                        "weight_scheme": row["weight_scheme"],
                        "book": col.replace("dd_contrib_", ""),
                        "weighted_return_during_max_dd": row[col],
                        "max_drawdown": row["max_drawdown"],
                        "dd_start": row["dd_start"],
                        "dd_end": row["dd_end"],
                    }
                )
    return pd.DataFrame(rows)


def write_text_outputs(
    out_dir: Path,
    book_summary: pd.DataFrame,
    corr: pd.DataFrame,
    combo_perf: pd.DataFrame,
    combo_dd: pd.DataFrame,
    best_combos: pd.DataFrame,
    book_roles: pd.DataFrame,
) -> None:
    lines = ["Multibook portfolio campaign", "", "Best cores:"]
    for _, row in best_combos.iterrows():
        lines.append(
            f"- {row['decision_label']}: {row['combo_name']} | {row['weight_scheme']} "
            f"| Sharpe={row['sharpe']:.3f} | return={row['total_return']:.3f} | maxDD={row['max_drawdown']:.3f}"
        )
    lines.append("")
    lines.append("Book roles:")
    for _, row in book_roles.iterrows():
        lines.append(f"- {row['book']}: {row['role']} | {row['reason']}")
    (out_dir / "promotion_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best4 = best_combos[best_combos["n_books"] == 4]
    best3 = best_combos[best_combos["n_books"] == 3]
    best4_sharpe = safe_float(best4.iloc[0]["sharpe"]) if not best4.empty else np.nan
    best3_sharpe = safe_float(best3.iloc[0]["sharpe"]) if not best3.empty else np.nan
    sufficient = best4_sharpe >= max(best3_sharpe - 0.05, 0.0) if np.isfinite(best4_sharpe) and np.isfinite(best3_sharpe) else False
    conclusion = [
        "Conclusion",
        "",
        f"Global decision: {'current_core_is_sufficient' if sufficient else 'need_additional_country_candidate'}",
        "Use the best 3-book core as the current clean candidate if it dominates the 4-book version on risk-adjusted metrics.",
        "Germany remains needs_validation because its local rule is simple and strong but stress_trending risk is unresolved.",
        "",
        "Read portfolio_core_ranking.csv, combination_performance.csv and combination_drawdown_summary.csv for the numeric basis.",
    ]
    (out_dir / "conclusion.txt").write_text("\n".join(conclusion) + "\n", encoding="utf-8")

