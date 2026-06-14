from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.core4_audit_common import resolve_path


LOGGER = logging.getLogger("core4_institutional_memo")

DEFAULT_AUDIT_PACK_DIR = Path("data/experiments/core4_audit_pack/20260604_222941")
DEFAULT_VALIDATION_PACK_DIR = Path("data/experiments/core4_validation_pack/20260514_105610")
DEFAULT_OUTPUT_DIR = Path("data/reports/core4_institutional_memo")
DEFAULT_FREEZE_PATH = Path("config/frozen/core_4_country_v1.freeze.json")
DEFAULT_REPORTING_DIR = Path("data/reports/core4_daily_reporting")
DEFAULT_ALLOCATION_RESEARCH_DIR = Path(
    "data/experiments/portfolio_allocation_research_france_germany_netherlands_sweden_20260423_000056_codex_main_final"
)
DEFAULT_MULTIBOOK_DIR = Path("data/experiments/multibook_portfolio_sweden_germany_france_netherlands_20260421_190655")

REPORT_TITLE = "Core 4 Country Stat Arb - PM Research Memo"
OUTPUT_HTML = "core4_country_v1_institutional_memo.html"
OUTPUT_MARKDOWN = "core4_country_v1_institutional_memo.md"
OUTPUT_MANIFEST = "memo_inputs_manifest.json"

SCANNER_LOOKBACKS = {"3m": 63, "6m": 126, "12m": 252}
SCANNER_THRESHOLDS = {
    "corr_min": 0.30,
    "adf_p_max": 0.05,
    "eg_p_max": 0.05,
    "half_life_max": 100,
}
MONTHLY_UNIVERSE_RULES = {
    "eligibility_allowed": "ELIGIBLE",
    "top_k": 20,
    "min_pairs_required": 3,
}

COUNTRY_RESOURCE_PATHS: dict[str, dict[str, Path]] = {
    "france": {
        "metadata": Path("data/experiments/country_research_france_20180101_20251231_20260419_210305/metadata.json"),
        "reference": Path(
            "data/experiments/country_research_france_20180101_20251231_20260419_210305/reference_selection.json"
        ),
        "scan": Path("data/experiments/robust_cross_sectional_long_2015_2025/scans/france.parquet"),
    },
    "germany": {
        "metadata": Path("data/experiments/germany_core_entry_20180101_20251231_20260421_215937/metadata.json"),
        "reference": Path("data/experiments/country_research_germany_20180101_20251231_20260419_210416/reference_selection.json"),
        "scan": Path("data/experiments/robust_cross_sectional_long_2015_2025/scans/germany.parquet"),
    },
    "netherlands": {
        "metadata": Path("data/experiments/country_research_netherlands_20180101_20251231_20260420_192536/metadata.json"),
        "reference": Path(
            "data/experiments/country_research_netherlands_20180101_20251231_20260420_192536/reference_selection.json"
        ),
        "scan": Path("data/experiments/robust_cross_sectional_long_2015_2025/scans/netherlands.parquet"),
    },
    "sweden": {
        "metadata": Path("data/experiments/country_research_sweden_20180101_20251231_20260419_210503/metadata.json"),
        "reference": Path("data/experiments/country_research_sweden_20180101_20251231_20260419_210503/reference_selection.json"),
        "scan": Path("data/experiments/robust_cross_sectional_long_2015_2025/scans/sweden.parquet"),
    },
}

MAIN_FIGURE_IDS = [
    "universe_coverage",
    "pair_trading_schematic",
    "zscore_schematic",
    "statistical_gates_schematic",
    "country_correlation_matrix",
    "equity_curve",
    "drawdown_curve",
    "gross_net_exposure",
    "borrow_stress_ann_return",
]

FIGURE_TREATMENTS: list[dict[str, str]] = [
    {
        "id": "universe_coverage",
        "title": "Universe Coverage By Country",
        "treatment": "kept_in_main_memo",
        "reason": "Shows the breadth of the local stock universes and keeps the memo grounded in actual names and scope.",
    },
    {
        "id": "pair_trading_schematic",
        "title": "Pair Trading Concept Schematic",
        "treatment": "kept_in_main_memo",
        "reason": "Explains the core idea visually before the memo moves into performance and readiness.",
    },
    {
        "id": "zscore_schematic",
        "title": "Z-Score Entry And Exit Schematic",
        "treatment": "kept_in_main_memo",
        "reason": "Explains how the spread is standardized and where entry, exit, and stop zones sit.",
    },
    {
        "id": "statistical_gates_schematic",
        "title": "Statistical Gate Overview",
        "treatment": "kept_in_main_memo",
        "reason": "Summarizes the statistical filters in one PM-friendly visual.",
    },
    {
        "id": "country_correlation_matrix",
        "title": "Country Correlation Matrix",
        "treatment": "kept_in_main_memo",
        "reason": "Supports the diversification case without relying on allocator heatmaps.",
    },
    {
        "id": "equity_curve",
        "title": "Frozen Portfolio Equity Curve",
        "treatment": "kept_in_main_memo",
        "reason": "Primary PM view of compounded historical performance.",
    },
    {
        "id": "drawdown_curve",
        "title": "Frozen Portfolio Drawdown Curve",
        "treatment": "kept_in_main_memo",
        "reason": "Primary PM view of historical downside and recovery behavior.",
    },
    {
        "id": "gross_net_exposure",
        "title": "Gross And Net Exposure Over Time",
        "treatment": "kept_in_main_memo",
        "reason": "Useful implementation diagnostic without dropping into trade-level clutter.",
    },
    {
        "id": "borrow_stress_ann_return",
        "title": "Borrow Stress Annualized Return",
        "treatment": "kept_in_main_memo",
        "reason": "Compact visual for the main borrow-cost conclusion.",
    },
    {
        "id": "execution_delay_sharpe",
        "title": "Execution Delay Stress Sharpe",
        "treatment": "moved_to_appendix",
        "reason": "Still useful, but the diagnostic is not order-level and should not dominate the core decision path.",
    },
    {
        "id": "borrow_stress_sharpe",
        "title": "Borrow Stress Sharpe",
        "treatment": "moved_to_appendix",
        "reason": "Secondary to the borrow-cost table once annualized return is shown.",
    },
    {
        "id": "monthly_heatmap",
        "title": "Monthly Return Heatmap",
        "treatment": "excluded",
        "reason": "Adds density without improving comprehension.",
    },
    {
        "id": "rolling_sharpe",
        "title": "Rolling Sharpe Proxy",
        "treatment": "excluded",
        "reason": "Proxy metric that can look more precise than the underlying evidence.",
    },
    {
        "id": "country_standalone_profile",
        "title": "Country Standalone Profile",
        "treatment": "excluded",
        "reason": "The memo keeps the country discussion in readable tables instead of extra diagnostic charts.",
    },
    {
        "id": "country_annual_contribution_heatmap",
        "title": "Country Annual Contribution Heatmap",
        "treatment": "excluded",
        "reason": "Too detailed for the main PM narrative.",
    },
    {
        "id": "leave_one_country_out_delta_sharpe",
        "title": "Leave-One-Country-Out Delta Sharpe",
        "treatment": "excluded",
        "reason": "Valid diagnostic, but not necessary in the main memo once country roles are explained.",
    },
    {
        "id": "allocator_sharpe_heatmap",
        "title": "Allocator Sharpe Heatmap",
        "treatment": "excluded",
        "reason": "Looks like allocator tuning rather than an investment memo.",
    },
    {
        "id": "allocator_drawdown_heatmap",
        "title": "Allocator Drawdown Heatmap",
        "treatment": "excluded",
        "reason": "Too optimization-heavy for a PM-facing document.",
    },
    {
        "id": "sweden_entry_is_heatmap",
        "title": "Sweden Entry Grid IS Sharpe",
        "treatment": "excluded",
        "reason": "Detailed campaign evidence intentionally removed from the main memo.",
    },
    {
        "id": "sweden_entry_oos_heatmap",
        "title": "Sweden Entry Grid OOS Sharpe",
        "treatment": "excluded",
        "reason": "Detailed campaign evidence intentionally removed from the main memo.",
    },
    {
        "id": "sweden_speedfilter_is_oos_heatmap",
        "title": "Sweden Speed Filter IS/OOS Sharpe",
        "treatment": "excluded",
        "reason": "Too campaign-specific for the intended audience.",
    },
    {
        "id": "sweden_robustness_split_heatmap",
        "title": "Sweden Overlay Split Robustness",
        "treatment": "excluded",
        "reason": "Research appendix material, not a front-of-memo concept chart.",
    },
    {
        "id": "germany_oos_sharpe_heatmap",
        "title": "Germany OOS Config Sharpe",
        "treatment": "excluded",
        "reason": "Detailed configuration chart removed to avoid a research-dump feel.",
    },
    {
        "id": "active_pairs_over_time",
        "title": "Active Pairs Over Time",
        "treatment": "excluded",
        "reason": "Operationally interesting, but not core to the memo's teaching arc.",
    },
    {
        "id": "largest_book_weight",
        "title": "Largest Book Weight Over Time",
        "treatment": "excluded",
        "reason": "Redundant once allocation and exposure are explained directly.",
    },
    {
        "id": "book_level_exposure",
        "title": "Book-Level Gross Exposure Over Time",
        "treatment": "excluded",
        "reason": "Too granular for the main PM narrative.",
    },
    {
        "id": "trades_by_book",
        "title": "Trades By Book",
        "treatment": "excluded",
        "reason": "Less instructive than representative-stock and representative-pair tables.",
    },
    {
        "id": "holding_period_histogram",
        "title": "Holding Period Distribution",
        "treatment": "excluded",
        "reason": "Trade-level detail that does not teach the core logic better than the text and tables.",
    },
    {
        "id": "trade_pnl_distribution",
        "title": "Trade PnL Distribution",
        "treatment": "excluded",
        "reason": "Trade-level PnL scale is not the right anchor for this memo.",
    },
    {
        "id": "cumulative_pnl_by_book",
        "title": "Cumulative PnL By Book",
        "treatment": "excluded",
        "reason": "Invites over-reading of trade-level proxies rather than portfolio evidence.",
    },
    {
        "id": "execution_delay_max_drawdown",
        "title": "Execution Delay Stress Max Drawdown",
        "treatment": "excluded",
        "reason": "Secondary diagnostic once the delay table is shown.",
    },
]


@dataclass(frozen=True)
class Core4InstitutionalMemoOptions:
    audit_pack_dir: Path = DEFAULT_AUDIT_PACK_DIR
    validation_pack_dir: Path = DEFAULT_VALIDATION_PACK_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    freeze_path: Path = DEFAULT_FREEZE_PATH
    reporting_dir: Path = DEFAULT_REPORTING_DIR
    allocation_research_dir: Path = DEFAULT_ALLOCATION_RESEARCH_DIR
    multibook_dir: Path = DEFAULT_MULTIBOOK_DIR
    smoke: bool = False


def build_core4_institutional_memo(
    options: Core4InstitutionalMemoOptions,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = _resolve_inputs(options, project_root=project_root)
    output_dir = resolved["output_dir"]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_figures(figures_dir)

    bundle = _load_bundle(resolved, project_root=project_root)
    metrics = _compute_metrics(bundle)
    figures = _generate_figures(bundle, metrics, figures_dir)
    figure_manifest = _build_figure_manifest(figures)
    markdown = _render_markdown(bundle, metrics, figures, figure_manifest)
    html = _render_html(bundle, metrics, figures, figure_manifest)
    manifest = _build_manifest(bundle, metrics, figures, figure_manifest, output_dir)

    html_path = output_dir / OUTPUT_HTML
    markdown_path = output_dir / OUTPUT_MARKDOWN
    manifest_path = output_dir / OUTPUT_MANIFEST
    html_path.write_text(html, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "html_path": html_path,
        "markdown_path": markdown_path,
        "manifest_path": manifest_path,
        "figures": figures,
        "figure_manifest": figure_manifest,
        "bundle": bundle,
        "metrics": metrics,
    }


def _resolve_inputs(options: Core4InstitutionalMemoOptions, *, project_root: Path) -> dict[str, Path]:
    resolved = {
        "audit_pack_dir": resolve_path(project_root, options.audit_pack_dir),
        "validation_pack_dir": resolve_path(project_root, options.validation_pack_dir),
        "output_dir": resolve_path(project_root, options.output_dir),
        "freeze_path": resolve_path(project_root, options.freeze_path),
        "reporting_dir": resolve_path(project_root, options.reporting_dir),
        "allocation_research_dir": resolve_path(project_root, options.allocation_research_dir),
        "multibook_dir": resolve_path(project_root, options.multibook_dir),
        "portfolio_reference_path": resolve_path(project_root, "config/core_portfolio_reference.json"),
        "project_root": project_root,
    }
    resolved["output_dir"].mkdir(parents=True, exist_ok=True)
    return resolved


def _clear_generated_figures(figures_dir: Path) -> None:
    for png_path in figures_dir.glob("*.png"):
        png_path.unlink()


def _load_bundle(paths: dict[str, Path], *, project_root: Path) -> dict[str, Any]:
    audit_dir = paths["audit_pack_dir"]
    validation_dir = paths["validation_pack_dir"]
    reporting_dir = paths["reporting_dir"]
    allocation_dir = paths["allocation_research_dir"]
    multibook_dir = paths["multibook_dir"]

    trade_ledger_sources = pd.read_csv(audit_dir / "trade_ledger_sources.csv")
    bundle = {
        "paths": paths,
        "project_root": project_root,
        "freeze": json.loads(paths["freeze_path"].read_text(encoding="utf-8")),
        "portfolio_reference": json.loads(paths["portfolio_reference_path"].read_text(encoding="utf-8")),
        "audit_summary_text": (audit_dir / "audit_pack_summary.md").read_text(encoding="utf-8"),
        "validation_summary_text": (validation_dir / "validation_summary.md").read_text(encoding="utf-8"),
        "daily_reporting_text": _read_text_optional(reporting_dir / "conclusion.txt"),
        "allocation_research_text": _read_text_optional(allocation_dir / "conclusion.txt"),
        "multibook_text": _read_text_optional(multibook_dir / "promotion_decision.txt"),
        "daily_positions": pd.read_csv(audit_dir / "daily_positions.csv"),
        "daily_book_exposures": pd.read_csv(audit_dir / "daily_book_exposures.csv"),
        "daily_portfolio_exposures": pd.read_csv(audit_dir / "daily_portfolio_exposures.csv"),
        "trade_ledger": pd.read_csv(audit_dir / "trade_ledger.csv"),
        "borrow_cost_stress": pd.read_csv(audit_dir / "borrow_cost_stress.csv"),
        "execution_delay_stress": pd.read_csv(audit_dir / "execution_delay_stress.csv"),
        "trade_ledger_sources": trade_ledger_sources,
        "portfolio_daily_equity": pd.read_csv(reporting_dir / "portfolio_daily_equity.csv"),
        "portfolio_daily_drawdown": pd.read_csv(reporting_dir / "portfolio_daily_drawdown.csv"),
        "book_weights_history": pd.read_csv(reporting_dir / "book_weights_history.csv"),
        "allocator_comparison_summary": pd.read_csv(reporting_dir / "allocator_comparison_summary.csv"),
        "portfolio_monthly_returns": _read_csv_optional(reporting_dir / "portfolio_monthly_returns.csv"),
        "country_correlation_matrix": _read_csv_optional(validation_dir / "correlation_matrix.csv"),
        "country_standalone": _read_csv_optional(validation_dir / "country_standalone.csv"),
        "annual_country_contribution": _read_csv_optional(validation_dir / "annual_country_contribution.csv"),
        "leave_one_country_out": _read_csv_optional(validation_dir / "leave_one_country_out.csv"),
        "country_resources": _load_country_resources(project_root, trade_ledger_sources),
    }
    return bundle


def _load_country_resources(project_root: Path, trade_ledger_sources: pd.DataFrame) -> dict[str, dict[str, Any]]:
    resources: dict[str, dict[str, Any]] = {}
    for country, rel_paths in COUNTRY_RESOURCE_PATHS.items():
        item: dict[str, Any] = {}
        for name, rel_path in rel_paths.items():
            resolved_path = resolve_path(project_root, rel_path)
            if name == "scan":
                item[name] = _read_parquet_optional(resolved_path)
                item[f"{name}_path"] = resolved_path
            else:
                item[name] = _read_json_optional(resolved_path)
                item[f"{name}_path"] = resolved_path

        source_row = trade_ledger_sources.loc[trade_ledger_sources["country"].astype(str).str.lower().eq(country)].head(1)
        if not source_row.empty:
            source_path = Path(str(source_row.iloc[0]["path"]))
            item["source_trade_path"] = source_path
            item["source_trade_rows"] = _read_csv_optional(source_path)
            item["source_status"] = str(source_row.iloc[0]["status"])
        else:
            item["source_trade_path"] = None
            item["source_trade_rows"] = pd.DataFrame()
            item["source_status"] = "unknown"

        resources[country] = item
    return resources


def _read_text_optional(path: Path) -> str:
    if not path.exists():
        LOGGER.warning("Optional text input not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8")


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Optional CSV input not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Optional parquet input not found: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.warning("Unable to read parquet %s: %s", path, exc)
        return pd.DataFrame()


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        LOGGER.warning("Optional JSON input not found: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    freeze = bundle["freeze"]
    validation = _parse_validation_metrics(bundle["validation_summary_text"])
    audit = _parse_audit_summary(bundle["audit_summary_text"])

    trade_ledger = bundle["trade_ledger"].copy()
    daily_book_exposures = bundle["daily_book_exposures"].copy()
    daily_portfolio_exposures = bundle["daily_portfolio_exposures"].copy()
    borrow_cost_stress = bundle["borrow_cost_stress"].copy()
    execution_delay_stress = bundle["execution_delay_stress"].copy()
    allocator_summary = bundle["allocator_comparison_summary"].copy()
    country_standalone = bundle["country_standalone"].copy()
    annual_country_contribution = bundle["annual_country_contribution"].copy()
    leave_one_country_out = bundle["leave_one_country_out"].copy()

    for col in ["gross_exposure", "net_exposure", "num_active_pairs", "largest_book_weight"]:
        if col in daily_portfolio_exposures.columns:
            daily_portfolio_exposures[col] = pd.to_numeric(daily_portfolio_exposures[col], errors="coerce")

    latest_exposure = {}
    if not daily_portfolio_exposures.empty:
        latest = daily_portfolio_exposures.sort_values("date").iloc[-1]
        latest_exposure = {
            "date": str(latest.get("date", "")),
            "gross_exposure": float(latest.get("gross_exposure", np.nan)),
            "net_exposure": float(latest.get("net_exposure", np.nan)),
            "num_active_pairs": float(latest.get("num_active_pairs", np.nan)),
            "largest_book_weight": float(latest.get("largest_book_weight", np.nan)),
        }

    borrow_500_row = _match_row(borrow_cost_stress, "scenario", "borrow_500bps")
    coverage_counts = daily_book_exposures["coverage_status"].astype(str).value_counts(dropna=False).to_dict()
    reference_allocator = freeze["reference_allocation"]["allocator_id"]
    benchmark_allocator = freeze["benchmark_allocation"]["allocator_id"]
    reference_summary = _lookup_allocator_summary(allocator_summary, reference_allocator)
    benchmark_summary = _lookup_allocator_summary(allocator_summary, benchmark_allocator)
    yearly_returns = _build_yearly_returns(bundle["portfolio_monthly_returns"], reference_allocator)
    trade_counts_by_country = {
        str(country).lower(): int(count)
        for country, count in trade_ledger.groupby(trade_ledger["book"].astype(str).str.lower()).size().items()
    }

    weights_by_country = {}
    if not country_standalone.empty:
        for _, row in country_standalone.iterrows():
            weights_by_country[str(row["country"]).lower()] = float(row.get("frozen_weight", np.nan))

    universe_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    example_pair_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []

    for book in sorted(freeze["countries_books"], key=lambda item: _country_sort_key(str(item["country"]))):
        country = str(book["country"]).lower()
        resources = bundle["country_resources"][country]
        universe_rows.append(_build_universe_row(country, resources, trade_ledger, book))
        config_rows.append(_build_config_row(country, resources, book))
        example_pair_rows.append(_build_example_pair_row(country, resources, trade_ledger))
        country_rows.append(
            {
                "country": country.title(),
                "role": _country_role_label(country),
                "selected_configuration": str(book["config_name"]),
                "weight": weights_by_country.get(country, np.nan),
                "pm_comment": _country_pm_comment(country),
            }
        )

    country_evidence = _build_country_evidence(
        country_rows=country_rows,
        universe_rows=universe_rows,
        config_rows=config_rows,
        country_standalone=country_standalone,
        annual_country_contribution=annual_country_contribution,
        leave_one_country_out=leave_one_country_out,
        trade_counts_by_country=trade_counts_by_country,
        country_resources=bundle["country_resources"],
    )

    return {
        "strategy_id": str(freeze["strategy_id"]),
        "status": "paper_ready_with_limitations",
        "status_display": "paper-ready with limitations",
        "audit_verdict": audit.get("verdict", "paper_ready_with_limitations"),
        "countries": [str(book["country"]).title() for book in sorted(freeze["countries_books"], key=lambda item: _country_sort_key(str(item["country"])))],
        "validation_period": freeze["validation_period"],
        "validation_metrics": validation,
        "reference_allocator": reference_allocator,
        "benchmark_allocator": benchmark_allocator,
        "reference_allocator_summary": reference_summary,
        "benchmark_allocator_summary": benchmark_summary,
        "reporting_comparison_text": _capitalize_first(_extract_after_colon(bundle["daily_reporting_text"], "Decision read-through")),
        "reconstructed_positions": int(len(bundle["daily_positions"])),
        "reconstructed_trades": int(len(trade_ledger)),
        "paired_book_days": int(coverage_counts.get("paired_with_reconstructed_positions", 0)),
        "proxy_only_book_days": int(coverage_counts.get("exposure_without_reconstructed_pairs", 0)),
        "latest_portfolio_exposure": latest_exposure,
        "borrow_stress_rows": borrow_cost_stress.to_dict(orient="records"),
        "execution_delay_rows": execution_delay_stress.to_dict(orient="records"),
        "borrow_500bps": borrow_500_row,
        "known_limitations": list(freeze.get("known_limitations", [])),
        "yearly_returns": yearly_returns,
        "country_rows": country_rows,
        "universe_rows": universe_rows,
        "config_rows": config_rows,
        "example_pair_rows": example_pair_rows,
        "country_evidence": country_evidence,
        "scanner_thresholds": SCANNER_THRESHOLDS,
        "scanner_lookbacks": SCANNER_LOOKBACKS,
        "monthly_universe_rules": MONTHLY_UNIVERSE_RULES,
    }


def _match_row(df: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {}
    match = df.loc[df[column].astype(str).eq(value)].head(1)
    return match.iloc[0].to_dict() if not match.empty else {}


def _lookup_allocator_summary(df: pd.DataFrame, allocator_id: str) -> dict[str, Any]:
    if df.empty or "config_id" not in df.columns:
        return {}
    row = df.loc[df["config_id"].astype(str).eq(allocator_id)].head(1)
    return row.iloc[0].to_dict() if not row.empty else {}


def _build_yearly_returns(df: pd.DataFrame, allocator_id: str) -> list[dict[str, Any]]:
    if df.empty or "config_id" not in df.columns:
        return []
    scoped = df.loc[df["config_id"].astype(str).eq(allocator_id)].copy()
    if scoped.empty:
        return []
    scoped["trade_month"] = pd.to_datetime(scoped["trade_month"], errors="coerce")
    scoped["month_return"] = pd.to_numeric(scoped["month_return"], errors="coerce").fillna(0.0)
    scoped["year"] = scoped["trade_month"].dt.year
    grouped = scoped.groupby("year", dropna=False)["month_return"].apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    return [{"year": int(year), "return": value} for year, value in grouped.items() if pd.notna(year)]


def _build_country_evidence(
    *,
    country_rows: list[dict[str, Any]],
    universe_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    country_standalone: pd.DataFrame,
    annual_country_contribution: pd.DataFrame,
    leave_one_country_out: pd.DataFrame,
    trade_counts_by_country: dict[str, int],
    country_resources: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    universe_by_country = {str(row["country"]).lower(): row for row in universe_rows}
    config_by_country = {str(row["country"]).lower(): row for row in config_rows}
    standalone_by_country = {
        str(row["country"]).lower(): row.to_dict() for _, row in country_standalone.iterrows()
    } if not country_standalone.empty else {}
    leave_one_by_country = {
        str(row["excluded_country"]).lower(): row.to_dict() for _, row in leave_one_country_out.iterrows()
    } if not leave_one_country_out.empty else {}

    contribution_by_country: dict[str, dict[str, Any]] = {}
    if not annual_country_contribution.empty:
        grouped = annual_country_contribution.groupby(annual_country_contribution["country"].astype(str).str.lower(), dropna=False)
        for country, group in grouped:
            contribution_by_country[str(country)] = {
                "sum_weighted_contribution": float(pd.to_numeric(group["annual_weighted_contribution"], errors="coerce").sum()),
                "mean_rank": float(pd.to_numeric(group["contribution_rank"], errors="coerce").mean()),
                "mean_abs_share": float(pd.to_numeric(group["contribution_pct_of_total_absolute"], errors="coerce").mean()),
            }

    evidence: dict[str, dict[str, Any]] = {}
    for row in country_rows:
        country = str(row["country"]).lower()
        evidence[country] = {
            "country": row["country"],
            "role": row["role"],
            "selected_configuration": row["selected_configuration"],
            "weight": row["weight"],
            "pm_comment": row["pm_comment"],
            "trade_count": int(trade_counts_by_country.get(country, 0)),
            "source_status": country_resources.get(country, {}).get("source_status", "unknown"),
            "standalone": standalone_by_country.get(country, {}),
            "leave_one_out": leave_one_by_country.get(country, {}),
            "contribution": contribution_by_country.get(country, {}),
            **universe_by_country.get(country, {}),
            **config_by_country.get(country, {}),
        }
    return evidence


def _build_universe_row(
    country: str,
    resources: dict[str, Any],
    trade_ledger: pd.DataFrame,
    freeze_book: dict[str, Any],
) -> dict[str, Any]:
    scan_df = resources["scan"].copy()
    asset_count = 0
    scan_rows = 0
    eligible_rows = 0
    if not scan_df.empty:
        assets = sorted(set(scan_df["asset_1"].astype(str)).union(set(scan_df["asset_2"].astype(str))))
        asset_count = len(assets)
        scan_rows = int(len(scan_df))
        if "eligibility" in scan_df.columns:
            eligible_rows = int((scan_df["eligibility"].astype(str) == MONTHLY_UNIVERSE_RULES["eligibility_allowed"]).sum())

    book_trades = trade_ledger.loc[trade_ledger["book"].astype(str).str.lower().eq(country)].copy()
    observed_pairs = int(book_trades["pair_id"].nunique()) if not book_trades.empty else 0
    representative_stocks = ", ".join(_top_assets(book_trades, limit=6))

    return {
        "country": country.title(),
        "role": _country_role_label(country),
        "asset_count": asset_count,
        "scan_rows": scan_rows,
        "eligible_rows": eligible_rows,
        "observed_pairs": observed_pairs,
        "representative_stocks": representative_stocks or "Unavailable",
        "selected_configuration": str(freeze_book["config_name"]),
    }


def _build_config_row(country: str, resources: dict[str, Any], freeze_book: dict[str, Any]) -> dict[str, Any]:
    metadata = resources["metadata"]
    reference = metadata.get("reference", {}) if isinstance(metadata.get("reference"), dict) else resources.get("reference", {})
    selected_rule = _selected_rule(country, freeze_book["config_name"], metadata)
    return {
        "country": country.title(),
        "selected_rule": selected_rule,
        "scan_rhythm": _scan_rhythm_label(reference),
        "z_window": reference.get("z_window", np.nan),
        "entry_mode": _entry_mode_label(reference.get("entry_mode")),
        "entry_z": 1.8,
        "exit_z": 0.6,
        "stop_z": 3.6,
        "max_holding_days": reference.get("max_holding_days", np.nan),
        "max_positions": reference.get("max_positions", np.nan),
        "top_n_candidates": reference.get("top_n_candidates", np.nan),
    }


def _build_example_pair_row(country: str, resources: dict[str, Any], trade_ledger: pd.DataFrame) -> dict[str, Any]:
    book_trades = trade_ledger.loc[trade_ledger["book"].astype(str).str.lower().eq(country)].copy()
    if book_trades.empty:
        return {
            "country": country.title(),
            "pair": "Unavailable",
            "ledger_trades": 0,
            "median_entry_z": np.nan,
            "median_corr_6m": np.nan,
            "median_half_life_6m": np.nan,
            "median_adf_pvalue_6m": np.nan,
            "median_eg_pvalue_6m": np.nan,
            "typical_exit": "Unavailable",
            "source_status": resources.get("source_status", "unknown"),
        }

    pair_name = str(book_trades["pair_id"].value_counts().idxmax())
    ledger_pair = book_trades.loc[book_trades["pair_id"].astype(str).eq(pair_name)].copy()
    source_df = _filter_source_rows_for_book(resources["source_trade_rows"], ledger_pair)
    pair_source = source_df.loc[source_df["pair_id"].astype(str).eq(pair_name)].copy() if not source_df.empty else pd.DataFrame()
    if pair_source.empty:
        pair_source = ledger_pair.copy()

    pair_label = _pair_label_from_rows(pair_source, ledger_pair, pair_name)

    return {
        "country": country.title(),
        "pair": pair_label,
        "ledger_trades": int(len(ledger_pair)),
        "median_entry_z": _median_abs(pair_source, "entry_z"),
        "median_corr_6m": _median(pair_source, "corr_6m"),
        "median_half_life_6m": _median(pair_source, "half_life_6m"),
        "median_adf_pvalue_6m": _median(pair_source, "adf_pvalue_6m"),
        "median_eg_pvalue_6m": _median(pair_source, "eg_pvalue_6m"),
        "typical_exit": _mode_value(pair_source, "exit_reason") or "Unavailable",
        "source_status": resources.get("source_status", "unknown"),
    }


def _filter_source_rows_for_book(source_df: pd.DataFrame, ledger_pair: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return source_df
    filtered = source_df.copy()
    if "config_name" in filtered.columns and "source_config_name" in ledger_pair.columns:
        mode_cfg = ledger_pair["source_config_name"].dropna().astype(str)
        if not mode_cfg.empty:
            target = mode_cfg.mode().iloc[0]
            scoped = filtered.loc[filtered["config_name"].astype(str).eq(target)].copy()
            if not scoped.empty:
                filtered = scoped
    return filtered


def _top_assets(book_trades: pd.DataFrame, *, limit: int) -> list[str]:
    if book_trades.empty:
        return []
    counts = pd.concat([book_trades["leg_1"], book_trades["leg_2"]]).astype(str).value_counts()
    return [_pretty_asset(asset) for asset in counts.head(limit).index.tolist()]


def _median(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.median()) if not values.dropna().empty else np.nan


def _median_abs(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").abs()
    return float(values.median()) if not values.dropna().empty else np.nan


def _mode_value(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    series = df[column].dropna().astype(str)
    if series.empty:
        return ""
    return str(series.mode().iloc[0])


def _parse_validation_metrics(text: str) -> dict[str, float]:
    return {
        "annualized_return": _extract_pct(text, "Annualized return"),
        "annualized_volatility": _extract_pct(text, "Annualized volatility"),
        "sharpe": _extract_number(text, "Sharpe"),
        "max_drawdown": _extract_pct(text, "Max drawdown"),
        "cumulative_return": _extract_pct(text, "Cumulative return"),
    }


def _parse_audit_summary(text: str) -> dict[str, str]:
    return {
        "status": _extract_after_colon(text, "Status"),
        "verdict": _extract_after_colon(text, "Verdict"),
        "freeze_date": _extract_after_colon(text, "Freeze date"),
    }


def _generate_figures(bundle: dict[str, Any], metrics: dict[str, Any], figures_dir: Path) -> list[dict[str, Any]]:
    generators = {
        "universe_coverage": ("Universe Coverage By Country", _plot_universe_coverage),
        "pair_trading_schematic": ("Pair Trading Concept Schematic", _plot_pair_trading_schematic),
        "zscore_schematic": ("Z-Score Entry And Exit Schematic", _plot_zscore_schematic),
        "statistical_gates_schematic": ("Statistical Gate Overview", _plot_statistical_gates_schematic),
        "country_correlation_matrix": ("Country Correlation Matrix", _plot_country_correlation_matrix),
        "equity_curve": ("Frozen Portfolio Equity Curve", _plot_equity_curve),
        "drawdown_curve": ("Frozen Portfolio Drawdown Curve", _plot_drawdown_curve),
        "gross_net_exposure": ("Gross And Net Exposure Over Time", _plot_gross_net_exposure),
        "borrow_stress_ann_return": ("Borrow Stress Annualized Return", _plot_borrow_stress_ann_return),
    }

    figures: list[dict[str, Any]] = []
    for figure_id in MAIN_FIGURE_IDS:
        title, plotter = generators[figure_id]
        path = figures_dir / f"{figure_id}.png"
        plotter(bundle, metrics, path)
        figures.append(
            {
                "id": figure_id,
                "title": title,
                "path": path,
                "relative_path": str(Path("figures") / path.name).replace("\\", "/"),
            }
        )
    return figures


def _plot_universe_coverage(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    rows = metrics["universe_rows"]
    labels = [row["country"] for row in rows]
    asset_counts = [row["asset_count"] for row in rows]
    observed_pairs = [row["observed_pairs"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, asset_counts, width, label="Unique stocks in scan universe", color="#386641")
    ax.bar(x + width / 2, observed_pairs, width, label="Unique traded pairs in audit ledger", color="#6a994e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Universe breadth versus realized pair usage")
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pair_trading_schematic(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    x = np.linspace(0, 20, 240)
    stock_a = 100 + 0.8 * x + 2.2 * np.sin(x / 2.0)
    stock_b = 102 + 0.75 * x + 2.0 * np.sin(x / 2.0 + 0.2)
    stock_b[130:170] += np.linspace(0, 8, 40)
    stock_b[170:210] += np.linspace(8, 1, 40)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(x, stock_a, label="Stock A", color="#1d3557", linewidth=2.0)
    ax.plot(x, stock_b, label="Stock B", color="#c1121f", linewidth=2.0)
    ax.axvspan(x[130], x[170], color="#f4d35e", alpha=0.25)
    ax.text(x[140], stock_b[145] + 3.5, "Spread widens", fontsize=10)
    ax.annotate("Enter market-neutral pair", xy=(x[145], stock_b[145]), xytext=(x[95], stock_b[145] + 8),
                arrowprops={"arrowstyle": "->", "color": "#333333"}, fontsize=10)
    ax.annotate("Exit when spread normalizes", xy=(x[195], stock_b[195]), xytext=(x[150], stock_b[195] - 10),
                arrowprops={"arrowstyle": "->", "color": "#333333"}, fontsize=10)
    ax.set_title("Pair trading: two related stocks temporarily pull apart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Indexed price")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_zscore_schematic(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    x = np.linspace(0, 30, 300)
    z = 0.35 * np.sin(x / 1.8)
    z[90:130] += np.linspace(0.0, 2.4, 40)
    z[130:170] += np.linspace(2.4, 0.1, 40)
    z[210:245] -= np.linspace(0.0, 2.2, 35)
    z[245:280] -= np.linspace(2.2, 0.1, 35)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(x, z, color="#0f4c5c", linewidth=2.0)
    for level, color, label in [
        (1.8, "#b56576", "Entry band +1.8"),
        (-1.8, "#b56576", "Entry band -1.8"),
        (0.6, "#588157", "Exit band +0.6"),
        (-0.6, "#588157", "Exit band -0.6"),
        (3.6, "#c1121f", "Stop +3.6"),
        (-3.6, "#c1121f", "Stop -3.6"),
        (0.0, "#666666", "Mean"),
    ]:
        ax.axhline(level, color=color, linewidth=1.0, linestyle="--")
        if label != "Mean":
            ax.text(x[-1] + 0.2, level, label, va="center", fontsize=9, color=color)
    ax.set_xlim(x[0], x[-1] + 4)
    ax.set_ylim(-4.2, 4.2)
    ax.set_title("Spread standardized into a z-score")
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-score")
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_statistical_gates_schematic(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    boxes = [
        ("1. Correlation", "Need co-movement\n6m corr >= 0.30"),
        ("2. Engle-Granger", "Long-run relation\np <= 0.05"),
        ("3. ADF on spread", "Stationary spread\np <= 0.05"),
        ("4. Half-life", "Mean reversion practical\n<= 100 days"),
        ("5. Ranking", "Keep ELIGIBLE names\nTop 20 candidates"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    start_x = 0.3
    width = 1.75
    gap = 0.18
    for idx, (title, body) in enumerate(boxes):
        x = start_x + idx * (width + gap)
        patch = FancyBboxPatch(
            (x, 0.9),
            width,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor="#f5efe6",
            edgecolor="#b08968",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, 1.62, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + width / 2, 1.18, body, ha="center", va="center", fontsize=9)
        if idx < len(boxes) - 1:
            ax.annotate("", xy=(x + width + gap - 0.02, 1.4), xytext=(x + width, 1.4), arrowprops={"arrowstyle": "->"})

    ax.text(0.35, 2.2, "The idea is simple: do not trade every two-stock story. Trade only the ones that pass multiple filters.", fontsize=10)
    ax.set_title("Statistical gates before a pair becomes tradable")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_country_correlation_matrix(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    df = bundle["country_correlation_matrix"].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Correlation matrix unavailable", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    labels = [str(item).title() for item in df["book"].tolist()]
    matrix = df.drop(columns=["book"]).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Average daily return correlation across country books")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_equity_curve(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    df = _reference_rows(bundle["portfolio_daily_equity"], metrics["reference_allocator"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(df["date"], df["equity"], color="#1d3557", linewidth=2.0)
    ax.set_title("Frozen portfolio equity curve")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.20, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_drawdown_curve(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    df = _reference_rows(bundle["portfolio_daily_drawdown"], metrics["reference_allocator"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["drawdown"] = pd.to_numeric(df["drawdown"], errors="coerce")

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.fill_between(df["date"], df["drawdown"], 0.0, color="#c1121f", alpha=0.80)
    ax.set_title("Frozen portfolio drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(alpha=0.20, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_gross_net_exposure(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    df = bundle["daily_portfolio_exposures"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["gross_exposure"] = pd.to_numeric(df["gross_exposure"], errors="coerce")
    df["net_exposure"] = pd.to_numeric(df["net_exposure"], errors="coerce")

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(df["date"], df["gross_exposure"], label="Gross exposure", color="#0f4c5c", linewidth=1.8)
    ax.plot(df["date"], df["net_exposure"], label="Net exposure", color="#b56576", linewidth=1.2)
    ax.set_title("Reconstructed portfolio exposure")
    ax.set_ylabel("Exposure")
    ax.grid(alpha=0.20, linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_borrow_stress_ann_return(bundle: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    df = bundle["borrow_cost_stress"].copy()
    df["borrow_bps_annualized"] = pd.to_numeric(df["borrow_bps_annualized"], errors="coerce")
    df["annualized_return_after_borrow"] = pd.to_numeric(df["annualized_return_after_borrow"], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(df["borrow_bps_annualized"], df["annualized_return_after_borrow"], color="#6a994e", width=36)
    ax.set_title("Borrow stress annualized return")
    ax.set_xlabel("Borrow cost (bps annualized)")
    ax.set_ylabel("Annualized return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _reference_rows(df: pd.DataFrame, allocator_id: str) -> pd.DataFrame:
    if df.empty or "config_id" not in df.columns:
        return df.copy()
    scoped = df.loc[df["config_id"].astype(str).eq(allocator_id)].copy()
    return scoped if not scoped.empty else df.copy()


def _build_figure_manifest(figures: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    generated_by_id = {item["id"]: item for item in figures}
    manifest = {
        "kept_in_main_memo": [],
        "moved_to_appendix": [],
        "excluded": [],
    }
    for item in FIGURE_TREATMENTS:
        row = {
            "id": item["id"],
            "title": item["title"],
            "reason": item["reason"],
        }
        if item["id"] in generated_by_id:
            row["path"] = str(generated_by_id[item["id"]]["path"])
            row["relative_path"] = generated_by_id[item["id"]]["relative_path"]
        manifest[item["treatment"]].append(row)
    return manifest


def _render_html(
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    figures: list[dict[str, Any]],
    figure_manifest: dict[str, list[dict[str, Any]]],
) -> str:
    figure_map = {item["id"]: item for item in figures}
    main_sections = [
        _section_1_executive_summary(metrics, figure_map),
        _section_2_what_core4_does(metrics),
        _section_3_universe(metrics, figure_map),
        _section_4_pair_trading_primer(metrics, figure_map),
        _section_5_zscore_and_tests(metrics, figure_map),
        _section_6_strategy_construction(metrics),
        _section_7_country_books_and_pairs(metrics),
        _section_8_country_sleeve_deep_dive(metrics),
        _section_9_portfolio_construction(metrics, figure_map),
        _section_10_historical_performance(metrics, figure_map),
        _section_11_audit_pack(metrics, figure_map),
        _section_12_stress_tests(metrics, figure_map),
        _section_13_from_paper_to_live(metrics),
        _section_14_verdict(metrics),
    ]
    appendix = _section_15_appendix(bundle, metrics, figure_manifest)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(REPORT_TITLE)}</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --paper: #fcfbf8;
      --ink: #1f2a30;
      --muted: #58656a;
      --line: #d8d0c1;
      --accent: #0f4c5c;
      --accent-soft: #dce7eb;
      --warn-bg: #f8efe2;
      --warn-ink: #7a4a15;
      --ok-bg: #e7f1ea;
      --ok-ink: #1f5d3c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #faf7f1 0%, var(--bg) 58%, #eae2d5 100%);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      line-height: 1.58;
    }}
    .page {{
      width: min(1100px, calc(100vw - 28px));
      margin: 24px auto 42px;
      background: var(--paper);
      border: 1px solid rgba(31, 42, 48, 0.08);
      box-shadow: 0 20px 46px rgba(31, 42, 48, 0.08);
      padding: 32px 40px 36px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 2.02rem;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 0.98rem;
    }}
    .lede {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 1.02rem;
    }}
    h2 {{
      margin: 28px 0 10px;
      padding-top: 6px;
      border-top: 1px solid rgba(15, 76, 92, 0.12);
      font-size: 1.28rem;
      line-height: 1.2;
    }}
    h3 {{
      margin: 18px 0 8px;
      color: var(--accent);
      font-size: 1rem;
    }}
    p {{ margin: 10px 0; }}
    pre {{
      margin: 12px 0;
      padding: 12px 14px;
      background: #f6f3ed;
      border: 1px solid rgba(31, 42, 48, 0.08);
      overflow-x: auto;
      font-size: 0.92rem;
    }}
    code {{
      font-family: Consolas, "SFMono-Regular", Menlo, monospace;
      font-size: 0.92em;
      background: rgba(15, 76, 92, 0.05);
      padding: 0.08rem 0.28rem;
      border-radius: 0.22rem;
    }}
    .callout {{
      margin: 14px 0;
      padding: 12px 14px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
    }}
    .callout.warn {{
      background: var(--warn-bg);
      border-left-color: #c77d28;
      color: var(--warn-ink);
    }}
    .callout.ok {{
      background: var(--ok-bg);
      border-left-color: #4a8f65;
      color: var(--ok-ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 16px;
      font-size: 0.94rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 0.81rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      background: rgba(15, 76, 92, 0.03);
    }}
    figure {{
      margin: 16px 0 20px;
      padding: 12px;
      border: 1px solid rgba(31, 42, 48, 0.08);
      background: rgba(255, 255, 255, 0.72);
    }}
    figure img {{
      width: 100%;
      display: block;
      border: 1px solid rgba(31, 42, 48, 0.08);
    }}
    figcaption {{
      margin-top: 8px;
      font-size: 0.88rem;
      color: var(--muted);
    }}
    ul.compact {{
      margin: 10px 0 14px 20px;
      padding: 0;
    }}
    ul.compact li {{
      margin: 5px 0;
    }}
    .small {{
      font-size: 0.84rem;
      color: var(--muted);
      word-break: break-word;
    }}
    @media print {{
      body {{ background: white; }}
      .page {{
        width: auto;
        margin: 0;
        box-shadow: none;
        border: 0;
        padding: 14mm 12mm;
      }}
      figure, table {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>{escape(REPORT_TITLE)}</h1>
      <p class="subtitle">Frozen strategy: <code>{escape(metrics["strategy_id"])}</code> | Validation window: {escape(_validation_window(metrics))} | Status: <code>{escape(metrics["status"])}</code></p>
      <p class="lede">This memo is designed to be readable both as an investment note and as a teaching document. It explains the mechanics of pair trading before it moves into the historical evidence and the implementation limits.</p>
    </header>
    <main id="memo-main">
      {''.join(main_sections)}
    </main>
    <section id="memo-appendix">
      {appendix}
    </section>
  </div>
</body>
</html>
"""


def _render_markdown(
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    figures: list[dict[str, Any]],
    figure_manifest: dict[str, list[dict[str, Any]]],
) -> str:
    figure_map = {item["id"]: item for item in figures}
    parts = [
        f"# {REPORT_TITLE}",
        "",
        f"Frozen strategy: `{metrics['strategy_id']}` | Validation window: {_validation_window(metrics)} | Status: `{metrics['status']}`",
        "",
        "This memo is designed to be readable both as an investment note and as a teaching document. It explains the mechanics of pair trading before it moves into the historical evidence and the implementation limits.",
        "",
        _section_1_executive_summary(metrics, figure_map, html=False),
        _section_2_what_core4_does(metrics, html=False),
        _section_3_universe(metrics, figure_map, html=False),
        _section_4_pair_trading_primer(metrics, figure_map, html=False),
        _section_5_zscore_and_tests(metrics, figure_map, html=False),
        _section_6_strategy_construction(metrics, html=False),
        _section_7_country_books_and_pairs(metrics, html=False),
        _section_8_country_sleeve_deep_dive(metrics, html=False),
        _section_9_portfolio_construction(metrics, figure_map, html=False),
        _section_10_historical_performance(metrics, figure_map, html=False),
        _section_11_audit_pack(metrics, figure_map, html=False),
        _section_12_stress_tests(metrics, figure_map, html=False),
        _section_13_from_paper_to_live(metrics, html=False),
        _section_14_verdict(metrics, html=False),
        _section_15_appendix(bundle, metrics, figure_manifest, html=False),
    ]
    return "\n".join(parts).strip() + "\n"


def _section_1_executive_summary(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    rows = [
        ("Strategy", metrics["strategy_id"]),
        ("Status", metrics["status"]),
        ("Countries", ", ".join(metrics["countries"])),
        ("Validation Period", _validation_window(metrics)),
        ("Annualized Return", _fmt_pct(metrics["validation_metrics"]["annualized_return"])),
        ("Annualized Volatility", _fmt_pct(metrics["validation_metrics"]["annualized_volatility"])),
        ("Sharpe", _fmt_number(metrics["validation_metrics"]["sharpe"])),
        ("Max Drawdown", _fmt_pct(metrics["validation_metrics"]["max_drawdown"])),
        ("Cumulative Return", _fmt_pct(metrics["validation_metrics"]["cumulative_return"])),
        ("Audit Verdict", metrics["audit_verdict"]),
    ]
    paragraphs = [
        "Core 4 is a multi-country equity statistical arbitrage portfolio. It combines four local books: France, Germany, Netherlands, and Sweden.",
        "The frozen object is attractive on history, but the main question for a PM is not only whether the backtest is strong. It is whether the strategy is understandable, documented, and implementable in a disciplined way.",
        "This memo therefore does two jobs. First, it explains the simple mechanics of pair trading, z-scores, and statistical gates. Second, it shows the actual historical evidence for the frozen Core 4 object.",
        "The conclusion is unchanged: Core 4 is paper-ready with limitations. It should enter controlled paper trading, not live trading.",
    ]
    if html:
        body = "".join(f"<p>{escape(text)}</p>" for text in paragraphs)
        body += "<div class='callout ok'><strong>Recommendation:</strong> approve a controlled paper-trading phase, not a live launch.</div>"
        body += _table_html(["Headline Metric", "Value"], rows)
        return f"<section><h2>1. Executive Summary</h2>{body}</section>"
    lines = ["## 1. Executive Summary", ""]
    lines.extend(paragraphs)
    lines.extend(["", "Recommendation: approve a controlled paper-trading phase, not a live launch.", ""])
    lines.append(_table_md(["Headline Metric", "Value"], rows))
    return "\n".join(lines)


def _section_2_what_core4_does(metrics: dict[str, Any], html: bool = True) -> str:
    paragraphs = [
        "At a high level, Core 4 does not bet that European equities will go up or down. It looks for two stocks inside the same local market that usually behave in a related way, waits until that relationship stretches too far, and then bets on partial normalization.",
        "That makes it a relative-value strategy rather than a directional strategy. The trade idea is about convergence between two names, not about forecasting the whole market.",
        "The country split matters. Instead of pretending there is one universal rule for every market, the frozen portfolio keeps four local books and then combines them at the portfolio layer. That is simpler to explain and easier to audit.",
    ]
    if html:
        return "<section><h2>2. What Core 4 Actually Does</h2>" + "".join(f"<p>{escape(text)}</p>" for text in paragraphs) + "</section>"
    return "## 2. What Core 4 Actually Does\n\n" + "\n\n".join(paragraphs)


def _section_3_universe(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    rows = [
        (
            row["country"],
            f"{row['asset_count']:,}",
            f"{row['scan_rows']:,}",
            f"{row['eligible_rows']:,}",
            f"{row['observed_pairs']:,}",
            row["representative_stocks"],
        )
        for row in metrics["universe_rows"]
    ]
    paragraphs = [
        "The strategy starts from local equity universes, not from hand-picked single names. Each country has its own stock pool, its own scan history, and its own realized set of traded pairs.",
        "The table below shows the breadth of the local universes and the kind of names that actually appear in the reconstructed trade ledger. This is important because a stat-arb portfolio should feel like a repeatable process applied to a universe, not like a collection of anecdotes.",
    ]
    if html:
        body = "".join(f"<p>{escape(text)}</p>" for text in paragraphs)
        body += _table_html(
            ["Country", "Approx. stock universe", "Scan pair observations", "Eligible scan rows", "Observed traded pairs", "Representative stocks"],
            rows,
        )
        body += _figure_html(figure_map, "universe_coverage")
        return f"<section><h2>3. Investment Universe And Representative Stocks</h2>{body}</section>"
    return (
        "## 3. Investment Universe And Representative Stocks\n\n"
        + "\n\n".join(paragraphs)
        + "\n\n"
        + _table_md(
            ["Country", "Approx. stock universe", "Scan pair observations", "Eligible scan rows", "Observed traded pairs", "Representative stocks"],
            rows,
        )
        + "\n\n"
        + _figure_md(figure_map, "universe_coverage")
    )


def _section_4_pair_trading_primer(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    paragraphs = [
        "The simplest mental model is an elastic band. If two stocks usually move in a related way, there is a spread between them that often fluctuates around a local norm. When that spread stretches too far, the trade bets on a snap-back toward normality.",
        "The position is market-neutral in spirit: one leg is long, the other is short. That means the trade is supposed to care more about the relative move between the two names than about the absolute direction of the whole market.",
        "A good pair trade therefore needs three things. First, the pair must have enough statistical structure to be worth trading. Second, the spread must move far enough away from normal to justify an entry. Third, the trade needs a disciplined exit if normalization happens, if it fails, or if it simply takes too long.",
    ]
    formula = "spread_t = price_A_t - beta_t * price_B_t"
    if html:
        body = "".join(f"<p>{escape(text)}</p>" for text in paragraphs)
        body += f"<pre><code>{escape(formula)}</code></pre>"
        body += _figure_html(figure_map, "pair_trading_schematic")
        return f"<section><h2>4. Pair Trading Primer</h2>{body}</section>"
    return (
        "## 4. Pair Trading Primer\n\n"
        + "\n\n".join(paragraphs)
        + f"\n\n```text\n{formula}\n```\n\n"
        + _figure_md(figure_map, "pair_trading_schematic")
    )


def _section_5_zscore_and_tests(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    test_rows = [
        ("Correlation", "Do the two stocks co-move enough to be comparable?", "6m correlation floor around 0.30", "Avoids pairs that are just unrelated noise."),
        ("Engle-Granger", "Is there evidence of a long-run relationship?", "p-value <= 0.05", "Rejects many pairs that only look close temporarily."),
        ("ADF on spread", "Does the traded spread look stationary?", "p-value <= 0.05", "Looks for mean reversion rather than persistent drift."),
        ("Half-life", "If the spread moves, does it tend to come back fast enough?", "<= 100 scan days", "Avoids spreads that revert too slowly for a practical trade."),
        ("Ranking and eligibility", "Among valid pairs, which ones are good enough to keep?", "Only ELIGIBLE pairs; keep top 20 candidates", "Turns a broad universe into a manageable tradable set."),
        ("Entry / exit bands", "When do we act on a valid pair?", "Typical frozen logic uses entry 1.8, exit 0.6, stop 3.6", "Requires a real dislocation, not tiny noise."),
    ]
    paragraphs = [
        "The z-score answers a simple question: how unusual is the current spread compared with its own recent history? A z-score of +2 means the spread is roughly two rolling standard deviations above its rolling mean. A z-score of -2 means the opposite.",
        "This matters because a raw move of 1 point means very different things for a quiet pair and a volatile pair. The z-score puts them on a comparable scale.",
        "Core 4 does not trade on z-score alone. The statistical gates sit in front of the trade. They are there to stop the portfolio from treating every temporary gap as a genuine mean-reversion opportunity.",
    ]
    formula = "z_score_t = (spread_t - rolling_mean_t) / rolling_std_t"
    if html:
        body = "".join(f"<p>{escape(text)}</p>" for text in paragraphs)
        body += f"<pre><code>{escape(formula)}</code></pre>"
        body += _table_html(["Test or concept", "Question answered", "Rule used here", "PM interpretation"], test_rows)
        body += _figure_html(figure_map, "zscore_schematic")
        body += _figure_html(figure_map, "statistical_gates_schematic")
        return f"<section><h2>5. Z-Score, Spread And Statistical Tests</h2>{body}</section>"
    return (
        "## 5. Z-Score, Spread And Statistical Tests\n\n"
        + "\n\n".join(paragraphs)
        + f"\n\n```text\n{formula}\n```\n\n"
        + _table_md(["Test or concept", "Question answered", "Rule used here", "PM interpretation"], test_rows)
        + "\n\n"
        + _figure_md(figure_map, "zscore_schematic")
        + "\n\n"
        + _figure_md(figure_map, "statistical_gates_schematic")
    )


def _section_6_strategy_construction(metrics: dict[str, Any], html: bool = True) -> str:
    rows = [
        ("Local universe", "Start from a local country universe rather than from ad hoc pair ideas."),
        ("Scanner", "Evaluate pair relationships over 3m, 6m, and 12m lookbacks."),
        ("Eligibility", "Require statistical quality before a pair can enter the candidate set."),
        ("Ranking", f"Keep the best {metrics['monthly_universe_rules']['top_k']} candidates among eligible pairs."),
        ("Entry", "Wait for a meaningful z-score dislocation before opening a market-neutral pair."),
        ("Exit", "Close on normalization, stop-loss, or time expiry depending on the local configuration."),
        ("Country book", "Aggregate trades into one local sleeve per country."),
        ("Portfolio allocator", "Combine local sleeves with a constrained inverse-volatility reference allocator."),
        ("Audit layer", "Reconstruct positions, ledger, exposures, borrow stress, and execution-delay diagnostics."),
    ]
    intro = "The process is easier to understand if read as a pipeline. A pair does not go directly from a chart idea to a portfolio position."
    if html:
        return f"<section><h2>6. Strategy Construction</h2><p>{escape(intro)}</p>{_table_html(['Step', 'Simple meaning'], rows)}</section>"
    return "## 6. Strategy Construction\n\n" + intro + "\n\n" + _table_md(["Step", "Simple meaning"], rows)


def _section_7_country_books_and_pairs(metrics: dict[str, Any], html: bool = True) -> str:
    config_rows = [
        (
            row["country"],
            row["selected_rule"],
            row["scan_rhythm"],
            str(int(row["z_window"])) if pd.notna(row["z_window"]) else "NaN",
            row["entry_mode"],
            _fmt_number(row["entry_z"]),
            _fmt_number(row["exit_z"]),
            _fmt_number(row["stop_z"]),
            str(int(row["max_holding_days"])) if pd.notna(row["max_holding_days"]) else "NaN",
            str(int(row["max_positions"])) if pd.notna(row["max_positions"]) else "NaN",
        )
        for row in metrics["config_rows"]
    ]
    pair_rows = [
        (
            row["country"],
            row["pair"],
            str(row["ledger_trades"]),
            _fmt_number(row["median_entry_z"]),
            _fmt_number(row["median_corr_6m"]),
            _fmt_number(row["median_half_life_6m"]),
            _fmt_small_number(row["median_adf_pvalue_6m"]),
            _fmt_small_number(row["median_eg_pvalue_6m"]),
            row["typical_exit"],
            row["source_status"],
        )
        for row in metrics["example_pair_rows"]
    ]
    intro = "The frozen portfolio is not four copies of the same book. Each local sleeve has its own scan rhythm, windowing choices, and practical trading profile."
    note = (
        "Representative pairs are selected to illustrate the type of relationships found in the reconstructed ledger. "
        "They should not be interpreted as the best-performing pairs or as standalone trade recommendations. "
        "Exit labels such as TP or SL are descriptive of the reconstructed sample and must be read alongside the full trade distribution."
    )
    if html:
        body = f"<p>{escape(intro)}</p>"
        body += _table_html(
            ["Country", "Selected local rule", "Scan rhythm", "Z window", "Entry mode", "Entry z", "Exit z", "Stop z", "Max hold days", "Max positions"],
            config_rows,
        )
        body += f"<div class='callout'>{escape(note)}</div>"
        body += _table_html(
            ["Country", "Representative pair", "Ledger trades", "Median |entry z|", "Median 6m corr", "Median half-life", "Median ADF p", "Median EG p", "Typical exit", "Source status"],
            pair_rows,
        )
        return f"<section><h2>7. Country Books And Representative Pairs</h2>{body}</section>"
    return (
        "## 7. Country Books And Representative Pairs\n\n"
        + intro
        + "\n\n"
        + _table_md(
            ["Country", "Selected local rule", "Scan rhythm", "Z window", "Entry mode", "Entry z", "Exit z", "Stop z", "Max hold days", "Max positions"],
            config_rows,
        )
        + "\n\n"
        + note
        + "\n\n"
        + _table_md(
            ["Country", "Representative pair", "Ledger trades", "Median |entry z|", "Median 6m corr", "Median half-life", "Median ADF p", "Median EG p", "Typical exit", "Source status"],
            pair_rows,
        )
    )


def _country_evidence(metrics: dict[str, Any], country: str) -> dict[str, Any]:
    return metrics["country_evidence"].get(country.lower(), {})


def _country_role_table_rows(metrics: dict[str, Any]) -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for country in ["france", "germany", "netherlands", "sweden"]:
        evidence = _country_evidence(metrics, country)
        if country == "france":
            rows.append(
                (
                    evidence.get("country", "France"),
                    "Concentrated local alpha sleeve",
                    "Simple, readable, selective baseline with exact ledger source.",
                    "Position concentration because max_positions = 1.",
                    "Does performance survive exact native position logging and exact exit tagging?",
                    "Medium-High",
                )
            )
        elif country == "germany":
            rows.append(
                (
                    evidence.get("country", "Germany"),
                    "Filtered defensive / regime-aware sleeve",
                    "Explicit filter against problematic high-correlation regimes, with stress-trending bypass.",
                    "Governance is weaker because the available row-level source is near_match.",
                    "Can exact Germany ledger rows and signal-state reconstruction be reconciled end-to-end?",
                    "Medium",
                )
            )
        elif country == "netherlands":
            rows.append(
                (
                    evidence.get("country", "Netherlands"),
                    "Diversifying small-universe sleeve",
                    "Distinct pair set despite a smaller universe; helps the portfolio mix.",
                    "Capacity, crowding, and concentration risk in a 16-stock universe.",
                    "Are returns robust after concentration-by-name, pair, turnover, and borrow checks?",
                    "Medium",
                )
            )
        else:
            rows.append(
                (
                    evidence.get("country", "Sweden"),
                    "Differentiated weekly / regime-filter sleeve",
                    "Distinct mechanics and strong diversification potential.",
                    "More engineered rule set than the baseline daily books.",
                    "Is the weekly plus speed-filter edge robust across periods and scan-weekday choices?",
                    "Medium",
                )
            )
    return rows


def _country_standalone_rows(metrics: dict[str, Any]) -> list[tuple[str, str, str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    comments = {
        "france": "Selective baseline sleeve; exact row-level source available.",
        "germany": "Strong standalone profile, but trade count relies on a near_match source.",
        "netherlands": "Weakest standalone return, but diversification case remains plausible.",
        "sweden": "Differentiated weekly/regime-filter sleeve with the deepest trade sample.",
    }
    for country in ["france", "germany", "netherlands", "sweden"]:
        evidence = _country_evidence(metrics, country)
        standalone = evidence.get("standalone", {})
        rows.append(
            (
                evidence.get("country", country.title()),
                _fmt_pct(standalone.get("ann_return")),
                _fmt_pct(standalone.get("ann_vol")),
                _fmt_number(standalone.get("sharpe")),
                _fmt_pct(standalone.get("max_drawdown")),
                _fmt_pct(standalone.get("cumulative_return")),
                str(evidence.get("trade_count", "not_available")),
                comments[country],
            )
        )
    return rows


def _leave_one_out_rows(metrics: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    rows = [
        (
            "Core 4 reference",
            _fmt_pct(metrics["validation_metrics"]["annualized_return"]),
            _fmt_number(metrics["validation_metrics"]["sharpe"]),
            _fmt_pct(metrics["validation_metrics"]["max_drawdown"]),
            "Frozen reference portfolio.",
        )
    ]
    for country in ["france", "germany", "netherlands", "sweden"]:
        evidence = _country_evidence(metrics, country)
        loo = evidence.get("leave_one_out", {})
        interpretation = _interpret_leave_one_out(country, loo)
        rows.append(
            (
                f"Ex-{evidence.get('country', country.title())}",
                _fmt_pct(loo.get("ann_return")),
                _fmt_number(loo.get("sharpe")),
                _fmt_pct(loo.get("max_drawdown")),
                interpretation,
            )
        )
    return rows


def _interpret_leave_one_out(country: str, leave_one_out: dict[str, Any]) -> str:
    if not leave_one_out:
        return "requires_export"
    if country == "germany":
        return "Largest Sharpe loss versus Core 4; strongest dependency signal in the current validation pack."
    if country == "netherlands":
        return "Return rises without Netherlands, but Sharpe falls and drawdown deepens; consistent with a diversifying sleeve."
    if country == "sweden":
        return "Return rises without Sweden, but Sharpe falls materially; suggests useful diversification rather than redundancy."
    return "Removing France lowers return and Sharpe; the sleeve looks additive but not singularly dominant."


def _section_8_country_sleeve_deep_dive(metrics: dict[str, Any], html: bool = True) -> str:
    intro = (
        "This section moves from portfolio-level description to book-level PM interpretation. "
        "The labels below are evidence-based judgments from the frozen config, validation pack, and audit pack; they are not claims of isolated causality."
    )
    role_rows = _country_role_table_rows(metrics)

    france = _country_evidence(metrics, "france")
    germany = _country_evidence(metrics, "germany")
    netherlands = _country_evidence(metrics, "netherlands")
    sweden = _country_evidence(metrics, "sweden")

    sections = [
        (
            "France sleeve",
            [
                f"France acts as a concentrated local alpha sleeve. The frozen configuration keeps <code>max_positions = {int(france.get('max_positions', 1))}</code> inside an approximate {int(france.get('asset_count', 0))}-stock scan universe, and the standalone export reports annualized return {_fmt_pct(france.get('standalone', {}).get('ann_return'))} with Sharpe {_fmt_number(france.get('standalone', {}).get('sharpe'))}.",
                f"The governance advantage is simplicity: the selected local rule is the baseline reference, the row-level trade source is <code>{escape(str(france.get('source_status', 'unknown')))}</code>, and the audit ledger still captures {int(france.get('trade_count', 0))} reconstructed trades across {int(france.get('observed_pairs', 0))} observed pairs. The trade-off is concentration. A selective sleeve with one active slot is easier to explain, but a small number of decisions can matter disproportionately when the book is live.",
                "The main risk is dependence on a limited number of active trades. Before live review, the next checks should be recurring-pair stability, trade-level distribution balance, and whether reconstructed exits remain consistent with exact native exit labels once they are persisted.",
            ],
            "France looks like the cleanest sleeve to explain and audit, but it should be sized mentally as a selective book rather than treated as diversified by itself.",
        ),
        (
            "Germany sleeve",
            [
                "Germany is the most explicitly filtered sleeve in Core 4. The selected local rule excludes pairs with <code>abs(6m_corr) &gt; 0.75</code> except when the latest known scan-date regime is <code>stress_trending</code>. The evidence suggests the intent is to avoid pairs that are too mechanically correlated to produce clean mean-reversion trades, while preserving flexibility in stressed market conditions.",
                f"Economically, Germany matters. The standalone export reports annualized return {_fmt_pct(germany.get('standalone', {}).get('ann_return'))} and Sharpe {_fmt_number(germany.get('standalone', {}).get('sharpe'))}, while the validation summary flags Germany as the strongest leave-one-country-out dependency and the most contributive frozen country. That is useful, but it raises the governance bar rather than lowering it.",
                f"The weakest point is operational, not conceptual. Germany is the only sleeve whose row-level source is marked <code>{escape(str(germany.get('source_status', 'unknown')))}</code> rather than exact. Before live review, exact Germany ledger reconciliation, persisted signal state, and entry/exit reconstruction need to be validated end-to-end.",
            ],
            "Germany may be one of the most valuable sleeves economically, but it currently carries the weakest audit trail. That combination supports paper monitoring, not capital allocation.",
        ),
        (
            "Netherlands sleeve",
            [
                f"Netherlands comes from the smallest local universe in Core 4: about {int(netherlands.get('asset_count', 0))} stocks, versus roughly {int(france.get('asset_count', 0))} in France and {int(germany.get('asset_count', 0))} in Germany. That makes concentration, sector clustering, and idiosyncratic risk the first PM questions.",
                f"At the same time, the sleeve is not obviously redundant. The audit ledger still shows {int(netherlands.get('trade_count', 0))} reconstructed trades across {int(netherlands.get('observed_pairs', 0))} observed pairs, and the leave-one-country-out table improves annualized return without Netherlands but lowers Sharpe and deepens max drawdown. The evidence suggests Netherlands behaves more like a diversifying sleeve than a standalone return engine.",
                "The main implementation risks are capacity and crowding. Before live review, the next export should show concentration by name, concentration by pair, true turnover, and borrow availability at the security level. If those diagnostics are not yet exported, they should be treated as <code>requires_export</code>.",
            ],
            "Netherlands is easier to defend as a diversification sleeve than as a primary alpha source. Its case depends on portfolio mixing benefits surviving concentration and capacity checks.",
        ),
        (
            "Sweden sleeve",
            [
                f"Sweden is the most mechanically distinct sleeve. The frozen reference scans <code>{escape(str(sweden.get('scan_rhythm', 'not_available')))}</code>, uses a shorter <code>z_window = {int(sweden.get('z_window', 0))}</code>, and applies <code>{escape(str(sweden.get('entry_mode', 'not_available')))}</code>. The local metadata also confirms a regime filter in the frozen reference.",
                f"The intuition is readable: trade on a slower scan rhythm, avoid entering spreads that are still accelerating too quickly, and try to capture more stable local relationships. The standalone export reports annualized return {_fmt_pct(sweden.get('standalone', {}).get('ann_return'))} with Sharpe {_fmt_number(sweden.get('standalone', {}).get('sharpe'))}, and the audit ledger shows the deepest sample with {int(sweden.get('trade_count', 0))} reconstructed trades.",
                "The trade-off is that Sweden is more engineered than the other books. That does not invalidate it, but it means the live-readiness burden is higher. Before live review, the next checks should focus on out-of-sample robustness, sensitivity to scan weekday, and stability of the speed filter under native signal persistence.",
            ],
            "Sweden is attractive because it is different, not because it is simpler. A PM should treat that difference as a hypothesis to monitor closely in paper trading.",
        ),
    ]

    if html:
        body = f"<p>{escape(intro)}</p>"
        body += _table_html(
            ["Country", "Portfolio role", "Main strength", "Main weakness", "Key validation question", "Confidence level"],
            role_rows,
        )
        for title, paragraphs, pm_view in sections:
            body += f"<h3>{title}</h3>"
            body += "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
            body += f"<div class='callout'><strong>PM interpretation.</strong> {escape(pm_view)}</div>"
        return f"<section><h2>8. Country Sleeve Deep Dive</h2>{body}</section>"

    lines = [
        "## 8. Country Sleeve Deep Dive",
        "",
        intro,
        "",
        _table_md(
            ["Country", "Portfolio role", "Main strength", "Main weakness", "Key validation question", "Confidence level"],
            role_rows,
        ),
    ]
    for title, paragraphs, pm_view in sections:
        lines.extend(["", f"### {title}", ""])
        lines.extend(paragraphs)
        lines.extend(["", f"PM interpretation: {pm_view}"])
    return "\n".join(lines)


def _section_9_portfolio_construction(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    rows = [
        (
            metrics["reference_allocator"],
            "Frozen reference",
            "Prudent paper-trading candidate with concentration controls.",
            "Selected",
        ),
        (
            metrics["benchmark_allocator"],
            "Control benchmark",
            "Useful comparison point, but not the preferred paper-trading allocator.",
            "Comparative only",
        ),
    ]
    paragraphs = [
        "The portfolio layer is deliberately simpler than the research catalogue behind it. The retained reference allocator is inverse_vol__lb126__weekly__floor_cap, while equal_weight__lb126__monthly__unconstrained is kept as a benchmark only.",
        "That choice is not about maximizing one in-sample number. It is about using a more defensible paper-trading object with clearer control on concentration and less risk of presenting a sample winner as if it were the only sensible construction.",
    ]
    if metrics["reporting_comparison_text"]:
        paragraphs.append(metrics["reporting_comparison_text"])
    if html:
        body = "".join(f"<p>{escape(text)}</p>" for text in paragraphs)
        body += _table_html(["Allocation", "Role", "Rationale", "Status"], rows)
        body += _figure_html(figure_map, "country_correlation_matrix")
        return f"<section><h2>9. Portfolio Construction And Diversification</h2>{body}</section>"
    return (
        "## 9. Portfolio Construction And Diversification\n\n"
        + "\n\n".join(paragraphs)
        + "\n\n"
        + _table_md(["Allocation", "Role", "Rationale", "Status"], rows)
        + "\n\n"
        + _figure_md(figure_map, "country_correlation_matrix")
    )


def _section_10_historical_performance(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    metric_rows = [
        ("Annualized Return", _fmt_pct(metrics["validation_metrics"]["annualized_return"])),
        ("Annualized Volatility", _fmt_pct(metrics["validation_metrics"]["annualized_volatility"])),
        ("Sharpe", _fmt_number(metrics["validation_metrics"]["sharpe"])),
        ("Max Drawdown", _fmt_pct(metrics["validation_metrics"]["max_drawdown"])),
        ("Cumulative Return", _fmt_pct(metrics["validation_metrics"]["cumulative_return"])),
    ]
    yearly_rows = [(str(item["year"]), _fmt_pct(item["return"])) for item in metrics["yearly_returns"]]
    standalone_rows = _country_standalone_rows(metrics)
    leave_one_rows = _leave_one_out_rows(metrics)
    note = (
        "The memo does not re-optimize the strategy. Performance shown here is the frozen research object, and the country diagnostics are read from the exported validation pack rather than recomputed for presentation."
    )
    diagnostic_note = (
        "Country-level standalone metrics and leave-one-country-out diagnostics materially strengthen the PM discussion. "
        "The leave-one-country-out test uses naive equal-weight three-country variants, so it is a dependency check, not an optimizer."
    )
    contribution_note = (
        "The validation summary also identifies Germany as the most contributive frozen country and the strongest current dependency. "
        "That is directionally informative, but it still needs exact native contribution logging before live review."
    )
    if html:
        body = f"<section><h2>10. Historical Performance And Country Diagnostics</h2><p>{escape(note)}</p>"
        body += _table_html(["Metric", "Value"], metric_rows)
        if yearly_rows:
            body += _table_html(["Year", "Return"], yearly_rows)
        body += f"<p>{escape(diagnostic_note)}</p>"
        body += _table_html(
            ["Country", "Annualized return", "Annualized volatility", "Sharpe", "Max drawdown", "Cumulative return", "Number of trades", "Main comment"],
            standalone_rows,
        )
        body += f"<p>{escape(contribution_note)}</p>"
        body += _table_html(
            ["Portfolio variant", "Annualized return", "Sharpe", "Max drawdown", "Interpretation"],
            leave_one_rows,
        )
        body += _figure_html(figure_map, "equity_curve")
        body += _figure_html(figure_map, "drawdown_curve")
        body += "</section>"
        return body
    lines = [
        "## 10. Historical Performance And Country Diagnostics",
        "",
        note,
        "",
        _table_md(["Metric", "Value"], metric_rows),
    ]
    if yearly_rows:
        lines.extend(["", _table_md(["Year", "Return"], yearly_rows)])
    lines.extend(
        [
            "",
            diagnostic_note,
            "",
            _table_md(
                ["Country", "Annualized return", "Annualized volatility", "Sharpe", "Max drawdown", "Cumulative return", "Number of trades", "Main comment"],
                standalone_rows,
            ),
            "",
            contribution_note,
            "",
            _table_md(
                ["Portfolio variant", "Annualized return", "Sharpe", "Max drawdown", "Interpretation"],
                leave_one_rows,
            ),
        ]
    )
    lines.extend(["", _figure_md(figure_map, "equity_curve"), "", _figure_md(figure_map, "drawdown_curve")])
    return "\n".join(lines)


def _section_11_audit_pack(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    rows = [
        ("Frozen config", "Explicit frozen strategy manifest and reference allocation.", "Strong", "The investment object exists and is versionable."),
        ("Historical validation", "Validation pack over 2018-01-01 to 2025-12-31.", "Strong", "Good historical evidence for paper review."),
        ("Daily reporting", "Dedicated reporting exports exist.", "Good", "Useful, but not yet attached to a paper-trading monitoring workflow."),
        ("Daily positions", f"{metrics['reconstructed_positions']:,} reconstructed rows.", "Reconstructed / not native", "Useful for analysis, not a substitute for native position logs."),
        ("Trade ledger", f"{metrics['reconstructed_trades']:,} reconstructed trades across the four books.", "Reconstructed / partial exactness", "Good enough for review, but still not the same as native execution logs."),
        ("Borrow cost", "Aggregate borrow stress grid exists.", "Aggregate proxy", "Comforting, but not security-level borrow evidence."),
        ("Execution delay", "Delay stress exists, but some delays improve the sample.", "Diagnostic only / not order-level", "Cannot be used as proof of executable implementation."),
        ("Paper monitoring", "Still to implement.", "Operational gap", "Needed before live approval."),
    ]
    intro = "This is where the memo shifts from strategy logic to implementation evidence. The important distinction is between exact frozen evidence, reconstructed evidence, and proxy evidence."
    if html:
        body = f"<p>{escape(intro)}</p>"
        body += _table_html(["Area", "Current Evidence", "Status", "PM Interpretation"], rows)
        body += _figure_html(figure_map, "gross_net_exposure")
        return f"<section><h2>11. Audit Pack And Implementation Readiness</h2>{body}</section>"
    return (
        "## 11. Audit Pack And Implementation Readiness\n\n"
        + intro
        + "\n\n"
        + _table_md(["Area", "Current Evidence", "Status", "PM Interpretation"], rows)
        + "\n\n"
        + _figure_md(figure_map, "gross_net_exposure")
    )


def _section_12_stress_tests(
    metrics: dict[str, Any],
    figure_map: dict[str, dict[str, Any]],
    html: bool = True,
) -> str:
    borrow_rows = [
        (
            _borrow_label(str(row["scenario"])),
            _fmt_pct(row["annualized_return_after_borrow"]),
            _fmt_number(row["sharpe_after_borrow"]),
            _fmt_pct(row["max_drawdown_after_borrow"]),
        )
        for row in metrics["borrow_stress_rows"]
    ]
    delay_rows = [
        (
            str(row["scenario"]),
            _fmt_pct(row["annualized_return"]),
            _fmt_pct(row["annualized_volatility"]),
            _fmt_number(row["sharpe"]),
            _fmt_pct(row["max_drawdown"]),
        )
        for row in metrics["execution_delay_rows"]
    ]
    borrow_text = (
        f"Even at 500bps annualized borrow cost, the current proxy still reports annualized return {_fmt_pct(metrics['borrow_500bps'].get('annualized_return_after_borrow'))}, "
        f"Sharpe {_fmt_number(metrics['borrow_500bps'].get('sharpe_after_borrow'))}, and max drawdown {_fmt_pct(metrics['borrow_500bps'].get('max_drawdown_after_borrow'))}. "
        "That is encouraging, but it remains a portfolio-level proxy rather than a security-level borrow model."
    )
    delay_text = (
        "The execution-delay test is not order-level. The fact that some delayed scenarios improve the historical sample is exactly why it should be treated only as a provisional diagnostic."
    )
    if html:
        body = "<section><h2>12. Stress Tests</h2>"
        body += f"<h3>Borrow Cost</h3><p>{escape(borrow_text)}</p>"
        body += _table_html(["Scenario", "Annualized Return", "Sharpe", "Max Drawdown"], borrow_rows)
        body += _figure_html(figure_map, "borrow_stress_ann_return")
        body += f"<h3>Execution Delay</h3><div class='callout warn'>{escape(delay_text)}</div>"
        body += _table_html(["Scenario", "Annualized Return", "Volatility", "Sharpe", "Max Drawdown"], delay_rows)
        body += "</section>"
        return body
    return (
        "## 12. Stress Tests\n\n"
        + "### Borrow Cost\n\n"
        + borrow_text
        + "\n\n"
        + _table_md(["Scenario", "Annualized Return", "Sharpe", "Max Drawdown"], borrow_rows)
        + "\n\n"
        + _figure_md(figure_map, "borrow_stress_ann_return")
        + "\n\n### Execution Delay\n\n"
        + delay_text
        + "\n\n"
        + _table_md(["Scenario", "Annualized Return", "Volatility", "Sharpe", "Max Drawdown"], delay_rows)
    )


def _section_13_from_paper_to_live(metrics: dict[str, Any], html: bool = True) -> str:
    native_logging_rows = [
        "daily positions generated natively, not reconstructed",
        "signal state persisted daily",
        "pair eligibility state persisted daily",
        "z-score, beta, spread, rolling mean/std persisted at decision time",
    ]
    ledger_rows = [
        "exact trade ledger per country",
        "entry date, exit date, entry reason, exit reason",
        "trade-level PnL reconciliation to daily portfolio returns",
        "Germany exact reconciliation",
    ]
    execution_rows = [
        "entry and exit price convention",
        "order timing",
        "slippage model",
        "borrow cost by security",
        "locate / shortability assumptions",
        "delay stress based on true signal state, not proxy",
    ]
    monitoring_rows = [
        "daily NAV",
        "exposures by country",
        "gross and net exposure",
        "active pairs",
        "largest pair/name concentration",
        "drawdown alert",
        "pair decay monitoring",
        "exception report",
    ]
    table_rows = [
        ("Native daily positions", "Reconstructed only.", "No; monitored workaround acceptable.", "Yes.", "High"),
        ("Persisted signal state and pair eligibility", "not_available", "No.", "Yes.", "High"),
        ("Decision-time z-score, beta, spread, mean/std", "not_available", "No.", "Yes.", "High"),
        ("Exact trade ledger per country", "France, Netherlands, Sweden exact; Germany near_match.", "No; limitation accepted for paper.", "Yes.", "High"),
        ("Trade-level PnL reconciliation to daily returns", "Partial.", "No; paper can begin without exact reconciliation.", "Yes.", "High"),
        ("Execution conventions, slippage, borrow, shortability", "Proxy / aggregate only.", "Minimum operating assumptions required.", "Yes.", "High"),
        ("True execution-delay stress from native state", "Proxy only.", "No.", "Yes.", "High"),
        ("Daily paper monitoring pack", "Not yet implemented.", "Yes.", "Yes.", "High"),
    ]

    if html:
        body = "<section><h2>13. From Paper-Ready To Live-Ready</h2>"
        body += "<h3>Native logging</h3><p>Required:</p><ul class='compact'>" + "".join(f"<li>{escape(item)}</li>" for item in native_logging_rows) + "</ul>"
        body += "<h3>Ledger reconciliation</h3><p>Required:</p><ul class='compact'>" + "".join(f"<li>{escape(item)}</li>" for item in ledger_rows) + "</ul>"
        body += "<h3>Execution model</h3><p>Required:</p><ul class='compact'>" + "".join(f"<li>{escape(item)}</li>" for item in execution_rows) + "</ul>"
        body += "<h3>Paper monitoring</h3><p>Required:</p><ul class='compact'>" + "".join(f"<li>{escape(item)}</li>" for item in monitoring_rows) + "</ul>"
        body += _table_html(["Requirement", "Current status", "Needed before paper", "Needed before live", "Priority"], table_rows)
        body += (
            "<div class='callout warn'>Paper trading can start with limitations because the frozen research object is already documented. "
            "Live trading still requires native logs, exact ledger reconciliation, and daily operating controls.</div>"
        )
        return body + "</section>"

    lines = [
        "## 13. From Paper-Ready To Live-Ready",
        "",
        "### Native logging",
        "",
        "Required:",
    ]
    lines.extend(f"- {item}" for item in native_logging_rows)
    lines.extend(["", "### Ledger reconciliation", "", "Required:"])
    lines.extend(f"- {item}" for item in ledger_rows)
    lines.extend(["", "### Execution model", "", "Required:"])
    lines.extend(f"- {item}" for item in execution_rows)
    lines.extend(["", "### Paper monitoring", "", "Required:"])
    lines.extend(f"- {item}" for item in monitoring_rows)
    lines.extend(
        [
            "",
            _table_md(["Requirement", "Current status", "Needed before paper", "Needed before live", "Priority"], table_rows),
            "",
            "Paper trading can start with limitations because the frozen research object is already documented. Live trading still requires native logs, exact ledger reconciliation, and daily operating controls.",
        ]
    )
    return "\n".join(lines)


def _section_14_verdict(metrics: dict[str, Any], html: bool = True) -> str:
    verdict_text = (
        "Core 4 is suitable for controlled paper trading as a frozen research object. It is not yet suitable for live capital allocation. "
        "The historical evidence is sufficiently strong to justify a monitored paper phase, but the operational evidence remains incomplete. "
        "The next phase must test whether the edge survives native signal generation, exact ledger reconstruction, borrow constraints, execution latency, and daily monitoring."
    )
    rows = [
        ("Approve paper trading", "Yes, in a controlled monitored phase."),
        ("Approve live trading", "No"),
        ("Required before live", "Native daily position logs"),
        ("Required before live", "Exact Germany ledger reconciliation"),
        ("Required before live", "Order-level execution and slippage stress"),
        ("Required before live", "Security-level borrow assumptions"),
        ("Required before live", "Paper monitoring and governance controls"),
    ]
    if html:
        body = "<section><h2>14. Verdict And Recommendation</h2>"
        body += f"<div class='callout ok'><strong>Verdict:</strong> <code>{escape(metrics['status'])}</code>. {escape(verdict_text)}</div>"
        body += _table_html(["Decision", "Answer"], rows)
        return body + "</section>"
    return (
        "## 14. Verdict And Recommendation\n\n"
        f"Verdict: `{metrics['status']}`\n\n"
        + verdict_text
        + "\n\n"
        + _table_md(["Decision", "Answer"], rows)
    )


def _section_15_appendix(
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    figure_manifest: dict[str, list[dict[str, Any]]],
    html: bool = True,
) -> str:
    source_rows = [
        ("Freeze manifest", _rel(bundle["paths"]["freeze_path"], bundle["project_root"])),
        ("Portfolio reference", _rel(bundle["paths"]["portfolio_reference_path"], bundle["project_root"])),
        ("Validation summary", _rel(bundle["paths"]["validation_pack_dir"] / "validation_summary.md", bundle["project_root"])),
        ("Audit summary", _rel(bundle["paths"]["audit_pack_dir"] / "audit_pack_summary.md", bundle["project_root"])),
        ("Daily positions", _rel(bundle["paths"]["audit_pack_dir"] / "daily_positions.csv", bundle["project_root"])),
        ("Daily book exposures", _rel(bundle["paths"]["audit_pack_dir"] / "daily_book_exposures.csv", bundle["project_root"])),
        ("Daily portfolio exposures", _rel(bundle["paths"]["audit_pack_dir"] / "daily_portfolio_exposures.csv", bundle["project_root"])),
        ("Trade ledger", _rel(bundle["paths"]["audit_pack_dir"] / "trade_ledger.csv", bundle["project_root"])),
        ("Borrow stress", _rel(bundle["paths"]["audit_pack_dir"] / "borrow_cost_stress.csv", bundle["project_root"])),
        ("Execution delay stress", _rel(bundle["paths"]["audit_pack_dir"] / "execution_delay_stress.csv", bundle["project_root"])),
    ]
    figure_rows = []
    for section_name in ["kept_in_main_memo", "moved_to_appendix", "excluded"]:
        for item in figure_manifest[section_name]:
            figure_rows.append((section_name, item["id"], item["reason"]))
    regen_cmd = (
        "python scripts/build_core4_institutional_memo.py "
        "--audit-pack-dir data/experiments/core4_audit_pack/20260604_222941 "
        "--validation-pack-dir data/experiments/core4_validation_pack/20260514_105610 "
        "--output-dir data/reports/core4_institutional_memo --log-level INFO"
    )
    if html:
        body = "<h2>15. Appendix</h2>"
        body += _table_html(["Source Artifact", "Path"], source_rows)
        body += _table_html(["Figure Treatment", "Figure ID", "Reason"], figure_rows)
        body += "<h3>Known Limitations</h3><ul class='compact'>" + "".join(
            f"<li>{escape(item)}</li>" for item in metrics["known_limitations"]
        ) + "</ul>"
        body += f"<p class='small'><strong>Regeneration command:</strong> <code>{escape(regen_cmd)}</code></p>"
        return body
    return (
        "## 15. Appendix\n\n"
        + _table_md(["Source Artifact", "Path"], source_rows)
        + "\n\n"
        + _table_md(["Figure Treatment", "Figure ID", "Reason"], figure_rows)
        + "\n\nKnown limitations:\n"
        + "\n".join(f"- {item}" for item in metrics["known_limitations"])
        + f"\n\nRegeneration command:\n`{regen_cmd}`"
    )


def _build_manifest(
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    figures: list[dict[str, Any]],
    figure_manifest: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "report_title": REPORT_TITLE,
        "strategy_id": metrics["strategy_id"],
        "status": metrics["status"],
        "section_count": 15,
        "outputs": {
            "html": str(output_dir / OUTPUT_HTML),
            "markdown": str(output_dir / OUTPUT_MARKDOWN),
            "figures_dir": str(output_dir / "figures"),
        },
        "inputs": {
            "freeze_path": _rel(bundle["paths"]["freeze_path"], bundle["project_root"]),
            "portfolio_reference_path": _rel(bundle["paths"]["portfolio_reference_path"], bundle["project_root"]),
            "audit_pack_dir": _rel(bundle["paths"]["audit_pack_dir"], bundle["project_root"]),
            "validation_pack_dir": _rel(bundle["paths"]["validation_pack_dir"], bundle["project_root"]),
            "reporting_dir": _rel(bundle["paths"]["reporting_dir"], bundle["project_root"]),
            "allocation_research_dir": _rel(bundle["paths"]["allocation_research_dir"], bundle["project_root"]),
            "multibook_dir": _rel(bundle["paths"]["multibook_dir"], bundle["project_root"]),
        },
        "figure_count_in_main_memo": len(figures),
        "figure_manifest": figure_manifest,
        "headline_metrics": {
            "validation_period": _validation_window(metrics),
            "annualized_return": metrics["validation_metrics"]["annualized_return"],
            "annualized_volatility": metrics["validation_metrics"]["annualized_volatility"],
            "sharpe": metrics["validation_metrics"]["sharpe"],
            "max_drawdown": metrics["validation_metrics"]["max_drawdown"],
            "cumulative_return": metrics["validation_metrics"]["cumulative_return"],
            "reconstructed_positions": metrics["reconstructed_positions"],
            "reconstructed_trades": metrics["reconstructed_trades"],
            "borrow_500bps": metrics["borrow_500bps"],
        },
        "known_limitations": metrics["known_limitations"],
    }


def _table_html(headers: list[str], rows: list[tuple[Any, ...]] | list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _table_md(headers: list[str], rows: list[tuple[Any, ...]] | list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _figure_html(figure_map: dict[str, dict[str, Any]], figure_id: str) -> str:
    item = figure_map.get(figure_id)
    if not item:
        return ""
    return (
        f"<figure><img src='{escape(item['relative_path'])}' alt='{escape(item['title'])}' />"
        f"<figcaption>{escape(item['title'])}</figcaption></figure>"
    )


def _figure_md(figure_map: dict[str, dict[str, Any]], figure_id: str) -> str:
    item = figure_map.get(figure_id)
    if not item:
        return ""
    return f"![{item['title']}]({item['relative_path']})"


def _validation_window(metrics: dict[str, Any]) -> str:
    period = metrics["validation_period"]
    return f"{period['start']} to {period['end']}"


def _scan_rhythm_label(reference: dict[str, Any]) -> str:
    frequency = str(reference.get("scan_frequency", "")).lower()
    weekday = str(reference.get("scan_weekday", "")).upper()
    if frequency == "weekly" and weekday:
        return f"Weekly ({weekday})"
    if frequency == "daily":
        return "Daily"
    if frequency:
        return frequency.title()
    return "Unavailable"


def _selected_rule(country: str, config_name: str, metadata: dict[str, Any]) -> str:
    if country == "germany":
        for version in metadata.get("versions", []):
            if str(version.get("config_name")) == str(config_name):
                return str(version.get("rule", config_name))
    labels = {
        "reference": "Baseline local mean-reversion rule",
        "best_plus_regime_filter": "Baseline plus simple regime filter",
        "pair_filter_corr_abs_le_0p75_bypass_scan_stress_trending": "Pair filter corr<=0.75 with stress-trending scan bypass",
    }
    return labels.get(str(config_name), str(config_name))


def _entry_mode_label(value: Any) -> str:
    labels = {
        "baseline_entry": "Baseline z-score entry",
        "entry_zspeed_ewma_cap": "Z-score entry with speed cap",
    }
    text = str(value)
    return labels.get(text, text.replace("_", " ").title())


def _country_role_label(country: str) -> str:
    return "Diversification sleeve" if country == "netherlands" else "Core sleeve"


def _country_pm_comment(country: str) -> str:
    comments = {
        "france": "Clean baseline core sleeve.",
        "germany": "Important contributor, promoted after validation, but still requires ledger reconciliation before live.",
        "netherlands": "Retained for diversification despite weaker standalone profile.",
        "sweden": "Validated enhanced local sleeve with a simple regime overlay.",
    }
    return comments.get(country, "Retained in the frozen portfolio.")


def _country_sort_key(country: str) -> int:
    order = {
        "france": 0,
        "germany": 1,
        "netherlands": 2,
        "sweden": 3,
    }
    return order.get(country.lower(), 99)


def _borrow_label(scenario: str) -> str:
    match = re.search(r"borrow_([0-9]+)bps", scenario)
    return f"{match.group(1)} bps" if match else scenario


def _pretty_pair(pair_id: str) -> str:
    if not pair_id:
        return pair_id
    parts = str(pair_id).split("_")
    if len(parts) >= 4:
        midpoint = len(parts) // 2
        left = " ".join(parts[:midpoint])
        right = " ".join(parts[midpoint:])
        return f"{left} / {right}"
    return str(pair_id).replace("_", " ")


def _pretty_asset(asset: str) -> str:
    return str(asset).replace("_", " ")


def _pair_label_from_rows(pair_source: pd.DataFrame, ledger_pair: pd.DataFrame, fallback: str) -> str:
    for left_col, right_col in [("asset_1", "asset_2"), ("asset_left", "asset_right"), ("leg_1", "leg_2")]:
        frame = pair_source if left_col in pair_source.columns and right_col in pair_source.columns else ledger_pair
        if left_col in frame.columns and right_col in frame.columns and not frame.empty:
            left = _pretty_asset(str(frame.iloc[0][left_col]))
            right = _pretty_asset(str(frame.iloc[0][right_col]))
            return f"{left} / {right}"
    return _pretty_pair(fallback)


def _rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _extract_after_colon(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _extract_pct(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.\-]+)%", text)
    return float(match.group(1)) / 100.0 if match else np.nan


def _extract_number(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.\-]+)", text)
    return float(match.group(1)) if match else np.nan


def _fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.2f}"


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{100.0 * float(value):.2f}%"


def _fmt_small_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"
