from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sweden_edge_decomposition_campaign import (
    ASSET_REGISTRY_PATH,
    BASE_SELECTION_MODE,
    BASE_SELECTION_VARIANT,
    DATA_PATH,
    UNIVERSE,
    build_universe_assets,
    load_or_build_scans,
)
from scripts.run_sweden_filter_ablation_campaign import (
    AblationConfig,
    FilterThresholds,
    build_ablation_configs,
    build_concentration_comparison,
    build_slot_utilization_output,
    build_trade_level_comparison,
    build_variant_vs_best_comparison,
    config_to_dict,
    derive_filter_thresholds,
    find_reference_output,
    load_reference_metadata,
    run_config,
)
from utils.edge_decomposition import (
    REGIME_RULES_DESCRIPTION,
    build_pair_level_summary,
    build_trade_diagnostics,
    compute_market_regime_features,
    load_price_panel,
    summarize_edge_by_segment,
)


LOGGER = logging.getLogger("sweden_filter_robustness")

DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"
SCAN_CACHE_LOAD_START = "2018-01-05"
REFERENCE_ABLATION_DIR_NAME = "sweden_filter_ablation_20180101_20251231_20260418_224839"

ROBUSTNESS_VARIANTS = {
    "best_reference",
    "best_plus_regime_filter",
    "best_plus_pair_filter",
    "best_plus_regime_entry",
}

REGIME_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
)

STRUCTURE_SEGMENT_COLS = (
    "market_regime",
    "stress_bucket",
    "trending_bucket",
    "neutral_bucket",
    "corr_type",
    "beta_stability_bucket",
    "abs_z_entry_quintile",
    "z_speed_ewma_quintile",
    "exit_reason_bucket",
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    label: str
    start: str
    end: str


SPLITS = (
    SplitSpec("split_1_old", "2018_2020", "2018-01-01", "2020-12-31"),
    SplitSpec("split_2_mid", "2021_2023", "2021-01-01", "2023-12-31"),
    SplitSpec("split_3_recent", "2024_2025", "2024-01-01", "2025-12-31"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal robustness campaign for Sweden filter ablation variants."
    )
    parser.add_argument(
        "--reference-output",
        default=None,
        help="Reference edge-decomposition output if ablation thresholds are not available.",
    )
    parser.add_argument(
        "--ablation-output",
        default=None,
        help="Previous Sweden filter ablation output folder to reuse exact thresholds.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "experiments"),
        help="Root folder for campaign outputs.",
    )
    parser.add_argument("--output-suffix", default=None, help="Optional suffix appended to output directory.")
    parser.add_argument("--rebuild-scans", action="store_true", help="Rebuild Sweden scans.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short smoke split only, keeping the exact variant logic.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def selected_configs() -> list[AblationConfig]:
    return [c for c in build_ablation_configs() if c.name in ROBUSTNESS_VARIANTS]


def smoke_splits() -> tuple[SplitSpec, ...]:
    return (SplitSpec("smoke_2025_q1", "smoke_2025_q1", "2025-01-01", "2025-03-31"),)


def build_output_dir(*, output_root: Path, suffix: str | None, smoke: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"sweden_filter_robustness_20180101_20251231_{stamp}"
    if smoke:
        name = f"{name}_smoke"
    if suffix:
        name = f"{name}_{suffix}"
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_exact_thresholds(
    *,
    output_root: Path,
    explicit_ablation: str | None,
    explicit_reference: str | None,
) -> tuple[FilterThresholds, dict[str, Any], Path | None]:
    ablation_dir = resolve_ablation_output(output_root, explicit_ablation)
    if ablation_dir is not None:
        meta_path = ablation_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            thresholds = meta.get("thresholds", {})
            try:
                return (
                    FilterThresholds(
                        abs_z_extreme_min=float(thresholds["abs_z_extreme_min"]),
                        zspeed_ewma_extreme_min=float(thresholds["zspeed_ewma_extreme_min"]),
                        beta_stability_degraded_min=float(thresholds["beta_stability_degraded_min"]),
                        source_dir=ablation_dir,
                    ),
                    meta,
                    ablation_dir,
                )
            except Exception:
                LOGGER.warning("Could not parse thresholds from %s; falling back to edge reference.", meta_path)

    edge_dir = find_reference_output(output_root, explicit_reference)
    return derive_filter_thresholds(edge_dir), load_reference_metadata(edge_dir), ablation_dir


def resolve_ablation_output(output_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Ablation output not found: {path}")
        return path

    preferred = output_root / REFERENCE_ABLATION_DIR_NAME
    if preferred.exists():
        return preferred

    candidates = [
        p
        for p in output_root.glob("sweden_filter_ablation_*")
        if (p / "metadata.json").exists() and not p.name.endswith("_smoke")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def add_split_columns(df: pd.DataFrame, split: SplitSpec) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "split_name", split.name)
    out.insert(1, "split_label", split.label)
    out.insert(2, "split_start", split.start)
    out.insert(3, "split_end", split.end)
    return out


def run_split(
    *,
    split: SplitSpec,
    configs: list[AblationConfig],
    scans: pd.DataFrame,
    thresholds: FilterThresholds,
    price_panel: pd.DataFrame,
    market_features: pd.DataFrame,
    asset_metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    LOGGER.info("Running split=%s %s -> %s", split.name, split.start, split.end)
    runs: list[dict[str, Any]] = []
    enriched_frames: list[pd.DataFrame] = []

    for config in configs:
        run = run_config(
            config=config,
            base_scans=scans,
            thresholds=thresholds,
            market_features=market_features,
            start=split.start,
            end=split.end,
        )
        runs.append(run)
        enriched = build_trade_diagnostics(
            trades=run["result"]["trades"],
            config_name=config.name,
            params=run["params"],
            scans=run["scans"],
            scan_usage=run["result"].get("scan_usage", pd.DataFrame()),
            price_panel=price_panel,
            market_features=market_features,
            ranking_mode=f"{BASE_SELECTION_MODE}:{BASE_SELECTION_VARIANT}",
            asset_metadata=asset_metadata,
        )
        enriched_frames.append(enriched)
        LOGGER.info("Split %s config %s enriched trades: %d", split.name, config.name, len(enriched))

    enriched = pd.concat(enriched_frames, ignore_index=True) if enriched_frames else pd.DataFrame()
    if enriched.empty:
        raise RuntimeError(f"No enriched trades for split={split.name}")

    trade_level = add_split_columns(build_trade_level_comparison(enriched), split)
    trade_level = trade_level[trade_level["config_name"].isin(ROBUSTNESS_VARIANTS)].reset_index(drop=True)

    portfolio = pd.DataFrame([r["result"]["stats"] for r in runs])
    portfolio = add_split_columns(portfolio, split)

    concentration = add_split_columns(build_concentration_comparison(enriched), split)
    concentration = concentration[concentration["config_name"].isin(ROBUSTNESS_VARIANTS)].reset_index(drop=True)

    exit_behavior = add_split_columns(summarize_edge_by_segment(enriched, ("exit_reason_bucket",)), split)
    regime_breakdown = add_split_columns(summarize_edge_by_segment(enriched, REGIME_SEGMENT_COLS), split)
    segment_breakdown = add_split_columns(summarize_edge_by_segment(enriched, STRUCTURE_SEGMENT_COLS), split)
    pair_level = add_split_columns(build_pair_level_summary(enriched), split)
    monthly = add_split_columns(build_split_monthly_output(runs), split)
    slot_utilization = add_split_columns(build_slot_utilization_output(runs), split)
    filter_diag = add_split_columns(
        pd.concat(
            [r["result"].get("filter_diagnostics", pd.DataFrame()) for r in runs],
            ignore_index=True,
            sort=False,
        ),
        split,
    )
    variant_vs_best = add_split_columns(
        build_variant_vs_best_comparison(segment_breakdown, best_config="best_reference"),
        split,
    )

    return {
        "trades_enriched": add_split_columns(enriched, split),
        "trade_level": trade_level,
        "portfolio_level": portfolio,
        "concentration": concentration,
        "exit_behavior": exit_behavior,
        "regime_breakdown": regime_breakdown,
        "segment_breakdown": segment_breakdown,
        "pair_level": pair_level,
        "monthly_returns": monthly,
        "slot_utilization": slot_utilization,
        "filter_diagnostics": filter_diag,
        "variant_vs_best": variant_vs_best,
    }


def build_split_monthly_output(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        cfg = run["config"]
        monthly = run["result"].get("monthly", pd.DataFrame()).copy()
        if monthly.empty:
            continue
        monthly.insert(0, "config_name", cfg.name)
        monthly.insert(1, "variant", cfg.letter)
        frames.append(monthly)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_robustness_scorecard(
    *,
    trade_level: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    concentration: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_order = list(SPLITS)
    if set(trade_level["split_name"].unique()) == {"smoke_2025_q1"}:
        split_order = [SplitSpec("smoke_2025_q1", "smoke_2025_q1", "2025-01-01", "2025-03-31")]

    best_trade = trade_level[trade_level["config_name"] == "best_reference"].set_index("split_name")
    best_port = portfolio_level[portfolio_level["config_name"] == "best_reference"].set_index("split_name")

    for config_name in sorted(portfolio_level["config_name"].dropna().unique()):
        p = portfolio_level[portfolio_level["config_name"] == config_name].copy()
        t = trade_level[trade_level["config_name"] == config_name].copy()
        c = concentration[concentration["config_name"] == config_name].copy()

        sharpes = pd.to_numeric(p["engine_sharpe"], errors="coerce")
        returns = pd.to_numeric(p["total_return_engine"], errors="coerce")
        dds = pd.to_numeric(p["engine_max_drawdown"], errors="coerce")
        avg_pos = pd.to_numeric(p["avg_open_positions"], errors="coerce")
        avg_pnl = pd.to_numeric(t["avg_pnl_per_trade"], errors="coerce")
        breadth = pd.to_numeric(c["nb_paires_tradees"], errors="coerce")

        out_sharpe = 0
        out_total_pnl = 0
        out_avg_pnl = 0
        lower_dd = 0
        for split in split_order:
            sname = split.name
            row_p = p[p["split_name"] == sname]
            row_t = t[t["split_name"] == sname]
            if sname not in best_port.index or row_p.empty:
                continue
            b_p = best_port.loc[sname]
            if _safe_float(row_p.iloc[0].get("engine_sharpe")) > _safe_float(b_p.get("engine_sharpe")):
                out_sharpe += 1
            if abs(_safe_float(row_p.iloc[0].get("engine_max_drawdown"))) < abs(_safe_float(b_p.get("engine_max_drawdown"))):
                lower_dd += 1
            if sname in best_trade.index and not row_t.empty:
                b_t = best_trade.loc[sname]
                if _safe_float(row_t.iloc[0].get("total_pnl")) > _safe_float(b_t.get("total_pnl")):
                    out_total_pnl += 1
                if _safe_float(row_t.iloc[0].get("avg_pnl_per_trade")) > _safe_float(b_t.get("avg_pnl_per_trade")):
                    out_avg_pnl += 1

        anomaly_count = int(p.get("anomaly_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        rows.append(
            {
                "config_name": config_name,
                "variant": str(p["variant"].iloc[0]) if "variant" in p.columns and not p.empty else "",
                "mean_sharpe_across_splits": float(sharpes.mean()) if sharpes.notna().any() else np.nan,
                "min_sharpe_across_splits": float(sharpes.min()) if sharpes.notna().any() else np.nan,
                "max_sharpe_across_splits": float(sharpes.max()) if sharpes.notna().any() else np.nan,
                "sharpe_std_across_splits": float(sharpes.std(ddof=1)) if sharpes.notna().sum() > 1 else np.nan,
                "mean_total_return_across_splits": float(returns.mean()) if returns.notna().any() else np.nan,
                "mean_avg_pnl_trade_across_splits": float(avg_pnl.mean()) if avg_pnl.notna().any() else np.nan,
                "mean_max_dd_across_splits": float(dds.mean()) if dds.notna().any() else np.nan,
                "mean_avg_positions_across_splits": float(avg_pos.mean()) if avg_pos.notna().any() else np.nan,
                "mean_nb_pairs_traded_across_splits": float(breadth.mean()) if breadth.notna().any() else np.nan,
                "anomaly_count": anomaly_count,
                "splits_outperforming_B_on_sharpe": out_sharpe,
                "splits_outperforming_B_on_total_pnl": out_total_pnl,
                "splits_outperforming_B_on_avg_pnl_trade": out_avg_pnl,
                "splits_with_lower_dd_than_B": lower_dd,
            }
        )

    out = pd.DataFrame(rows)
    out["robustness_comment"] = out.apply(_robustness_comment, axis=1)
    return out.sort_values(["variant"]).reset_index(drop=True)


def _robustness_comment(row: pd.Series) -> str:
    name = str(row.get("config_name", ""))
    if name == "best_reference":
        return "Reference variant B."
    sharpe_wins = int(row.get("splits_outperforming_B_on_sharpe", 0))
    pnl_wins = int(row.get("splits_outperforming_B_on_total_pnl", 0))
    anomaly_count = int(row.get("anomaly_count", 0))
    avg_pairs = _safe_float(row.get("mean_nb_pairs_traded_across_splits"))
    if sharpe_wins == 3 and pnl_wins >= 2 and anomaly_count == 0 and avg_pairs >= 50:
        return "Robust candidate versus B across splits."
    if sharpe_wins == 3 and avg_pairs < 50:
        return "Sharpe robust, but breadth is materially reduced."
    if sharpe_wins >= 2:
        return "Partially robust; inspect weak split before promotion."
    return "Not robust versus B across splits."


def build_rolling_12m_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for (split_name, config_name), group in monthly.groupby(["split_name", "config_name"], dropna=False):
        g = group.copy()
        g["trade_month"] = pd.to_datetime(g["trade_month"], format="%Y-%m", errors="coerce")
        g = g.sort_values("trade_month")
        r = pd.to_numeric(g["month_return"], errors="coerce")
        g["rolling_12m_return"] = (1.0 + r).rolling(12, min_periods=6).apply(np.prod, raw=True) - 1.0
        g["rolling_12m_avg_return"] = r.rolling(12, min_periods=6).mean()
        frames.append(g[["split_name", "split_label", "config_name", "variant", "trade_month", "rolling_12m_return", "rolling_12m_avg_return"]])
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_campaign_summary(
    *,
    scorecard: pd.DataFrame,
    trade_level: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    concentration: pd.DataFrame,
    thresholds: FilterThresholds,
    splits: tuple[SplitSpec, ...],
) -> str:
    lines = [
        "Sweden filter robustness campaign",
        "",
        "Splits:",
        *[f"- {s.name}: {s.start} -> {s.end}" for s in splits],
        "",
        "Variants: B=best_reference, C=best_plus_regime_filter, E=best_plus_pair_filter, F=best_plus_regime_entry.",
        "",
        "Fixed thresholds reused from ablation:",
        f"- abs_z_entry >= {thresholds.abs_z_extreme_min:.6f}",
        f"- z_speed_ewma >= {thresholds.zspeed_ewma_extreme_min:.6f}",
        f"- beta_std >= {thresholds.beta_stability_degraded_min:.6f}",
        "",
        "Regime rules:",
        f"- {REGIME_RULES_DESCRIPTION}",
        "",
        "Robustness scorecard:",
        _compact(scorecard, [
            "variant",
            "config_name",
            "mean_sharpe_across_splits",
            "min_sharpe_across_splits",
            "sharpe_std_across_splits",
            "splits_outperforming_B_on_sharpe",
            "splits_outperforming_B_on_total_pnl",
            "anomaly_count",
            "robustness_comment",
        ]),
        "",
        "Portfolio by split:",
        _compact(portfolio_level, [
            "split_name",
            "variant",
            "config_name",
            "total_return_engine",
            "engine_sharpe",
            "engine_max_drawdown",
            "avg_open_positions",
            "anomaly_flag",
        ]),
        "",
        "Trade-level by split:",
        _compact(trade_level, [
            "split_name",
            "variant",
            "config_name",
            "nb_trades",
            "total_pnl",
            "avg_pnl_per_trade",
            "win_rate",
            "nb_sl",
            "nb_time",
        ]),
        "",
        "Concentration by split:",
        _compact(concentration, [
            "split_name",
            "variant",
            "config_name",
            "nb_paires_tradees",
            "gross_profit",
            "gross_loss",
            "top5_share_net_pnl",
            "top10_share_net_pnl",
        ]),
    ]
    return "\n".join(lines) + "\n"


def build_conclusion(scorecard: pd.DataFrame, portfolio: pd.DataFrame, trade: pd.DataFrame, concentration: pd.DataFrame) -> str:
    lines = [
        "Conclusion",
        "",
        "Questions:",
        f"1. C stable vs B: {_answer_stability(scorecard, 'best_plus_regime_filter')}",
        f"2. F vs C: {_answer_f_vs_c(portfolio, trade)}",
        f"3. E over-filtering risk: {_answer_e(portfolio, concentration, scorecard)}",
        f"4. Most temporally stable variant: {_most_stable(scorecard)}",
        f"5. Best Sharpe/DD/breadth/slot compromise: {_best_compromise(scorecard, portfolio, concentration)}",
        f"6. Recent-period concentration: {_recent_concentration(portfolio)}",
        f"7. Split collapse: {_split_collapse(portfolio)}",
        f"8. Anomaly flag stability: {_anomaly_answer(scorecard)}",
        f"9. Next broader-validation candidate: {_next_candidate(scorecard, portfolio, concentration)}",
        f"10. Variant to discard despite strong global metrics: {_discard_candidate(scorecard, portfolio, concentration)}",
        "",
        "Methodological note:",
        "- No threshold was retuned in this campaign. Splits are independent backtest runs over their own start/end dates.",
    ]
    return "\n".join(lines) + "\n"


def _answer_stability(scorecard: pd.DataFrame, config: str) -> str:
    r = _row(scorecard, config)
    if r.empty:
        return "unavailable."
    return (
        f"{int(r.splits_outperforming_B_on_sharpe)}/3 Sharpe wins, "
        f"{int(r.splits_outperforming_B_on_total_pnl)}/3 total-pnl wins, "
        f"min Sharpe={_safe_float(r.min_sharpe_across_splits):.3f}."
    )


def _answer_f_vs_c(portfolio: pd.DataFrame, trade: pd.DataFrame) -> str:
    p_f = portfolio[portfolio["config_name"] == "best_plus_regime_entry"].set_index("split_name")
    p_c = portfolio[portfolio["config_name"] == "best_plus_regime_filter"].set_index("split_name")
    t_f = trade[trade["config_name"] == "best_plus_regime_entry"].set_index("split_name")
    t_c = trade[trade["config_name"] == "best_plus_regime_filter"].set_index("split_name")
    if p_f.empty or p_c.empty:
        return "unavailable."
    sharpe_wins = sum(_safe_float(p_f.loc[s, "engine_sharpe"]) > _safe_float(p_c.loc[s, "engine_sharpe"]) for s in p_f.index.intersection(p_c.index))
    pnl_wins = sum(_safe_float(t_f.loc[s, "total_pnl"]) > _safe_float(t_c.loc[s, "total_pnl"]) for s in t_f.index.intersection(t_c.index))
    return f"F beats C on Sharpe in {sharpe_wins}/3 splits and on trade total_pnl in {pnl_wins}/3 splits."


def _answer_e(portfolio: pd.DataFrame, concentration: pd.DataFrame, scorecard: pd.DataFrame) -> str:
    r = _row(scorecard, "best_plus_pair_filter")
    c = concentration[concentration["config_name"] == "best_plus_pair_filter"]
    if r.empty or c.empty:
        return "unavailable."
    return (
        f"Sharpe wins={int(r.splits_outperforming_B_on_sharpe)}/3, "
        f"mean pairs={_safe_float(r.mean_nb_pairs_traded_across_splits):.1f}; "
        "breadth must be checked before promotion."
    )


def _most_stable(scorecard: pd.DataFrame) -> str:
    d = scorecard[scorecard["config_name"] != "best_reference"].copy()
    if d.empty:
        return "unavailable."
    d = d.sort_values(["sharpe_std_across_splits", "min_sharpe_across_splits"], ascending=[True, False])
    r = d.iloc[0]
    return f"{r.config_name} (Sharpe std={_safe_float(r.sharpe_std_across_splits):.3f}, min Sharpe={_safe_float(r.min_sharpe_across_splits):.3f})."


def _best_compromise(scorecard: pd.DataFrame, portfolio: pd.DataFrame, concentration: pd.DataFrame) -> str:
    candidates = scorecard[scorecard["config_name"] != "best_reference"].copy()
    if candidates.empty:
        return "unavailable."
    candidates["_score"] = (
        candidates["splits_outperforming_B_on_sharpe"].fillna(0)
        + candidates["splits_outperforming_B_on_total_pnl"].fillna(0)
        + candidates["splits_with_lower_dd_than_B"].fillna(0)
        - (candidates["mean_avg_positions_across_splits"].fillna(0) < 1.0).astype(int)
    )
    r = candidates.sort_values(["_score", "mean_nb_pairs_traded_across_splits", "min_sharpe_across_splits"], ascending=[False, False, False]).iloc[0]
    return f"{r.config_name} by scorecard balance."


def _recent_concentration(portfolio: pd.DataFrame) -> str:
    recent = portfolio[portfolio["split_name"] == "split_3_recent"].copy()
    if recent.empty:
        return "unavailable."
    winners = recent.sort_values("engine_sharpe", ascending=False)[["config_name", "engine_sharpe"]].head(2)
    return "recent top Sharpe: " + ", ".join(f"{r.config_name}={_safe_float(r.engine_sharpe):.3f}" for r in winners.itertuples(index=False))


def _split_collapse(portfolio: pd.DataFrame) -> str:
    d = portfolio[portfolio["config_name"] != "best_reference"].copy()
    if d.empty:
        return "unavailable."
    worst = d.sort_values("engine_sharpe").iloc[0]
    return f"worst non-B split is {worst.config_name} on {worst.split_name}, Sharpe={_safe_float(worst.engine_sharpe):.3f}."


def _anomaly_answer(scorecard: pd.DataFrame) -> str:
    parts = []
    for r in scorecard.itertuples(index=False):
        parts.append(f"{r.variant}:{int(r.anomaly_count)}")
    return "anomaly counts by variant = " + ", ".join(parts) + "."


def _next_candidate(scorecard: pd.DataFrame, portfolio: pd.DataFrame, concentration: pd.DataFrame) -> str:
    d = scorecard[(scorecard["config_name"] != "best_reference") & (scorecard["anomaly_count"] == 0)].copy()
    if d.empty:
        return "none without anomaly."
    d = d.sort_values(["splits_outperforming_B_on_sharpe", "splits_outperforming_B_on_total_pnl", "mean_nb_pairs_traded_across_splits"], ascending=[False, False, False])
    return str(d.iloc[0]["config_name"])


def _discard_candidate(scorecard: pd.DataFrame, portfolio: pd.DataFrame, concentration: pd.DataFrame) -> str:
    d = scorecard[scorecard["config_name"].isin(["best_plus_pair_filter", "best_plus_regime_entry"])].copy()
    if d.empty:
        return "unavailable."
    d = d.sort_values("mean_nb_pairs_traded_across_splits")
    return f"{d.iloc[0].config_name} if breadth/slot utilization is deemed too low."


def _compact(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "(empty)"
    keep = [c for c in cols if c in df.columns]
    return df[keep].to_string(index=False) if keep else df.to_string(index=False)


def _row(df: pd.DataFrame, config: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    r = df[df["config_name"] == config]
    return r.iloc[0] if not r.empty else pd.Series(dtype=object)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def write_outputs(
    *,
    out_dir: Path,
    frames_by_name: dict[str, list[pd.DataFrame]],
    scorecard: pd.DataFrame,
    thresholds: FilterThresholds,
    threshold_metadata: dict[str, Any],
    ablation_dir: Path | None,
    splits: tuple[SplitSpec, ...],
) -> dict[str, Path]:
    combined = {
        name: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for name, frames in frames_by_name.items()
    }
    rolling = build_rolling_12m_summary(combined["monthly_returns"])
    manifest = pd.DataFrame([config_to_dict(c) for c in selected_configs()])

    paths = {
        "campaign_summary": out_dir / "campaign_summary.txt",
        "conclusion": out_dir / "conclusion.txt",
        "config_manifest": out_dir / "config_manifest.csv",
        "metadata": out_dir / "metadata.json",
        "split_trade_level": out_dir / "split_trade_level.csv",
        "split_portfolio_level": out_dir / "split_portfolio_level.csv",
        "split_concentration": out_dir / "split_concentration.csv",
        "split_exit_behavior": out_dir / "split_exit_behavior.csv",
        "split_regime_breakdown": out_dir / "split_regime_breakdown.csv",
        "split_monthly_returns": out_dir / "split_monthly_returns.csv",
        "robustness_scorecard": out_dir / "robustness_scorecard.csv",
        "split_slot_utilization": out_dir / "split_slot_utilization.csv",
        "split_pair_level_summary": out_dir / "split_pair_level_summary.csv",
        "variant_vs_best_by_split": out_dir / "variant_vs_best_by_split.csv",
        "rolling_12m_summary": out_dir / "rolling_12m_summary.csv",
        "split_filter_diagnostics": out_dir / "split_filter_diagnostics.csv",
        "split_trades_enriched": out_dir / "split_trades_enriched.csv",
        "split_segment_breakdown": out_dir / "split_segment_breakdown.csv",
    }

    combined["trade_level"].to_csv(paths["split_trade_level"], index=False)
    combined["portfolio_level"].to_csv(paths["split_portfolio_level"], index=False)
    combined["concentration"].to_csv(paths["split_concentration"], index=False)
    combined["exit_behavior"].to_csv(paths["split_exit_behavior"], index=False)
    combined["regime_breakdown"].to_csv(paths["split_regime_breakdown"], index=False)
    combined["monthly_returns"].to_csv(paths["split_monthly_returns"], index=False)
    scorecard.to_csv(paths["robustness_scorecard"], index=False)
    combined["slot_utilization"].to_csv(paths["split_slot_utilization"], index=False)
    combined["pair_level"].to_csv(paths["split_pair_level_summary"], index=False)
    combined["variant_vs_best"].to_csv(paths["variant_vs_best_by_split"], index=False)
    rolling.to_csv(paths["rolling_12m_summary"], index=False)
    combined["filter_diagnostics"].to_csv(paths["split_filter_diagnostics"], index=False)
    combined["trades_enriched"].to_csv(paths["split_trades_enriched"], index=False)
    combined["segment_breakdown"].to_csv(paths["split_segment_breakdown"], index=False)
    manifest.to_csv(paths["config_manifest"], index=False)

    metadata = {
        "universe": UNIVERSE,
        "global_start": DEFAULT_START,
        "global_end": DEFAULT_END,
        "splits": [s.__dict__ for s in splits],
        "variants": sorted(ROBUSTNESS_VARIANTS),
        "ablation_output_source": str(ablation_dir) if ablation_dir else None,
        "threshold_source": str(thresholds.source_dir),
        "threshold_metadata": threshold_metadata,
        "thresholds": {
            "abs_z_extreme_min": thresholds.abs_z_extreme_min,
            "zspeed_ewma_extreme_min": thresholds.zspeed_ewma_extreme_min,
            "beta_stability_degraded_min": thresholds.beta_stability_degraded_min,
        },
        "method_notes": [
            "No thresholds or filters are retuned in this campaign.",
            "Each split is run as an independent backtest with its own start/end dates.",
            "H1/H2/H3 logic is imported from scripts/run_sweden_filter_ablation_campaign.py.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = build_campaign_summary(
        scorecard=scorecard,
        trade_level=combined["trade_level"],
        portfolio_level=combined["portfolio_level"],
        concentration=combined["concentration"],
        thresholds=thresholds,
        splits=splits,
    )
    conclusion = build_conclusion(
        scorecard,
        combined["portfolio_level"],
        combined["trade_level"],
        combined["concentration"],
    )
    paths["campaign_summary"].write_text(summary, encoding="utf-8")
    paths["conclusion"].write_text(conclusion, encoding="utf-8")
    return paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_root = Path(args.output_root)
    splits = smoke_splits() if args.smoke else SPLITS
    out_dir = build_output_dir(output_root=output_root, suffix=args.output_suffix, smoke=bool(args.smoke))
    thresholds, threshold_meta, ablation_dir = load_exact_thresholds(
        output_root=output_root,
        explicit_ablation=args.ablation_output,
        explicit_reference=args.reference_output,
    )

    LOGGER.info("Output directory: %s", out_dir)
    LOGGER.info("Ablation threshold source: %s", ablation_dir or thresholds.source_dir)
    LOGGER.info(
        "Thresholds: abs_z>=%.6f zspeed_ewma>=%.6f beta_std>=%.6f",
        thresholds.abs_z_extreme_min,
        thresholds.zspeed_ewma_extreme_min,
        thresholds.beta_stability_degraded_min,
    )

    # The weekly Sweden cache starts on the first Friday of 2018. Backtest splits still
    # start on 2018-01-01; this only avoids rebuilding an equivalent scan cache.
    scans = load_or_build_scans(start=SCAN_CACHE_LOAD_START, end=DEFAULT_END, rebuild=bool(args.rebuild_scans))
    if scans.empty:
        raise RuntimeError("No Sweden scans available.")

    assets = build_universe_assets(scans)
    LOGGER.info("Loading Sweden price panel for %d assets.", len(assets))
    price_panel = load_price_panel(assets, DATA_PATH, start=DEFAULT_START, end=DEFAULT_END, buffer_days=520)
    if price_panel.empty:
        raise RuntimeError("No price panel available for Sweden.")
    market_features = compute_market_regime_features(price_panel)
    asset_metadata = pd.read_csv(ASSET_REGISTRY_PATH) if ASSET_REGISTRY_PATH.exists() else pd.DataFrame()

    frames_by_name: dict[str, list[pd.DataFrame]] = {
        "trades_enriched": [],
        "trade_level": [],
        "portfolio_level": [],
        "concentration": [],
        "exit_behavior": [],
        "regime_breakdown": [],
        "segment_breakdown": [],
        "pair_level": [],
        "monthly_returns": [],
        "slot_utilization": [],
        "filter_diagnostics": [],
        "variant_vs_best": [],
    }
    configs = selected_configs()
    for split in splits:
        split_frames = run_split(
            split=split,
            configs=configs,
            scans=scans,
            thresholds=thresholds,
            price_panel=price_panel,
            market_features=market_features,
            asset_metadata=asset_metadata,
        )
        for name, frame in split_frames.items():
            frames_by_name[name].append(frame)

    combined_trade = pd.concat(frames_by_name["trade_level"], ignore_index=True, sort=False)
    combined_port = pd.concat(frames_by_name["portfolio_level"], ignore_index=True, sort=False)
    combined_conc = pd.concat(frames_by_name["concentration"], ignore_index=True, sort=False)
    scorecard = build_robustness_scorecard(
        trade_level=combined_trade,
        portfolio_level=combined_port,
        concentration=combined_conc,
    )
    paths = write_outputs(
        out_dir=out_dir,
        frames_by_name=frames_by_name,
        scorecard=scorecard,
        thresholds=thresholds,
        threshold_metadata=threshold_meta,
        ablation_dir=ablation_dir,
        splits=splits,
    )

    LOGGER.info("Robustness campaign completed.")
    for name, path in paths.items():
        LOGGER.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
