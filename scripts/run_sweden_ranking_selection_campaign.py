from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.engine import _evaluate_entry_candidate
from backtesting.global_loop import (
    _get_or_build_global_context,
    _get_or_build_pair_states_for_window,
    _select_ranked_pairs_for_scan_day,
    run_global_ranking_daily_portfolio,
)
from object.class_file import BatchConfig, StrategyParams
from utils.campaign_journal import upsert_campaign_entry
from utils.inline_scanner import InlineScannerConfig, build_scans_inline


UNIVERSE = "sweden"
FULL_START = "2018-01-01"
FULL_END = "2025-12-31"
IS_START = "2018-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2025-12-31"
SCAN_FREQUENCY = "weekly"
SCAN_WEEKDAY = "FRI"

BASE_Z_ENTRY = 1.5
BASE_Z_EXIT = 0.5
BASE_Z_STOP = 3.0
BASE_Z_WINDOW = 40
BASE_MAX_HOLD = 20
BASE_TOP_N = 20
BASE_MAX_POSITIONS = 5
BASE_FEES = 0.0002

BASELINE_CONFIG_ID = "ranking_baseline"
PENALTY_GAP = 0.35
PENALTY_TURNOVER = 0.10
ROBUST_GAP_MAX = 0.50
BEAT_BASELINE_MIN_DELTA = 0.05

OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "sweden_ranking_selection_campaign_2018_2025"
SCAN_DIR = OUT_DIR / "scan_cache"
BEST_DIR = OUT_DIR / "best_artifacts"
SCAN_CACHE_PATH = SCAN_DIR / "sweden_weekly_fri_scans.parquet"
NOTEBOOK_CELL_PATH = OUT_DIR / "best_notebook_last_cell.py"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"

REQUIRED_SCAN_COLUMNS = {
    "scan_date",
    "asset_1",
    "asset_2",
    "eligibility",
    "eligibility_score",
    "12m_corr",
    "6m_half_life",
    "6m_spread_std",
    "6m_abs_last_z",
    "6m_mean_abs_delta_z",
}


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    start: str
    end: str


SEGMENTS = (
    SegmentSpec("FULL", FULL_START, FULL_END),
    SegmentSpec("IS", IS_START, IS_END),
    SegmentSpec("OOS", OOS_START, OOS_END),
)


def normalize_scans(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    out = df.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"]).dt.normalize()
    out["asset_1"] = out["asset_1"].astype(str).str.upper()
    out["asset_2"] = out["asset_2"].astype(str).str.upper()
    out["eligibility"] = out.get("eligibility", "").astype(str).str.upper()
    out["universe"] = universe

    numeric_cols = [
        "eligibility_score",
        "12m_corr",
        "6m_half_life",
        "6m_spread_std",
        "6m_abs_last_z",
        "6m_mean_abs_delta_z",
        "6m_mean_abs_delta_spread",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(
        ["scan_date", "asset_1", "asset_2", "eligibility_score"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(subset=["scan_date", "asset_1", "asset_2"], keep="first")
    return out.reset_index(drop=True)


def has_required_scan_columns(df: pd.DataFrame) -> bool:
    return REQUIRED_SCAN_COLUMNS.issubset(set(df.columns))


def detect_sector_data_available() -> bool:
    if not ASSET_REGISTRY_PATH.exists():
        return False

    reg = pd.read_csv(ASSET_REGISTRY_PATH)
    sw = reg[reg["category_id"].astype(str).str.lower() == UNIVERSE].copy()
    if sw.empty:
        return False

    sector_like_cols = [
        c for c in sw.columns
        if any(token in str(c).lower() for token in ("sector", "industry", "gics", "icb", "cluster"))
    ]
    for col in sector_like_cols:
        n_unique = int(sw[col].dropna().astype(str).nunique())
        if n_unique >= 2:
            return True
    return False


def build_or_load_weekly_scans(rebuild: bool = False) -> pd.DataFrame:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    if SCAN_CACHE_PATH.exists() and not rebuild:
        cached = normalize_scans(pd.read_parquet(SCAN_CACHE_PATH), UNIVERSE)
        if has_required_scan_columns(cached):
            return cached
        print("[SCAN] Cache exists but misses new ranking columns; rebuilding.")

    inline_cfg = InlineScannerConfig(
        raw_data_path=DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
    )
    scans = build_scans_inline(
        universes=[UNIVERSE],
        start_date=FULL_START,
        end_date=FULL_END,
        freq=SCAN_FREQUENCY,
        scan_weekday=SCAN_WEEKDAY,
        cfg=inline_cfg,
        print_every=20,
    )
    scans = normalize_scans(scans, UNIVERSE)
    scans.to_parquet(SCAN_CACHE_PATH, index=False)
    return scans


def segment_scans(scans: pd.DataFrame, segment: SegmentSpec) -> pd.DataFrame:
    out = scans.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"]).dt.normalize()
    start = pd.Timestamp(segment.start).normalize()
    end = pd.Timestamp(segment.end).normalize()
    buffer_start = (start - BDay(20)).normalize()
    mask = (out["scan_date"] >= buffer_start) & (out["scan_date"] <= end)
    return out.loc[mask].reset_index(drop=True)


def analysis_scans_for_segment(scans: pd.DataFrame, segment: SegmentSpec) -> pd.DataFrame:
    out = scans.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"]).dt.normalize()
    start = pd.Timestamp(segment.start).normalize()
    end = pd.Timestamp(segment.end).normalize()
    mask = (out["scan_date"] >= start) & (out["scan_date"] <= end)
    return out.loc[mask].reset_index(drop=True)


def base_strategy_kwargs() -> dict[str, Any]:
    return {
        "z_entry": BASE_Z_ENTRY,
        "z_exit": BASE_Z_EXIT,
        "z_stop": BASE_Z_STOP,
        "z_window": BASE_Z_WINDOW,
        "beta_mode": "static",
        "fees": BASE_FEES,
        "top_n_candidates": BASE_TOP_N,
        "max_positions": BASE_MAX_POSITIONS,
        "max_holding_days": BASE_MAX_HOLD,
        "exec_lag_days": 1,
        "scan_frequency": SCAN_FREQUENCY,
        "scan_weekday": SCAN_WEEKDAY,
        "signal_space": "raw",
        "selection_mode": "legacy",
        "selection_score_variant": "baseline",
        "eligibility_labels": ("ELIGIBLE",),
        "entry_mode": "baseline_entry",
    }


def build_campaign_configs(sector_data_available: bool) -> list[dict[str, Any]]:
    sector_note = (
        "Sector diversification was skipped because the repo does not expose a clean sector mapping for Sweden; "
        "replaced with a low-correlation penalty variant."
        if not sector_data_available
        else "Sector data was available, but this campaign keeps the low-correlation replacement for simplicity."
    )
    return [
        {
            "config_id": BASELINE_CONFIG_ID,
            "ranking_mode": "ranking_baseline",
            "selection_score_variant": "baseline",
            "variant_label": "legacy_eligibility_score_only",
            "notes": "Current Sweden weekly baseline: legacy sort on eligibility_score only.",
        },
        {
            "config_id": "ranking_half_life_weighted_p0p15",
            "ranking_mode": "ranking_half_life_weighted",
            "selection_score_variant": "half_life_weighted",
            "selection_half_life_penalty": 0.15,
            "variant_label": "eligibility_score_minus_0.15_half_life_rank",
            "notes": "Light penalty on longer 6m half-life.",
        },
        {
            "config_id": "ranking_half_life_weighted_p0p30",
            "ranking_mode": "ranking_half_life_weighted",
            "selection_score_variant": "half_life_weighted",
            "selection_half_life_penalty": 0.30,
            "variant_label": "eligibility_score_minus_0.30_half_life_rank",
            "notes": "Stronger penalty on longer 6m half-life.",
        },
        {
            "config_id": "ranking_spread_speed_penalized_p0p10",
            "ranking_mode": "ranking_spread_speed_penalized",
            "selection_score_variant": "spread_speed_penalized",
            "selection_speed_penalty": 0.10,
            "variant_label": "eligibility_score_minus_0.10_speed_rank",
            "notes": "Light penalty on noisy 6m mean abs delta-z at scan.",
        },
        {
            "config_id": "ranking_spread_speed_penalized_p0p20",
            "ranking_mode": "ranking_spread_speed_penalized",
            "selection_score_variant": "spread_speed_penalized",
            "selection_speed_penalty": 0.20,
            "variant_label": "eligibility_score_minus_0.20_speed_rank",
            "notes": "Stronger penalty on noisy 6m mean abs delta-z at scan.",
        },
        {
            "config_id": "ranking_distance_to_mean_over_half_life_w0p15",
            "ranking_mode": "ranking_distance_to_mean_over_half_life",
            "selection_score_variant": "distance_to_mean_over_half_life",
            "selection_distance_weight": 0.15,
            "variant_label": "eligibility_score_plus_0.15_abs_z_over_sqrt_half_life_rank",
            "notes": "Boost scan-time distance to mean scaled by sqrt(half-life).",
        },
        {
            "config_id": "ranking_distance_to_mean_over_half_life_w0p30",
            "ranking_mode": "ranking_distance_to_mean_over_half_life",
            "selection_score_variant": "distance_to_mean_over_half_life",
            "selection_distance_weight": 0.30,
            "variant_label": "eligibility_score_plus_0.30_abs_z_over_sqrt_half_life_rank",
            "notes": "Stronger boost to distance-over-half-life signal.",
        },
        {
            "config_id": "ranking_low_corr_penalized_p0p10",
            "ranking_mode": "ranking_low_corr_penalized",
            "selection_score_variant": "low_corr_penalized",
            "selection_corr_penalty": 0.10,
            "variant_label": "eligibility_score_minus_0.10_low_corr_rank",
            "notes": sector_note,
        },
        {
            "config_id": "ranking_low_corr_penalized_p0p20",
            "ranking_mode": "ranking_low_corr_penalized",
            "selection_score_variant": "low_corr_penalized",
            "selection_corr_penalty": 0.20,
            "variant_label": "eligibility_score_minus_0.20_low_corr_rank",
            "notes": sector_note,
        },
    ]


def build_strategy_params(config_row: dict[str, Any]) -> StrategyParams:
    kwargs = base_strategy_kwargs()
    for key in (
        "selection_score_variant",
        "selection_half_life_penalty",
        "selection_speed_penalty",
        "selection_distance_weight",
        "selection_corr_penalty",
    ):
        if key in config_row:
            kwargs[key] = config_row[key]
    return StrategyParams(**kwargs)


def metric_value(stats: dict[str, Any], key: str) -> float:
    val = stats.get(key, np.nan)
    try:
        return float(val)
    except Exception:
        return np.nan


def closed_trades_frame(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    out["exit_datetime"] = pd.to_datetime(out["exit_datetime"], errors="coerce")
    return out[out["exit_datetime"].notna()].copy()


def summarize_segment_result(segment: SegmentSpec, res: dict[str, Any]) -> dict[str, Any]:
    stats = res["stats"]
    trades = res["trades"].copy()
    closed = closed_trades_frame(trades)
    equity = res["equity"].copy()
    entry_filter_summary = res.get("entry_filter_summary", pd.DataFrame()).copy()

    trade_metric_col = "trade_return_isolated"
    if closed.empty or not closed.get(trade_metric_col, pd.Series(dtype=float)).notna().any():
        trade_metric_col = "trade_return"

    filter_map = (
        {str(r.metric): int(r.count) for r in entry_filter_summary.itertuples(index=False)}
        if not entry_filter_summary.empty
        else {}
    )
    threshold_hits = int(filter_map.get("entry_threshold_hits", 0))
    accepted = int(filter_map.get("entry_accepted", 0))

    hit_ratio = (
        float((pd.to_numeric(closed[trade_metric_col], errors="coerce") > 0).mean())
        if len(closed) > 0
        else np.nan
    )
    trade_metric_sum = (
        float(pd.to_numeric(closed[trade_metric_col], errors="coerce").sum())
        if len(closed) > 0
        else np.nan
    )

    return {
        "segment": segment.name,
        "start_date": segment.start,
        "end_date": segment.end,
        "final_equity": metric_value(stats, "Final Equity"),
        "total_return": (
            metric_value(stats, "Final Equity") - 1.0
            if np.isfinite(metric_value(stats, "Final Equity"))
            else np.nan
        ),
        "sharpe": metric_value(stats, "Sharpe"),
        "cagr": metric_value(stats, "CAGR"),
        "max_drawdown": metric_value(stats, "Max Drawdown"),
        "nb_trades": int(stats.get("Nb Trades", 0)),
        "closed_trades": int(len(closed)),
        "hit_ratio": hit_ratio,
        "avg_trade_return": (
            float(pd.to_numeric(closed[trade_metric_col], errors="coerce").mean())
            if len(closed) > 0
            else np.nan
        ),
        "trade_metric_sum": trade_metric_sum,
        "avg_open_positions": float(equity["n_open_positions"].mean()) if not equity.empty else np.nan,
        "pct_days_with_positions": float((equity["n_open_positions"] > 0).mean()) if not equity.empty else np.nan,
        "lookahead_violations": int(stats.get("Lookahead violations", 0)),
        "avg_scan_age_bdays": metric_value(stats, "Avg scan age (bdays)"),
        "anomaly_flag": bool(stats.get("Anomaly flag", False)),
        "anomaly_reasons": str(stats.get("Anomaly reasons", "")),
        "entry_threshold_hits": threshold_hits,
        "entry_accepted": accepted,
        "entry_accept_rate": (accepted / threshold_hits) if threshold_hits > 0 else np.nan,
    }


def flatten_segment_metrics(segment_metrics: dict[str, Any]) -> dict[str, Any]:
    prefix = segment_metrics["segment"].lower()
    out: dict[str, Any] = {}
    for key, value in segment_metrics.items():
        if key == "segment":
            continue
        out[f"{prefix}_{key}"] = value
    return out


def run_segment(
    config_row: dict[str, Any],
    segment: SegmentSpec,
    scans: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = build_strategy_params(config_row)
    cfg = BatchConfig(
        data_path=DATA_PATH,
        start_date=segment.start,
        end_date=segment.end,
    )
    segment_scan_df = segment_scans(scans, segment)
    res = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=[UNIVERSE],
        scans=segment_scan_df,
    )
    if not res:
        raise RuntimeError(f"No backtest result for {config_row['config_id']} on {segment.name}.")
    return summarize_segment_result(segment, res), res


def serialize_pair_ids(pair_ids: list[str]) -> str:
    return json.dumps(pair_ids)


def deserialize_pair_ids(text: str) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    try:
        raw = json.loads(text)
    except Exception:
        return set()
    return {str(x) for x in raw}


def collect_selection_diagnostics(
    config_row: dict[str, Any],
    segment: SegmentSpec,
    scans: pd.DataFrame,
) -> pd.DataFrame:
    params = build_strategy_params(config_row)
    cfg = BatchConfig(data_path=DATA_PATH, start_date=segment.start, end_date=segment.end)
    segment_scan_df = segment_scans(scans, segment)
    ctx = _get_or_build_global_context(cfg=cfg, params=params, universes=[UNIVERSE], scans=segment_scan_df)
    if ctx is None:
        raise RuntimeError(f"No global context for selection diagnostics: {config_row['config_id']} / {segment.name}.")

    pair_state_cache = _get_or_build_pair_states_for_window(ctx, int(params.z_window))
    analysis_scans = analysis_scans_for_segment(segment_scan_df, segment)
    if analysis_scans.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for scan_dt, df_day in analysis_scans.groupby("scan_date", sort=True):
        selected = _select_ranked_pairs_for_scan_day(
            df_day=df_day,
            params=params,
            top_n=int(params.top_n_candidates),
        )
        next_trade_dt = (pd.Timestamp(scan_dt).normalize() + BDay(int(params.exec_lag_days))).normalize()

        selected_pair_ids: list[str] = []
        state_ready = 0
        threshold_hits = 0
        accepted = 0

        for a1, a2 in selected:
            pid = f"{a1.upper()}_{a2.upper()}"
            selected_pair_ids.append(pid)

            dfp = pair_state_cache.get(pid)
            if dfp is None or next_trade_dt not in dfp.index:
                continue
            if not bool(dfp.at[next_trade_dt, "state_available"]):
                continue

            z_val = pd.to_numeric(dfp.at[next_trade_dt, "z"], errors="coerce")
            if pd.isna(z_val):
                continue

            state_ready += 1
            entry_eval = _evaluate_entry_candidate(dfp=dfp, dt=next_trade_dt, params=params)
            if str(entry_eval.get("reason", "")).strip() != "entry_below_threshold":
                threshold_hits += 1
            if bool(entry_eval.get("accepted", False)):
                accepted += 1

        rows.append(
            {
                "config_id": config_row["config_id"],
                "ranking_mode": config_row["ranking_mode"],
                "variant_label": config_row["variant_label"],
                "segment": segment.name,
                "scan_date": pd.Timestamp(scan_dt).normalize(),
                "next_trade_date": next_trade_dt,
                "selected_candidates": int(len(selected_pair_ids)),
                "state_ready_candidates": int(state_ready),
                "threshold_hit_candidates": int(threshold_hits),
                "accepted_candidates": int(accepted),
                "selected_pair_ids_json": serialize_pair_ids(selected_pair_ids),
                "scan_to_next_session_ok": bool(next_trade_dt > pd.Timestamp(scan_dt).normalize()),
            }
        )

    return pd.DataFrame(rows)


def attach_baseline_overlap(selection_by_scan: pd.DataFrame) -> pd.DataFrame:
    if selection_by_scan.empty:
        return selection_by_scan

    base = selection_by_scan[selection_by_scan["config_id"] == BASELINE_CONFIG_ID][
        ["segment", "scan_date", "selected_pair_ids_json"]
    ].rename(columns={"selected_pair_ids_json": "baseline_selected_pair_ids_json"})

    out = selection_by_scan.merge(base, on=["segment", "scan_date"], how="left")
    overlap_vals = []
    for row in out.itertuples(index=False):
        current = deserialize_pair_ids(row.selected_pair_ids_json)
        baseline = deserialize_pair_ids(row.baseline_selected_pair_ids_json)
        if not baseline:
            overlap_vals.append(np.nan)
            continue
        overlap_vals.append(100.0 * len(current & baseline) / len(baseline))
    out["candidate_overlap_pct_vs_baseline"] = overlap_vals
    return out


def summarize_selection_diagnostics(selection_by_scan: pd.DataFrame, sector_data_available: bool) -> pd.DataFrame:
    if selection_by_scan.empty:
        return pd.DataFrame()

    summary = (
        selection_by_scan.groupby(["config_id", "ranking_mode", "variant_label", "segment"], as_index=False)
        .agg(
            n_scans=("scan_date", "size"),
            avg_selected_candidates=("selected_candidates", "mean"),
            avg_state_ready_candidates=("state_ready_candidates", "mean"),
            avg_threshold_hit_candidates=("threshold_hit_candidates", "mean"),
            avg_accepted_candidates=("accepted_candidates", "mean"),
            candidate_overlap_pct_vs_baseline=("candidate_overlap_pct_vs_baseline", "mean"),
        )
    )
    summary["avg_state_ready_rate_pct"] = (
        100.0 * summary["avg_state_ready_candidates"] / summary["avg_selected_candidates"].replace(0, np.nan)
    )
    summary["avg_threshold_hit_rate_pct"] = (
        100.0 * summary["avg_threshold_hit_candidates"] / summary["avg_selected_candidates"].replace(0, np.nan)
    )
    summary["sector_data_available"] = bool(sector_data_available)
    summary["sector_concentration"] = np.nan
    return summary


def merge_selection_summary(run_level: pd.DataFrame, selection_summary: pd.DataFrame) -> pd.DataFrame:
    if selection_summary.empty:
        return run_level.copy()

    value_cols = [
        "n_scans",
        "avg_selected_candidates",
        "avg_state_ready_candidates",
        "avg_threshold_hit_candidates",
        "avg_accepted_candidates",
        "candidate_overlap_pct_vs_baseline",
        "avg_state_ready_rate_pct",
        "avg_threshold_hit_rate_pct",
        "sector_concentration",
    ]
    wide_rows: list[dict[str, Any]] = []
    for config_id, g in selection_summary.groupby("config_id", sort=False):
        row: dict[str, Any] = {"config_id": config_id}
        for record in g.itertuples(index=False):
            prefix = str(record.segment).lower()
            for col in value_cols:
                row[f"{prefix}_{col}"] = getattr(record, col)
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    return run_level.merge(wide, on="config_id", how="left")


def assert_no_lookahead(run_level: pd.DataFrame) -> None:
    cols = [c for c in run_level.columns if c.endswith("_lookahead_violations")]
    if not cols:
        return
    vals = run_level[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    total = int(vals.to_numpy(dtype=float).sum())
    if total != 0:
        raise RuntimeError(f"Look-ahead check failed: {total} violation(s) found across campaign outputs.")


def compute_final_scores(run_level: pd.DataFrame) -> pd.DataFrame:
    df = run_level.copy()
    baseline = df[df["config_id"] == BASELINE_CONFIG_ID]
    if baseline.empty:
        raise RuntimeError(f"Missing {BASELINE_CONFIG_ID} in run-level results.")

    baseline_row = baseline.iloc[0]
    baseline_oos_sharpe = float(baseline_row["oos_sharpe"])
    baseline_oos_trades = max(1, int(baseline_row["oos_nb_trades"]))

    oos_metric = pd.to_numeric(df["oos_sharpe"], errors="coerce")
    gap = (pd.to_numeric(df["is_sharpe"], errors="coerce") - oos_metric).abs()
    trade_count_gap_pct = (
        100.0
        * (pd.to_numeric(df["oos_nb_trades"], errors="coerce").fillna(0.0) - baseline_oos_trades)
        / baseline_oos_trades
    )
    turnover_penalty = trade_count_gap_pct.abs() / 100.0

    df["is_oos_gap"] = gap
    df["delta_oos_sharpe_vs_baseline"] = oos_metric - baseline_oos_sharpe
    df["trade_count_gap_pct"] = trade_count_gap_pct
    df["turnover_penalty"] = turnover_penalty
    df["final_score"] = (
        oos_metric.fillna(-5.0)
        - PENALTY_GAP * gap.fillna(5.0)
        - PENALTY_TURNOVER * turnover_penalty.fillna(1.0)
    )
    df["beats_baseline_oos"] = df["delta_oos_sharpe_vs_baseline"] > BEAT_BASELINE_MIN_DELTA
    df["robust_oos"] = (
        (df["is_oos_gap"] <= ROBUST_GAP_MAX)
        & (pd.to_numeric(df["oos_lookahead_violations"], errors="coerce").fillna(1) == 0)
        & (~df["oos_anomaly_flag"].astype(bool))
    )
    return df


def summarize_by_ranking_mode(run_level: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_level.groupby("ranking_mode", as_index=False)
        .agg(
            n_configs=("config_id", "size"),
            avg_oos_sharpe=("oos_sharpe", "mean"),
            best_oos_sharpe=("oos_sharpe", "max"),
            avg_final_score=("final_score", "mean"),
            best_final_score=("final_score", "max"),
            avg_oos_nb_trades=("oos_nb_trades", "mean"),
            avg_is_oos_gap=("is_oos_gap", "mean"),
            avg_oos_candidate_overlap_pct=("oos_candidate_overlap_pct_vs_baseline", "mean"),
            avg_oos_tradable_candidates=("oos_avg_threshold_hit_candidates", "mean"),
        )
        .sort_values(["best_final_score", "best_oos_sharpe"], ascending=[False, False])
    )
    return grouped.reset_index(drop=True)


def select_retained_row(ranking: pd.DataFrame) -> pd.Series:
    baseline_row = ranking[ranking["config_id"] == BASELINE_CONFIG_ID].iloc[0]
    best_row = ranking.iloc[0]
    if (
        str(best_row["config_id"]) != BASELINE_CONFIG_ID
        and bool(best_row["beats_baseline_oos"])
        and bool(best_row["robust_oos"])
        and float(best_row["final_score"]) >= float(baseline_row["final_score"])
    ):
        return best_row
    return baseline_row


def compare_best_vs_baseline(best_row: pd.Series, baseline_row: pd.Series) -> str:
    delta_sharpe = float(best_row["delta_oos_sharpe_vs_baseline"])
    trade_gap_pct = float(best_row["trade_count_gap_pct"])
    delta_tradable = (
        float(best_row["oos_avg_threshold_hit_candidates"] - baseline_row["oos_avg_threshold_hit_candidates"])
        if np.isfinite(best_row.get("oos_avg_threshold_hit_candidates", np.nan))
        and np.isfinite(baseline_row.get("oos_avg_threshold_hit_candidates", np.nan))
        else np.nan
    )
    overlap_pct = float(best_row.get("oos_candidate_overlap_pct_vs_baseline", np.nan))

    if delta_sharpe <= BEAT_BASELINE_MIN_DELTA:
        return "No genuine OOS improvement versus the ranking baseline."
    if np.isfinite(delta_tradable) and delta_tradable > 0.05 and abs(trade_gap_pct) <= 20.0:
        return (
            "The gain is consistent with slightly better economic prioritization of tradable pairs, "
            "not just with trading materially fewer names."
        )
    if trade_gap_pct < -20.0 and (not np.isfinite(delta_tradable) or delta_tradable <= 0.05):
        return "The gain looks driven mostly by taking fewer trades, with limited evidence of better pair prioritization."
    if np.isfinite(overlap_pct) and overlap_pct < 85.0:
        return "The gain seems to come from a meaningfully different candidate ordering, but the economic source stays mixed."
    return "The improvement is mixed and only partially attributable to better pair selection."


def build_conclusion_text(ranking: pd.DataFrame, retained_row: pd.Series, sector_data_available: bool) -> str:
    best_row = ranking.iloc[0]
    baseline_row = ranking[ranking["config_id"] == BASELINE_CONFIG_ID].iloc[0]
    comparison_text = compare_best_vs_baseline(best_row, baseline_row)

    if str(retained_row["config_id"]) == BASELINE_CONFIG_ID:
        decision_text = "Keep ranking_baseline for the next step and reject the challengers."
    else:
        decision_text = f"Promote {retained_row['config_id']} for the next step."

    rejected = ranking.loc[ranking["config_id"] != retained_row["config_id"], "config_id"].tolist()
    rejected_text = ", ".join(rejected[:8]) + ("..." if len(rejected) > 8 else "")
    sector_line = (
        "Sector diversification was not tested because the repo does not expose a clean sector mapping for Sweden; "
        "it was replaced by ranking_low_corr_penalized."
        if not sector_data_available
        else "Sector data was available."
    )

    lines = [
        "Sweden ranking-selection campaign conclusion",
        f"- Best configuration by final_score: {best_row['config_id']} ({best_row['variant_label']}).",
        f"- Retained configuration for follow-up: {retained_row['config_id']} ({retained_row['variant_label']}).",
        f"- OOS Sharpe: retained {retained_row['oos_sharpe']:.2f} vs baseline {baseline_row['oos_sharpe']:.2f}.",
        f"- OOS trades: retained {int(retained_row['oos_nb_trades'])} vs baseline {int(baseline_row['oos_nb_trades'])}; trade_count_gap_pct={retained_row['trade_count_gap_pct']:.1f}%.",
        f"- OOS tradable candidates per scan: retained {retained_row.get('oos_avg_threshold_hit_candidates', np.nan):.2f} vs baseline {baseline_row.get('oos_avg_threshold_hit_candidates', np.nan):.2f}.",
        f"- OOS candidate overlap vs baseline: {retained_row.get('oos_candidate_overlap_pct_vs_baseline', np.nan):.1f}%.",
        f"- IS/OOS gap: {retained_row['is_oos_gap']:.2f}; robust_oos={bool(retained_row['robust_oos'])}.",
        f"- Look-ahead check: FULL={int(retained_row['full_lookahead_violations'])}, IS={int(retained_row['is_lookahead_violations'])}, OOS={int(retained_row['oos_lookahead_violations'])}.",
        f"- Interpretation: {comparison_text}",
        f"- Decision: {decision_text}",
        f"- Sector note: {sector_line}",
        f"- Rejected variants: {rejected_text}",
    ]
    return "\n".join(lines)


def render_notebook_cell(best_row: pd.Series) -> str:
    extra_params = {
        "selection_score_variant": best_row["selection_score_variant"],
        "selection_half_life_penalty": best_row.get("selection_half_life_penalty"),
        "selection_speed_penalty": best_row.get("selection_speed_penalty"),
        "selection_distance_weight": best_row.get("selection_distance_weight"),
        "selection_corr_penalty": best_row.get("selection_corr_penalty"),
    }
    extra_param_lines = []
    for key, value in extra_params.items():
        if pd.isna(value):
            continue
        if isinstance(value, (float, np.floating)) and abs(float(value)) <= 1e-12 and key != "selection_score_variant":
            continue
        extra_param_lines.append(f"    {key}={repr(value)},")

    extra_block = "\n".join(extra_param_lines)
    if extra_block:
        extra_block = extra_block + "\n"

    return f"""from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inline_scanner import InlineScannerConfig, build_scans_inline
from backtesting.global_loop import run_global_ranking_daily_portfolio
from object.class_file import BatchConfig, StrategyParams


def normalize_scans(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    out = df.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"]).dt.normalize()
    out["asset_1"] = out["asset_1"].astype(str).str.upper()
    out["asset_2"] = out["asset_2"].astype(str).str.upper()
    out["eligibility"] = out.get("eligibility", "").astype(str).str.upper()
    out["universe"] = universe
    for col in [
        "eligibility_score",
        "12m_corr",
        "6m_half_life",
        "6m_spread_std",
        "6m_abs_last_z",
        "6m_mean_abs_delta_z",
        "6m_mean_abs_delta_spread",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values(
        ["scan_date", "asset_1", "asset_2", "eligibility_score"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(subset=["scan_date", "asset_1", "asset_2"], keep="first")
    return out.reset_index(drop=True)


UNIVERSE = "{UNIVERSE}"
FULL_START = "{FULL_START}"
FULL_END = "{FULL_END}"
IS_START = "{IS_START}"
IS_END = "{IS_END}"
OOS_START = "{OOS_START}"
OOS_END = "{OOS_END}"
SCAN_FREQUENCY = "{SCAN_FREQUENCY}"
SCAN_WEEKDAY = "{SCAN_WEEKDAY}"
RANKING_MODE = "{best_row['ranking_mode']}"
CONFIG_ID = "{best_row['config_id']}"

inline_cfg = InlineScannerConfig(
    raw_data_path=PROJECT_ROOT / "data" / "raw" / "d1",
    asset_registry_path=PROJECT_ROOT / "data" / "asset_registry.csv",
    lookback_days=504,
    min_obs=100,
    liquidity_lookback=20,
    liquidity_min_moves=0.0,
)

scans = build_scans_inline(
    universes=[UNIVERSE],
    start_date=FULL_START,
    end_date=FULL_END,
    freq=SCAN_FREQUENCY,
    scan_weekday=SCAN_WEEKDAY,
    cfg=inline_cfg,
    print_every=20,
)
scans = normalize_scans(scans, UNIVERSE)

params = StrategyParams(
    z_entry={BASE_Z_ENTRY},
    z_exit={BASE_Z_EXIT},
    z_stop={BASE_Z_STOP},
    z_window={BASE_Z_WINDOW},
    beta_mode="static",
    fees={BASE_FEES},
    top_n_candidates={BASE_TOP_N},
    max_positions={BASE_MAX_POSITIONS},
    max_holding_days={BASE_MAX_HOLD},
    exec_lag_days=1,
    scan_frequency="{SCAN_FREQUENCY}",
    scan_weekday="{SCAN_WEEKDAY}",
    signal_space="raw",
    selection_mode="legacy",
    eligibility_labels=("ELIGIBLE",),
    entry_mode="baseline_entry",
{extra_block})

cfg_full = BatchConfig(data_path=PROJECT_ROOT / "data" / "raw" / "d1", start_date=FULL_START, end_date=FULL_END)
cfg_is = BatchConfig(data_path=PROJECT_ROOT / "data" / "raw" / "d1", start_date=IS_START, end_date=IS_END)
cfg_oos = BatchConfig(data_path=PROJECT_ROOT / "data" / "raw" / "d1", start_date=OOS_START, end_date=OOS_END)

scans["scan_date"] = pd.to_datetime(scans["scan_date"]).dt.normalize()
is_buffer_start = (pd.Timestamp(IS_START) - BDay(20)).normalize()
oos_buffer_start = (pd.Timestamp(OOS_START) - BDay(20)).normalize()
scans_is = scans[(scans["scan_date"] >= is_buffer_start) & (scans["scan_date"] <= pd.Timestamp(IS_END))].copy()
scans_oos = scans[(scans["scan_date"] >= oos_buffer_start) & (scans["scan_date"] <= pd.Timestamp(OOS_END))].copy()

res_full = run_global_ranking_daily_portfolio(cfg=cfg_full, params=params, universes=[UNIVERSE], scans=scans)
res_is = run_global_ranking_daily_portfolio(cfg=cfg_is, params=params, universes=[UNIVERSE], scans=scans_is)
res_oos = run_global_ranking_daily_portfolio(cfg=cfg_oos, params=params, universes=[UNIVERSE], scans=scans_oos)

print("Universe:", UNIVERSE)
print("Scan policy:", SCAN_FREQUENCY, SCAN_WEEKDAY, "| exec_lag_days=1")
print("Eligibility:", ("ELIGIBLE",), "| entry:", params.entry_mode, "| ranking:", RANKING_MODE, "| config_id:", CONFIG_ID)

for label, res in [("FULL", res_full), ("IS", res_is), ("OOS", res_oos)]:
    print(f"\\n{{label}} stats")
    for key, value in res["stats"].items():
        print(f"  {{key}}: {{value}}")
    closed_trades = int(pd.to_datetime(res["trades"]["exit_datetime"], errors="coerce").notna().sum()) if not res["trades"].empty else 0
    print("  Closed trades:", closed_trades)

out_dir = PROJECT_ROOT / "data" / "experiments" / "sweden_ranking_selection_campaign_2018_2025" / "notebook_reproduction"
out_dir.mkdir(parents=True, exist_ok=True)
res_full["trades"].to_csv(out_dir / f"{{CONFIG_ID}}_full_trades.csv", index=False)
res_full["scan_usage"].to_csv(out_dir / f"{{CONFIG_ID}}_full_scan_usage.csv", index=False)
print("\\nSaved notebook reproduction artifacts to", out_dir)
"""


def save_best_artifacts(retained_row: pd.Series, retained_results: dict[str, dict[str, Any]]) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for segment_name, res in retained_results.items():
        seg = segment_name.lower()
        res["equity"].to_csv(BEST_DIR / f"{seg}_equity.csv", index=False)
        res["trades"].to_csv(BEST_DIR / f"{seg}_trades.csv", index=False)
        res["diagnostics"].to_csv(BEST_DIR / f"{seg}_diagnostics.csv", index=False)
        res["scan_usage"].to_csv(BEST_DIR / f"{seg}_scan_usage.csv", index=False)
    NOTEBOOK_CELL_PATH.write_text(render_notebook_cell(retained_row), encoding="utf-8")


def update_campaign_journal(retained_row: pd.Series, conclusion_text: str, sector_data_available: bool) -> None:
    decision_line = ""
    for line in conclusion_text.splitlines():
        if line.startswith("- Decision:"):
            decision_line = line.removeprefix("- Decision:").strip()
            break
    sector_line = (
        "sector diversification skipped; replaced by low-correlation penalty"
        if not sector_data_available
        else "sector data available"
    )
    summary_lines = [
        f"Universe: {UNIVERSE} | FULL {FULL_START} -> {FULL_END} | IS {IS_START} -> {IS_END} | OOS {OOS_START} -> {OOS_END}.",
        "Scope: entry baseline_entry fixed, eligibility ELIGIBLE fixed, weekly scan Friday, execution from next session only.",
        (
            f"Retained config: {retained_row['config_id']} | ranking_mode={retained_row['ranking_mode']} "
            f"| selection_score_variant={retained_row['selection_score_variant']}."
        ),
        (
            f"OOS Sharpe={retained_row['oos_sharpe']:.2f} | OOS trades={int(retained_row['oos_nb_trades'])} "
            f"| hit ratio={retained_row['oos_hit_ratio']:.2%} | final_score={retained_row['final_score']:.4f}."
        ),
        (
            f"Gap IS/OOS={retained_row['is_oos_gap']:.2f} | trade_count_gap_pct={retained_row['trade_count_gap_pct']:.1f}% "
            f"| candidate_overlap_vs_baseline={retained_row.get('oos_candidate_overlap_pct_vs_baseline', np.nan):.1f}%."
        ),
        (
            f"Selection diagnostics OOS: avg_selected_candidates={retained_row.get('oos_avg_selected_candidates', np.nan):.2f} "
            f"| avg_threshold_hit_candidates={retained_row.get('oos_avg_threshold_hit_candidates', np.nan):.2f}."
        ),
        f"Look-ahead violations: FULL={int(retained_row['full_lookahead_violations'])}, IS={int(retained_row['is_lookahead_violations'])}, OOS={int(retained_row['oos_lookahead_violations'])}.",
        f"Sector note: {sector_line}.",
        f"Decision: {decision_line or 'See conclusion.txt.'}",
    ]
    upsert_campaign_entry(
        campaign_key="sweden_ranking_selection_campaign_2018_2025",
        title="Sweden Ranking Selection Campaign 2018-2025",
        summary_lines=summary_lines,
        out_dir=OUT_DIR,
        notebook_cell_path=NOTEBOOK_CELL_PATH,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sector_data_available = detect_sector_data_available()
    scans = build_or_load_weekly_scans(rebuild=False)

    campaign_configs = build_campaign_configs(sector_data_available=sector_data_available)
    pd.DataFrame(campaign_configs).to_csv(OUT_DIR / "config_manifest.csv", index=False)

    run_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    selection_by_scan_frames: list[pd.DataFrame] = []
    results_by_config: dict[str, dict[str, dict[str, Any]]] = {}

    for config_row in campaign_configs:
        print(f"[RUN] {config_row['config_id']}")
        merged_row = {
            "config_id": config_row["config_id"],
            "ranking_mode": config_row["ranking_mode"],
            "variant_label": config_row["variant_label"],
            "selection_score_variant": config_row["selection_score_variant"],
            "notes": config_row["notes"],
            "selection_half_life_penalty": config_row.get("selection_half_life_penalty", np.nan),
            "selection_speed_penalty": config_row.get("selection_speed_penalty", np.nan),
            "selection_distance_weight": config_row.get("selection_distance_weight", np.nan),
            "selection_corr_penalty": config_row.get("selection_corr_penalty", np.nan),
        }
        config_segment_results: dict[str, dict[str, Any]] = {}

        for segment in SEGMENTS:
            print(f"  [SEGMENT] {segment.name}")
            segment_metrics, res = run_segment(config_row, segment, scans)
            merged_row.update(flatten_segment_metrics(segment_metrics))
            long_rows.append(
                {
                    "config_id": config_row["config_id"],
                    "ranking_mode": config_row["ranking_mode"],
                    "variant_label": config_row["variant_label"],
                    **segment_metrics,
                }
            )
            selection_by_scan_frames.append(collect_selection_diagnostics(config_row, segment, scans))
            config_segment_results[segment.name] = res

        run_rows.append(merged_row)
        results_by_config[config_row["config_id"]] = config_segment_results

    run_level = pd.DataFrame(run_rows)
    long_df = pd.DataFrame(long_rows)

    selection_by_scan = (
        pd.concat([df for df in selection_by_scan_frames if not df.empty], ignore_index=True)
        if selection_by_scan_frames
        else pd.DataFrame()
    )
    selection_by_scan = attach_baseline_overlap(selection_by_scan)
    selection_summary = summarize_selection_diagnostics(selection_by_scan, sector_data_available=sector_data_available)
    run_level = merge_selection_summary(run_level, selection_summary)
    assert_no_lookahead(run_level)
    run_level = compute_final_scores(run_level)

    ranking = run_level.sort_values(["final_score", "oos_sharpe"], ascending=[False, False]).reset_index(drop=True)
    retained_row = select_retained_row(ranking)
    ranking["retained_for_followup"] = ranking["config_id"] == retained_row["config_id"]
    summary_by_mode = summarize_by_ranking_mode(run_level)

    run_level.to_csv(OUT_DIR / "run_level.csv", index=False)
    long_df.to_csv(OUT_DIR / "run_level_segments_long.csv", index=False)
    summary_by_mode.to_csv(OUT_DIR / "summary_by_ranking_mode.csv", index=False)
    ranking.to_csv(OUT_DIR / "ranking_final.csv", index=False)
    selection_summary.to_csv(OUT_DIR / "selection_diagnostics.csv", index=False)
    if not selection_by_scan.empty:
        selection_by_scan.to_csv(OUT_DIR / "selection_diagnostics_by_scan.csv", index=False)

    conclusion_text = build_conclusion_text(ranking, retained_row, sector_data_available=sector_data_available)
    (OUT_DIR / "conclusion.txt").write_text(conclusion_text, encoding="utf-8")

    save_best_artifacts(retained_row, results_by_config[str(retained_row["config_id"])])
    update_campaign_journal(retained_row, conclusion_text, sector_data_available=sector_data_available)

    metadata = {
        "universe": UNIVERSE,
        "full_period": [FULL_START, FULL_END],
        "is_period": [IS_START, IS_END],
        "oos_period": [OOS_START, OOS_END],
        "scan_frequency": SCAN_FREQUENCY,
        "scan_weekday": SCAN_WEEKDAY,
        "sector_data_available": sector_data_available,
        "retained_config_id": str(retained_row["config_id"]),
        "baseline_config_id": BASELINE_CONFIG_ID,
        "penalty_gap": PENALTY_GAP,
        "penalty_turnover": PENALTY_TURNOVER,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== RETAINED CONFIG ===")
    print(
        retained_row[
            [
                "config_id",
                "ranking_mode",
                "variant_label",
                "oos_sharpe",
                "is_sharpe",
                "is_oos_gap",
                "oos_nb_trades",
                "trade_count_gap_pct",
                "oos_candidate_overlap_pct_vs_baseline",
                "oos_avg_threshold_hit_candidates",
                "final_score",
            ]
        ]
    )
    print("\n=== CONCLUSION ===")
    print(conclusion_text)
    print("\nNotebook cell saved to:", NOTEBOOK_CELL_PATH)


if __name__ == "__main__":
    main()
