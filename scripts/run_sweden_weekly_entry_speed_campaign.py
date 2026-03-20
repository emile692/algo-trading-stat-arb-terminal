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

from backtesting.global_loop import run_global_ranking_daily_portfolio
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
BASE_Z_WINDOW = 40
BASE_MAX_HOLD = 20
BASE_TOP_N = 20
BASE_MAX_POSITIONS = 5
BASE_FEES = 0.0002

PENALTY_GAP = 0.35
PENALTY_TRADES = 0.01
ROBUST_GAP_MAX = 0.50
BEAT_BASELINE_MIN_DELTA = 0.05

OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "sweden_weekly_entry_speed_campaign_2018_2025"
SCAN_DIR = OUT_DIR / "scan_cache"
BEST_DIR = OUT_DIR / "best_artifacts"
SCAN_CACHE_PATH = SCAN_DIR / "sweden_weekly_fri_scans.parquet"
NOTEBOOK_CELL_PATH = OUT_DIR / "best_notebook_last_cell.py"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"


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

    if "eligibility_score" in out.columns:
        out["eligibility_score"] = pd.to_numeric(out["eligibility_score"], errors="coerce")
    else:
        out["eligibility_score"] = np.nan

    out = out.sort_values(
        ["scan_date", "asset_1", "asset_2", "eligibility_score"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(subset=["scan_date", "asset_1", "asset_2"], keep="first")
    return out.reset_index(drop=True)


def build_or_load_weekly_scans(rebuild: bool = False) -> pd.DataFrame:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    if SCAN_CACHE_PATH.exists() and not rebuild:
        cached = pd.read_parquet(SCAN_CACHE_PATH)
        return normalize_scans(cached, UNIVERSE)

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


def base_strategy_kwargs() -> dict[str, Any]:
    return {
        "z_entry": BASE_Z_ENTRY,
        "z_exit": BASE_Z_ENTRY / 3.0,
        "z_stop": 2.0 * BASE_Z_ENTRY,
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
        "eligibility_labels": ("ELIGIBLE",),
    }


def build_campaign_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {
            "config_id": "baseline_entry",
            "entry_mode": "baseline_entry",
            "variant_label": "baseline_entry",
            "notes": "Current threshold-only entry.",
        },
        {
            "config_id": "entry_with_spread_speed_filter_cap_0p50",
            "entry_mode": "entry_with_spread_speed_filter",
            "spread_speed_cap": 0.50,
            "variant_label": "spread_speed_filter_cap_0.50",
            "notes": "Benchmark challenger: normalized spread-speed cap.",
        },
        {
            "config_id": "entry_zspeed_hard_cap_0p40",
            "entry_mode": "entry_zspeed_hard_cap",
            "zspeed_cap": 0.40,
            "variant_label": "zspeed_hard_cap_0.40",
            "notes": "Raw |delta_z| hard cap, tighter.",
        },
        {
            "config_id": "entry_zspeed_hard_cap_0p60",
            "entry_mode": "entry_zspeed_hard_cap",
            "zspeed_cap": 0.60,
            "variant_label": "zspeed_hard_cap_0.60",
            "notes": "Raw |delta_z| hard cap, looser.",
        },
        {
            "config_id": "entry_zspeed_ewma_cap_s2_c0p40",
            "entry_mode": "entry_zspeed_ewma_cap",
            "zspeed_ewma_span": 2,
            "zspeed_ewma_cap": 0.40,
            "variant_label": "zspeed_ewma_cap_span2_cap0.40",
            "notes": "EWMA(abs(delta_z)) cap with fast smoothing.",
        },
        {
            "config_id": "entry_zspeed_ewma_cap_s2_c0p60",
            "entry_mode": "entry_zspeed_ewma_cap",
            "zspeed_ewma_span": 2,
            "zspeed_ewma_cap": 0.60,
            "variant_label": "zspeed_ewma_cap_span2_cap0.60",
            "notes": "EWMA(abs(delta_z)) cap with fast smoothing, looser.",
        },
        {
            "config_id": "entry_zspeed_ewma_cap_s4_c0p40",
            "entry_mode": "entry_zspeed_ewma_cap",
            "zspeed_ewma_span": 4,
            "zspeed_ewma_cap": 0.40,
            "variant_label": "zspeed_ewma_cap_span4_cap0.40",
            "notes": "EWMA(abs(delta_z)) cap with slower smoothing.",
        },
        {
            "config_id": "entry_zspeed_ewma_cap_s4_c0p60",
            "entry_mode": "entry_zspeed_ewma_cap",
            "zspeed_ewma_span": 4,
            "zspeed_ewma_cap": 0.60,
            "variant_label": "zspeed_ewma_cap_span4_cap0.60",
            "notes": "EWMA(abs(delta_z)) cap with slower smoothing, looser.",
        },
        {
            "config_id": "entry_zspeed_vol_normalized_w5_c1p00",
            "entry_mode": "entry_zspeed_vol_normalized",
            "zspeed_vol_window": 5,
            "zspeed_vol_cap": 1.00,
            "variant_label": "zspeed_vol_normalized_window5_cap1.00",
            "notes": "Vol-normalized |delta_z| cap, short window.",
        },
        {
            "config_id": "entry_zspeed_vol_normalized_w5_c1p50",
            "entry_mode": "entry_zspeed_vol_normalized",
            "zspeed_vol_window": 5,
            "zspeed_vol_cap": 1.50,
            "variant_label": "zspeed_vol_normalized_window5_cap1.50",
            "notes": "Vol-normalized |delta_z| cap, short window, looser.",
        },
        {
            "config_id": "entry_zspeed_vol_normalized_w10_c1p00",
            "entry_mode": "entry_zspeed_vol_normalized",
            "zspeed_vol_window": 10,
            "zspeed_vol_cap": 1.00,
            "variant_label": "zspeed_vol_normalized_window10_cap1.00",
            "notes": "Vol-normalized |delta_z| cap, smoother reference vol.",
        },
        {
            "config_id": "entry_zspeed_vol_normalized_w10_c1p50",
            "entry_mode": "entry_zspeed_vol_normalized",
            "zspeed_vol_window": 10,
            "zspeed_vol_cap": 1.50,
            "variant_label": "zspeed_vol_normalized_window10_cap1.50",
            "notes": "Vol-normalized |delta_z| cap, smoother reference vol, looser.",
        },
        {
            "config_id": "entry_slowdown_confirmation",
            "entry_mode": "entry_slowdown_confirmation",
            "variant_label": "slowdown_confirmation_abs_dz_t_lt_abs_dz_tm1",
            "notes": "Require a deceleration in |delta_z| before entry.",
        },
    ]
    return configs


def build_strategy_params(config_row: dict[str, Any]) -> StrategyParams:
    kwargs = base_strategy_kwargs()
    for k in (
        "entry_mode",
        "spread_speed_cap",
        "zspeed_cap",
        "zspeed_ewma_span",
        "zspeed_ewma_cap",
        "zspeed_vol_window",
        "zspeed_vol_cap",
    ):
        if k in config_row:
            kwargs[k] = config_row[k]
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

    filter_map = {
        str(r.metric): int(r.count)
        for r in entry_filter_summary.itertuples(index=False)
    } if not entry_filter_summary.empty else {}

    threshold_hits = int(filter_map.get("entry_threshold_hits", 0))
    accepted = int(filter_map.get("entry_accepted", 0))
    blocked_total = int(
        sum(
            count for metric, count in filter_map.items()
            if metric.startswith("entry_blocked_") or metric == "entry_insufficient_history"
        )
    )
    hit_ratio = float((pd.to_numeric(closed[trade_metric_col], errors="coerce") > 0).mean()) if len(closed) > 0 else np.nan
    isolated_sum = (
        float(pd.to_numeric(closed[trade_metric_col], errors="coerce").sum())
        if len(closed) > 0
        else np.nan
    )
    avg_trade_return = (
        float(pd.to_numeric(closed[trade_metric_col], errors="coerce").mean())
        if len(closed) > 0
        else np.nan
    )

    return {
        "segment": segment.name,
        "start_date": segment.start,
        "end_date": segment.end,
        "final_equity": metric_value(stats, "Final Equity"),
        "total_return": metric_value(stats, "Final Equity") - 1.0 if np.isfinite(metric_value(stats, "Final Equity")) else np.nan,
        "sharpe": metric_value(stats, "Sharpe"),
        "cagr": metric_value(stats, "CAGR"),
        "max_drawdown": metric_value(stats, "Max Drawdown"),
        "nb_trades": int(stats.get("Nb Trades", 0)),
        "closed_trades": int(len(closed)),
        "hit_ratio": hit_ratio,
        "avg_trade_return": avg_trade_return,
        "trade_metric_sum": isolated_sum,
        "avg_open_positions": float(equity["n_open_positions"].mean()) if not equity.empty else np.nan,
        "pct_days_with_positions": float((equity["n_open_positions"] > 0).mean()) if not equity.empty else np.nan,
        "lookahead_violations": int(stats.get("Lookahead violations", 0)),
        "avg_scan_age_bdays": metric_value(stats, "Avg scan age (bdays)"),
        "anomaly_flag": bool(stats.get("Anomaly flag", False)),
        "anomaly_reasons": str(stats.get("Anomaly reasons", "")),
        "entry_threshold_hits": threshold_hits,
        "entry_accepted": accepted,
        "entry_blocked_total": blocked_total,
        "entry_accept_rate": (accepted / threshold_hits) if threshold_hits > 0 else np.nan,
    }


def flatten_segment_metrics(segment_metrics: dict[str, Any]) -> dict[str, Any]:
    prefix = segment_metrics["segment"].lower()
    out: dict[str, Any] = {}
    for k, v in segment_metrics.items():
        if k == "segment":
            continue
        out[f"{prefix}_{k}"] = v
    return out


def filter_summary_long(
    config_row: dict[str, Any],
    segment: SegmentSpec,
    entry_filter_summary: pd.DataFrame,
) -> pd.DataFrame:
    if entry_filter_summary.empty:
        return pd.DataFrame()

    out = entry_filter_summary.copy()
    out["config_id"] = config_row["config_id"]
    out["entry_mode"] = config_row["entry_mode"]
    out["variant_label"] = config_row["variant_label"]
    out["segment"] = segment.name
    threshold_hits = out.loc[out["metric"] == "entry_threshold_hits", "count"]
    denom = float(threshold_hits.iloc[0]) if len(threshold_hits) > 0 else np.nan
    out["ratio_vs_threshold_hits"] = out["count"] / denom if np.isfinite(denom) and denom > 0 else np.nan
    return out[["config_id", "entry_mode", "variant_label", "segment", "metric", "count", "ratio_vs_threshold_hits"]]


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


def compute_final_scores(run_level: pd.DataFrame) -> pd.DataFrame:
    df = run_level.copy()
    baseline = df[df["config_id"] == "baseline_entry"]
    if baseline.empty:
        raise RuntimeError("Missing baseline_entry in run-level results.")

    baseline_row = baseline.iloc[0]
    baseline_oos_sharpe = float(baseline_row["oos_sharpe"])
    baseline_oos_trades = int(baseline_row["oos_nb_trades"])

    oos_metric = pd.to_numeric(df["oos_sharpe"], errors="coerce")
    gap = (pd.to_numeric(df["is_sharpe"], errors="coerce") - oos_metric).abs()
    undertrade_penalty = np.maximum(0, baseline_oos_trades - pd.to_numeric(df["oos_nb_trades"], errors="coerce").fillna(0))

    df["is_oos_gap"] = gap
    df["delta_oos_sharpe_vs_baseline"] = oos_metric - baseline_oos_sharpe
    df["delta_oos_trades_vs_baseline"] = pd.to_numeric(df["oos_nb_trades"], errors="coerce") - baseline_oos_trades
    df["final_score"] = (
        oos_metric.fillna(-5.0)
        - PENALTY_GAP * gap.fillna(5.0)
        - PENALTY_TRADES * undertrade_penalty
    )
    df["beats_baseline_oos"] = df["delta_oos_sharpe_vs_baseline"] > BEAT_BASELINE_MIN_DELTA
    df["robust_oos"] = (
        (df["is_oos_gap"] <= ROBUST_GAP_MAX)
        & (pd.to_numeric(df["oos_lookahead_violations"], errors="coerce").fillna(1) == 0)
        & (~df["oos_anomaly_flag"].astype(bool))
    )
    return df


def summarize_by_entry_mode(run_level: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_level.groupby("entry_mode", as_index=False)
        .agg(
            n_configs=("config_id", "size"),
            avg_oos_sharpe=("oos_sharpe", "mean"),
            best_oos_sharpe=("oos_sharpe", "max"),
            avg_final_score=("final_score", "mean"),
            best_final_score=("final_score", "max"),
            avg_oos_nb_trades=("oos_nb_trades", "mean"),
            avg_is_oos_gap=("is_oos_gap", "mean"),
        )
        .sort_values("best_final_score", ascending=False)
    )
    return grouped.reset_index(drop=True)


def compare_best_vs_baseline(best_row: pd.Series, baseline_row: pd.Series) -> str:
    delta_sharpe = float(best_row["delta_oos_sharpe_vs_baseline"])
    delta_trades = float(best_row["delta_oos_trades_vs_baseline"])
    delta_hit = float(best_row["oos_hit_ratio"] - baseline_row["oos_hit_ratio"]) if np.isfinite(best_row["oos_hit_ratio"]) and np.isfinite(baseline_row["oos_hit_ratio"]) else np.nan

    if delta_sharpe <= BEAT_BASELINE_MIN_DELTA:
        return "No genuine OOS improvement versus baseline."
    if np.isfinite(delta_hit) and delta_hit > 0.02 and abs(delta_trades) <= max(3.0, 0.10 * max(1.0, baseline_row["oos_nb_trades"])):
        return "Improvement looks primarily like better timing rather than trade-count compression."
    if delta_trades < -0.20 * max(1.0, baseline_row["oos_nb_trades"]) and (not np.isfinite(delta_hit) or delta_hit <= 0.02):
        return "Improvement seems driven mostly by taking fewer trades, with limited evidence of better timing."
    return "Improvement is mixed: part timing, part change in trade count."


def build_conclusion_text(run_level: pd.DataFrame) -> str:
    ranking = run_level.sort_values(["final_score", "oos_sharpe"], ascending=[False, False]).reset_index(drop=True)
    best = ranking.iloc[0]
    baseline = ranking[ranking["config_id"] == "baseline_entry"].iloc[0]

    robustness_text = "acceptable" if bool(best["robust_oos"]) else "not acceptable"
    comparison_text = compare_best_vs_baseline(best, baseline)
    retain_text = (
        f"Retain {best['config_id']} for the next research step."
        if bool(best["beats_baseline_oos"]) and bool(best["robust_oos"])
        else "Keep baseline_entry as the reference for now and reject the challengers for promotion."
    )

    rejected = ranking.loc[ranking["config_id"] != best["config_id"], "config_id"].tolist()
    rejected_text = ", ".join(rejected[:6]) + ("..." if len(rejected) > 6 else "")

    lines = [
        "Sweden weekly entry-speed campaign conclusion",
        f"- Best configuration by final_score: {best['config_id']} ({best['variant_label']}).",
        f"- OOS Sharpe: {best['oos_sharpe']:.2f} vs baseline {baseline['oos_sharpe']:.2f}; OOS trades: {int(best['oos_nb_trades'])} vs baseline {int(baseline['oos_nb_trades'])}.",
        f"- IS/OOS gap: {best['is_oos_gap']:.2f} -> robustness is {robustness_text}.",
        f"- Look-ahead check: {int(best['oos_lookahead_violations'])} violation(s) on OOS and {int(best['full_lookahead_violations'])} on FULL.",
        f"- Interpretation: {comparison_text}",
        f"- Decision: {retain_text}",
        f"- Rejected variants: {rejected_text}",
    ]
    return "\n".join(lines)


def render_notebook_cell(best_row: pd.Series) -> str:
    params_literal = {
        "entry_mode": best_row["entry_mode"],
        "spread_speed_cap": best_row.get("spread_speed_cap"),
        "zspeed_cap": best_row.get("zspeed_cap"),
        "zspeed_ewma_span": best_row.get("zspeed_ewma_span"),
        "zspeed_ewma_cap": best_row.get("zspeed_ewma_cap"),
        "zspeed_vol_window": best_row.get("zspeed_vol_window"),
        "zspeed_vol_cap": best_row.get("zspeed_vol_cap"),
    }
    clean_params = {k: v for k, v in params_literal.items() if pd.notna(v)}
    extra_param_lines = "\n".join(f"    {k}={repr(v)}," for k, v in clean_params.items())
    if extra_param_lines:
        extra_param_lines = extra_param_lines + "\n"

    return f"""from pathlib import Path
import sys
import pandas as pd
import numpy as np
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
    if "eligibility_score" in out.columns:
        out["eligibility_score"] = pd.to_numeric(out["eligibility_score"], errors="coerce")
    else:
        out["eligibility_score"] = np.nan
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
    z_exit={BASE_Z_ENTRY / 3.0},
    z_stop={2.0 * BASE_Z_ENTRY},
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
{extra_param_lines})
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

for label, res in [("FULL", res_full), ("IS", res_is), ("OOS", res_oos)]:
    print(f"\\n{{label}} stats")
    for k, v in res["stats"].items():
        print(f"  {{k}}: {{v}}")
    print("  Closed trades:", int(pd.to_datetime(res["trades"]["exit_datetime"], errors="coerce").notna().sum()) if not res["trades"].empty else 0)

out_dir = PROJECT_ROOT / "data" / "experiments" / "sweden_weekly_entry_speed_campaign_2018_2025" / "notebook_reproduction"
out_dir.mkdir(parents=True, exist_ok=True)
res_full["trades"].to_csv(out_dir / "best_full_trades.csv", index=False)
res_full["scan_usage"].to_csv(out_dir / "best_full_scan_usage.csv", index=False)
print("\\nSaved notebook reproduction artifacts to", out_dir)
"""


def save_best_artifacts(best_row: pd.Series, best_results: dict[str, dict[str, Any]]) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for segment_name, res in best_results.items():
        seg = segment_name.lower()
        res["equity"].to_csv(BEST_DIR / f"{seg}_equity.csv", index=False)
        res["trades"].to_csv(BEST_DIR / f"{seg}_trades.csv", index=False)
        res["diagnostics"].to_csv(BEST_DIR / f"{seg}_diagnostics.csv", index=False)
        res["scan_usage"].to_csv(BEST_DIR / f"{seg}_scan_usage.csv", index=False)
        entry_filter_summary = res.get("entry_filter_summary", pd.DataFrame())
        if not entry_filter_summary.empty:
            entry_filter_summary.to_csv(BEST_DIR / f"{seg}_entry_filter_summary.csv", index=False)

    NOTEBOOK_CELL_PATH.write_text(render_notebook_cell(best_row), encoding="utf-8")


def update_campaign_journal(best_row: pd.Series, conclusion_text: str) -> None:
    decision_line = ""
    for line in conclusion_text.splitlines():
        if line.startswith("- Decision:"):
            decision_line = line.removeprefix("- Decision:").strip()
            break
    summary_lines = [
        f"Universe: {UNIVERSE} | FULL {FULL_START} -> {FULL_END} | IS {IS_START} -> {IS_END} | OOS {OOS_START} -> {OOS_END}.",
        "Scope: entry-mode research on Sweden weekly Friday scans with ELIGIBLE-only trading and baseline selection unchanged.",
        f"Retained config: {best_row['config_id']} | entry_mode={best_row['entry_mode']} | variant={best_row['variant_label']}.",
        (
            f"OOS Sharpe={best_row['oos_sharpe']:.2f} | OOS trades={int(best_row['oos_nb_trades'])} "
            f"| hit ratio={best_row['oos_hit_ratio']:.2%} | final_score={best_row['final_score']:.4f}."
        ),
        (
            f"Gap IS/OOS={best_row['is_oos_gap']:.2f} | delta OOS Sharpe vs baseline={best_row['delta_oos_sharpe_vs_baseline']:.2f} "
            f"| delta OOS trades vs baseline={int(best_row['delta_oos_trades_vs_baseline'])}."
        ),
        f"Look-ahead violations: FULL={int(best_row['full_lookahead_violations'])}, IS={int(best_row['is_lookahead_violations'])}, OOS={int(best_row['oos_lookahead_violations'])}.",
        f"Decision: {decision_line or 'See conclusion.txt.'}",
    ]
    upsert_campaign_entry(
        campaign_key="sweden_weekly_entry_speed_campaign_2018_2025",
        title="Sweden Weekly Entry Speed Campaign 2018-2025",
        summary_lines=summary_lines,
        out_dir=OUT_DIR,
        notebook_cell_path=NOTEBOOK_CELL_PATH,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scans = build_or_load_weekly_scans(rebuild=False)

    config_manifest = pd.DataFrame(build_campaign_configs())
    config_manifest.to_csv(OUT_DIR / "config_manifest.csv", index=False)

    run_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    filter_rows: list[pd.DataFrame] = []
    best_results_by_config: dict[str, dict[str, Any]] = {}

    for config_row in build_campaign_configs():
        print(f"[RUN] {config_row['config_id']}")
        merged_row = {
            "config_id": config_row["config_id"],
            "entry_mode": config_row["entry_mode"],
            "variant_label": config_row["variant_label"],
            "notes": config_row["notes"],
            "spread_speed_cap": config_row.get("spread_speed_cap", np.nan),
            "zspeed_cap": config_row.get("zspeed_cap", np.nan),
            "zspeed_ewma_span": config_row.get("zspeed_ewma_span", np.nan),
            "zspeed_ewma_cap": config_row.get("zspeed_ewma_cap", np.nan),
            "zspeed_vol_window": config_row.get("zspeed_vol_window", np.nan),
            "zspeed_vol_cap": config_row.get("zspeed_vol_cap", np.nan),
        }
        config_segment_results: dict[str, Any] = {}

        for segment in SEGMENTS:
            segment_metrics, res = run_segment(config_row, segment, scans)
            merged_row.update(flatten_segment_metrics(segment_metrics))
            long_row = {
                "config_id": config_row["config_id"],
                "entry_mode": config_row["entry_mode"],
                "variant_label": config_row["variant_label"],
                **segment_metrics,
            }
            long_rows.append(long_row)
            filter_rows.append(filter_summary_long(config_row, segment, res.get("entry_filter_summary", pd.DataFrame())))
            config_segment_results[segment.name] = res

        run_rows.append(merged_row)
        best_results_by_config[config_row["config_id"]] = config_segment_results

    run_level = pd.DataFrame(run_rows)
    run_level = compute_final_scores(run_level)
    ranking = run_level.sort_values(["final_score", "oos_sharpe"], ascending=[False, False]).reset_index(drop=True)
    summary_by_mode = summarize_by_entry_mode(run_level)
    long_df = pd.DataFrame(long_rows)
    filter_df = pd.concat([df for df in filter_rows if not df.empty], ignore_index=True) if filter_rows else pd.DataFrame()

    run_level.to_csv(OUT_DIR / "run_level.csv", index=False)
    long_df.to_csv(OUT_DIR / "run_level_segments_long.csv", index=False)
    summary_by_mode.to_csv(OUT_DIR / "summary_by_entry_mode.csv", index=False)
    ranking.to_csv(OUT_DIR / "ranking_final.csv", index=False)
    if not filter_df.empty:
        filter_df.to_csv(OUT_DIR / "entry_filter_diagnostics.csv", index=False)

    conclusion_text = build_conclusion_text(ranking)
    (OUT_DIR / "conclusion.txt").write_text(conclusion_text, encoding="utf-8")

    best_row = ranking.iloc[0]
    save_best_artifacts(best_row, best_results_by_config[str(best_row["config_id"])])
    update_campaign_journal(best_row, conclusion_text)

    metadata = {
        "universe": UNIVERSE,
        "full_period": [FULL_START, FULL_END],
        "is_period": [IS_START, IS_END],
        "oos_period": [OOS_START, OOS_END],
        "scan_frequency": SCAN_FREQUENCY,
        "scan_weekday": SCAN_WEEKDAY,
        "penalty_gap": PENALTY_GAP,
        "penalty_trades": PENALTY_TRADES,
        "best_config_id": str(best_row["config_id"]),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== BEST CONFIG ===")
    print(best_row[[
        "config_id",
        "entry_mode",
        "variant_label",
        "oos_sharpe",
        "is_sharpe",
        "is_oos_gap",
        "oos_nb_trades",
        "delta_oos_sharpe_vs_baseline",
        "delta_oos_trades_vs_baseline",
        "final_score",
    ]])
    print("\n=== CONCLUSION ===")
    print(conclusion_text)
    print("\nNotebook cell saved to:", NOTEBOOK_CELL_PATH)


if __name__ == "__main__":
    main()
