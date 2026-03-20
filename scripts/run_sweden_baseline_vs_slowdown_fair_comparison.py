from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
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

BASELINE_Z_ENTRY = 1.5
BASELINE_Z_EXIT = 0.5
BASELINE_Z_STOP = 3.0
BASELINE_Z_WINDOW = 40
BASELINE_MAX_HOLD = 20
BASELINE_TOP_N = 20
BASELINE_MAX_POSITIONS = 5
BASELINE_FEES = 0.0002

SLOWDOWN_Z_ENTRY_GRID = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

PENALTY_GAP = 0.35
PENALTY_ACTIVITY = 1.25
BASELINE_BEAT_MIN_DELTA = 0.03

OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "sweden_baseline_vs_slowdown_fair_comparison_2018_2025"
SCAN_DIR = OUT_DIR / "scan_cache"
BEST_DIR = OUT_DIR / "best_artifacts"
SCAN_CACHE_PATH = SCAN_DIR / "sweden_weekly_fri_scans.parquet"
LEGACY_SCAN_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "sweden_weekly_entry_speed_campaign_2018_2025"
    / "scan_cache"
    / "sweden_weekly_fri_scans.parquet"
)
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
        return normalize_scans(pd.read_parquet(SCAN_CACHE_PATH), UNIVERSE)

    if LEGACY_SCAN_CACHE_PATH.exists() and not rebuild:
        shutil.copy2(LEGACY_SCAN_CACHE_PATH, SCAN_CACHE_PATH)
        return normalize_scans(pd.read_parquet(SCAN_CACHE_PATH), UNIVERSE)

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


def scans_for_segment(scans: pd.DataFrame, segment: SegmentSpec) -> pd.DataFrame:
    out = scans.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"]).dt.normalize()
    buffer_start = (pd.Timestamp(segment.start) - BDay(20)).normalize()
    end = pd.Timestamp(segment.end).normalize()
    return out[(out["scan_date"] >= buffer_start) & (out["scan_date"] <= end)].reset_index(drop=True)


def baseline_strategy_kwargs(z_entry: float, entry_mode: str) -> dict[str, Any]:
    return {
        "z_entry": float(z_entry),
        "z_exit": BASELINE_Z_EXIT,
        "z_stop": BASELINE_Z_STOP,
        "z_window": BASELINE_Z_WINDOW,
        "beta_mode": "static",
        "fees": BASELINE_FEES,
        "top_n_candidates": BASELINE_TOP_N,
        "max_positions": BASELINE_MAX_POSITIONS,
        "max_holding_days": BASELINE_MAX_HOLD,
        "exec_lag_days": 1,
        "scan_frequency": SCAN_FREQUENCY,
        "scan_weekday": SCAN_WEEKDAY,
        "signal_space": "raw",
        "selection_mode": "legacy",
        "eligibility_labels": ("ELIGIBLE",),
        "entry_mode": entry_mode,
    }


def build_configs() -> list[dict[str, Any]]:
    configs = [
        {
            "config_id": "baseline_entry",
            "entry_mode": "baseline_entry",
            "variant_label": "baseline_anchor",
            "z_entry": BASELINE_Z_ENTRY,
            "notes": "Reference baseline kept fixed.",
        }
    ]
    for z_entry in SLOWDOWN_Z_ENTRY_GRID:
        label = str(z_entry).replace(".", "p")
        configs.append(
            {
                "config_id": f"slowdown_zentry_{label}",
                "entry_mode": "entry_slowdown_confirmation",
                "variant_label": f"slowdown_z_entry_{z_entry:.1f}",
                "z_entry": float(z_entry),
                "notes": "Same slowdown rule, only z_entry recalibrated to match OOS activity.",
            }
        )
    return configs


def closed_trades_frame(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    out["exit_datetime"] = pd.to_datetime(out["exit_datetime"], errors="coerce")
    return out[out["exit_datetime"].notna()].copy()


def metric_value(stats: dict[str, Any], key: str) -> float:
    val = stats.get(key, np.nan)
    try:
        return float(val)
    except Exception:
        return np.nan


def summarize_segment(segment: SegmentSpec, res: dict[str, Any]) -> dict[str, Any]:
    stats = res["stats"]
    trades = res["trades"].copy()
    closed = closed_trades_frame(trades)
    equity = res["equity"].copy()

    trade_metric_col = "trade_return_isolated"
    if closed.empty or not closed.get(trade_metric_col, pd.Series(dtype=float)).notna().any():
        trade_metric_col = "trade_return"

    trade_metric_sum = (
        float(pd.to_numeric(closed[trade_metric_col], errors="coerce").sum())
        if len(closed) > 0
        else np.nan
    )
    hit_ratio = (
        float((pd.to_numeric(closed[trade_metric_col], errors="coerce") > 0).mean())
        if len(closed) > 0
        else np.nan
    )

    return {
        "segment": segment.name,
        "start_date": segment.start,
        "end_date": segment.end,
        "sharpe": metric_value(stats, "Sharpe"),
        "nb_trades": int(stats.get("Nb Trades", 0)),
        "hit_ratio": hit_ratio,
        "trade_metric_sum": trade_metric_sum,
        "final_equity": metric_value(stats, "Final Equity"),
        "cagr": metric_value(stats, "CAGR"),
        "max_drawdown": metric_value(stats, "Max Drawdown"),
        "lookahead_violations": int(stats.get("Lookahead violations", 0)),
        "anomaly_flag": bool(stats.get("Anomaly flag", False)),
        "anomaly_reasons": str(stats.get("Anomaly reasons", "")),
        "avg_open_positions": float(equity["n_open_positions"].mean()) if not equity.empty else np.nan,
    }


def flatten_segment(segment_metrics: dict[str, Any]) -> dict[str, Any]:
    prefix = segment_metrics["segment"].lower()
    return {f"{prefix}_{k}": v for k, v in segment_metrics.items() if k != "segment"}


def run_config(config_row: dict[str, Any], scans: pd.DataFrame) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    row = {
        "config_id": config_row["config_id"],
        "entry_mode": config_row["entry_mode"],
        "variant_label": config_row["variant_label"],
        "notes": config_row["notes"],
        "z_entry": config_row["z_entry"],
    }
    results_by_segment: dict[str, dict[str, Any]] = {}

    for segment in SEGMENTS:
        params = StrategyParams(**baseline_strategy_kwargs(config_row["z_entry"], config_row["entry_mode"]))
        cfg = BatchConfig(
            data_path=DATA_PATH,
            start_date=segment.start,
            end_date=segment.end,
        )
        seg_scans = scans_for_segment(scans, segment)
        res = run_global_ranking_daily_portfolio(
            cfg=cfg,
            params=params,
            universes=[UNIVERSE],
            scans=seg_scans,
        )
        if not res:
            raise RuntimeError(f"No result for {config_row['config_id']} on {segment.name}.")
        row.update(flatten_segment(summarize_segment(segment, res)))
        results_by_segment[segment.name] = res

    return row, results_by_segment


def activity_zone(gap_pct: float) -> str:
    if not np.isfinite(gap_pct):
        return "non_comparable"
    if gap_pct <= 0.10:
        return "comparable"
    if gap_pct <= 0.20:
        return "acceptable"
    return "non_comparable"


def add_comparison_metrics(run_level: pd.DataFrame) -> pd.DataFrame:
    df = run_level.copy()
    baseline = df[df["config_id"] == "baseline_entry"]
    if baseline.empty:
        raise RuntimeError("Missing baseline_entry row.")
    base = baseline.iloc[0]

    baseline_oos_trades = float(base["oos_nb_trades"])
    baseline_oos_sharpe = float(base["oos_sharpe"])
    baseline_oos_hit = float(base["oos_hit_ratio"])
    baseline_oos_trade_metric_sum = float(base["oos_trade_metric_sum"])

    df["trade_count_gap_pct"] = (
        (pd.to_numeric(df["oos_nb_trades"], errors="coerce") - baseline_oos_trades).abs() / baseline_oos_trades
    )
    df["activity_zone"] = df["trade_count_gap_pct"].apply(activity_zone)
    df["is_oos_gap"] = (
        pd.to_numeric(df["is_sharpe"], errors="coerce") - pd.to_numeric(df["oos_sharpe"], errors="coerce")
    ).abs()
    df["delta_oos_sharpe_vs_baseline"] = pd.to_numeric(df["oos_sharpe"], errors="coerce") - baseline_oos_sharpe
    df["delta_oos_hit_ratio_vs_baseline"] = pd.to_numeric(df["oos_hit_ratio"], errors="coerce") - baseline_oos_hit
    df["delta_oos_trade_metric_sum_vs_baseline"] = (
        pd.to_numeric(df["oos_trade_metric_sum"], errors="coerce") - baseline_oos_trade_metric_sum
    )
    df["zone_priority"] = df["activity_zone"].map({"comparable": 0, "acceptable": 1, "non_comparable": 2}).fillna(2)
    df["final_score"] = (
        pd.to_numeric(df["oos_sharpe"], errors="coerce").fillna(-5.0)
        - PENALTY_GAP * df["is_oos_gap"].fillna(5.0)
        - PENALTY_ACTIVITY * df["trade_count_gap_pct"].fillna(1.0)
    )

    def verdict(row: pd.Series) -> str:
        if row["config_id"] == "baseline_entry":
            return "baseline_anchor"
        if row["activity_zone"] == "non_comparable":
            return "reject_not_fair"
        if row["delta_oos_sharpe_vs_baseline"] > BASELINE_BEAT_MIN_DELTA and row["final_score"] > float(base["final_score"]):
            return "promote_slowdown"
        if row["delta_oos_sharpe_vs_baseline"] > 0.0:
            return "fair_but_not_better_than_baseline"
        return "baseline_still_better"

    df["verdict_final"] = df.apply(verdict, axis=1)
    return df


def build_comparison_table(run_level: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "config_id",
        "entry_mode",
        "variant_label",
        "z_entry",
        "full_sharpe",
        "full_nb_trades",
        "full_hit_ratio",
        "is_sharpe",
        "is_nb_trades",
        "is_hit_ratio",
        "oos_sharpe",
        "oos_nb_trades",
        "oos_hit_ratio",
        "oos_trade_metric_sum",
        "is_oos_gap",
        "trade_count_gap_pct",
        "activity_zone",
        "delta_oos_sharpe_vs_baseline",
        "delta_oos_hit_ratio_vs_baseline",
        "delta_oos_trade_metric_sum_vs_baseline",
        "final_score",
        "verdict_final",
    ]
    ordered = run_level.sort_values(["zone_priority", "final_score", "oos_sharpe"], ascending=[True, False, False]).reset_index(drop=True)
    return ordered[cols].copy()


def build_conclusion(run_level: pd.DataFrame) -> str:
    baseline = run_level[run_level["config_id"] == "baseline_entry"].iloc[0]
    slowdown = run_level[run_level["entry_mode"] == "entry_slowdown_confirmation"].copy()
    slowdown_fair = slowdown[slowdown["activity_zone"].isin(["comparable", "acceptable"])].copy()

    if slowdown_fair.empty:
        best_fair = None
    else:
        best_fair = slowdown_fair.sort_values(["zone_priority", "final_score", "oos_sharpe"], ascending=[True, False, False]).iloc[0]

    if best_fair is None:
        answer_line = "No slowdown config reached comparable or acceptable OOS activity versus baseline."
        decision_line = "Keep baseline_entry."
    elif (
        float(best_fair["delta_oos_sharpe_vs_baseline"]) > BASELINE_BEAT_MIN_DELTA
        and float(best_fair["delta_oos_hit_ratio_vs_baseline"]) >= 0.0
        and float(best_fair["delta_oos_trade_metric_sum_vs_baseline"]) >= 0.0
        and str(best_fair["activity_zone"]) in {"comparable", "acceptable"}
    ):
        answer_line = (
            f"Yes: {best_fair['config_id']} beats baseline at fair-comparable OOS activity "
            f"({best_fair['activity_zone']}, gap={best_fair['trade_count_gap_pct']:.2%})."
        )
        decision_line = f"Keep {best_fair['config_id']} for next-step validation."
    else:
        answer_line = "No: once OOS activity is brought close to baseline, slowdown does not beat baseline cleanly."
        decision_line = "Keep baseline_entry."

    timing_line = "No fair-comparable slowdown candidate available to judge timing."
    if best_fair is not None:
        if (
            float(best_fair["delta_oos_sharpe_vs_baseline"]) > BASELINE_BEAT_MIN_DELTA
            and float(best_fair["delta_oos_hit_ratio_vs_baseline"]) > 0.0
            and float(best_fair["delta_oos_trade_metric_sum_vs_baseline"]) >= 0.0
        ):
            timing_line = "The gain is consistent with better timing, not just fewer trades."
        elif float(best_fair["delta_oos_sharpe_vs_baseline"]) > 0.0:
            timing_line = "Any residual slowdown gain still looks partly explained by activity mix rather than better timing."
        else:
            timing_line = "Slowdown still brings no convincing timing edge once the trade-count effect is neutralized."

    fair_line = "Best fair-comparable slowdown: none."
    if best_fair is not None:
        fair_line = (
            f"Best fair-comparable slowdown: {best_fair['config_id']} | "
            f"OOS Sharpe {best_fair['oos_sharpe']:.2f} vs baseline {baseline['oos_sharpe']:.2f} | "
            f"OOS trades {int(best_fair['oos_nb_trades'])} vs baseline {int(baseline['oos_nb_trades'])} | "
            f"gap {best_fair['trade_count_gap_pct']:.2%} ({best_fair['activity_zone']})."
        )

    lookahead_ok = int(pd.to_numeric(run_level["full_lookahead_violations"], errors="coerce").fillna(0).max()) == 0
    lookahead_line = "Look-ahead check: 0 violation on all FULL/IS/OOS runs." if lookahead_ok else "Look-ahead check failed."

    lines = [
        "Sweden baseline vs slowdown fair comparison",
        f"- Baseline reference: baseline_entry with z_entry={BASELINE_Z_ENTRY}, z_exit={BASELINE_Z_EXIT}, z_stop={BASELINE_Z_STOP}, z_window={BASELINE_Z_WINDOW}, max_holding_days={BASELINE_MAX_HOLD}.",
        f"- {answer_line}",
        f"- {timing_line}",
        f"- {fair_line}",
        f"- {lookahead_line}",
        f"- Decision: {decision_line}",
    ]
    return "\n".join(lines)


def render_notebook_cell(best_row: pd.Series) -> str:
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
    z_entry={float(best_row["z_entry"])},
    z_exit={BASELINE_Z_EXIT},
    z_stop={BASELINE_Z_STOP},
    z_window={BASELINE_Z_WINDOW},
    beta_mode="static",
    fees={BASELINE_FEES},
    top_n_candidates={BASELINE_TOP_N},
    max_positions={BASELINE_MAX_POSITIONS},
    max_holding_days={BASELINE_MAX_HOLD},
    exec_lag_days=1,
    scan_frequency="{SCAN_FREQUENCY}",
    scan_weekday="{SCAN_WEEKDAY}",
    signal_space="raw",
    selection_mode="legacy",
    eligibility_labels=("ELIGIBLE",),
    entry_mode="{best_row['entry_mode']}",
)

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
    closed_trades = int(pd.to_datetime(res["trades"]["exit_datetime"], errors="coerce").notna().sum()) if not res["trades"].empty else 0
    print("  Closed trades:", closed_trades)

out_dir = PROJECT_ROOT / "data" / "experiments" / "sweden_baseline_vs_slowdown_fair_comparison_2018_2025" / "notebook_reproduction"
out_dir.mkdir(parents=True, exist_ok=True)
res_full["trades"].to_csv(out_dir / "full_trades.csv", index=False)
res_full["scan_usage"].to_csv(out_dir / "full_scan_usage.csv", index=False)
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
    NOTEBOOK_CELL_PATH.write_text(render_notebook_cell(best_row), encoding="utf-8")


def update_campaign_journal(best_row: pd.Series, conclusion_text: str) -> None:
    decision_line = ""
    for line in conclusion_text.splitlines():
        if line.startswith("- Decision:"):
            decision_line = line.removeprefix("- Decision:").strip()
            break
    summary_lines = [
        f"Universe: {UNIVERSE} | FULL {FULL_START} -> {FULL_END} | IS {IS_START} -> {IS_END} | OOS {OOS_START} -> {OOS_END}.",
        "Scope: fair comparison baseline_entry vs slowdown with activity normalized, weekly Friday scans, ELIGIBLE-only.",
        f"Retained config: {best_row['config_id']} | entry_mode={best_row['entry_mode']} | variant={best_row['variant_label']}.",
        (
            f"OOS Sharpe={best_row['oos_sharpe']:.2f} | OOS trades={int(best_row['oos_nb_trades'])} "
            f"| hit ratio={best_row['oos_hit_ratio']:.2%} | final_score={best_row['final_score']:.4f}."
        ),
        (
            f"Gap IS/OOS={best_row['is_oos_gap']:.2f} | trade_count_gap_pct={best_row['trade_count_gap_pct']:.1f}% "
            f"| verdict={best_row.get('verdict_final', 'n/a')}."
        ),
        f"Decision: {decision_line or 'See conclusion.txt.'}",
    ]
    upsert_campaign_entry(
        campaign_key="sweden_baseline_vs_slowdown_fair_comparison_2018_2025",
        title="Sweden Baseline vs Slowdown Fair Comparison 2018-2025",
        summary_lines=summary_lines,
        out_dir=OUT_DIR,
        notebook_cell_path=NOTEBOOK_CELL_PATH,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scans = build_or_load_weekly_scans(rebuild=False)

    configs = build_configs()
    pd.DataFrame(configs).to_csv(OUT_DIR / "config_manifest.csv", index=False)

    run_rows: list[dict[str, Any]] = []
    results_by_config: dict[str, dict[str, Any]] = {}
    for config_row in configs:
        print(f"[RUN] {config_row['config_id']}")
        row, results = run_config(config_row, scans)
        run_rows.append(row)
        results_by_config[config_row["config_id"]] = results

    run_level = pd.DataFrame(run_rows)
    run_level = add_comparison_metrics(run_level)
    ranking_final = build_comparison_table(run_level)
    comparison = ranking_final.copy()

    run_level.to_csv(OUT_DIR / "run_level.csv", index=False)
    ranking_final.to_csv(OUT_DIR / "ranking_final.csv", index=False)
    comparison.to_csv(OUT_DIR / "comparison_baseline_vs_slowdown.csv", index=False)

    conclusion = build_conclusion(run_level)
    (OUT_DIR / "conclusion.txt").write_text(conclusion, encoding="utf-8")

    slowdown_fair = ranking_final[
        (ranking_final["entry_mode"] == "entry_slowdown_confirmation")
        & (ranking_final["activity_zone"].isin(["comparable", "acceptable"]))
    ]
    best_row = ranking_final[ranking_final["config_id"] == "baseline_entry"].iloc[0]
    if not slowdown_fair.empty:
        candidate = slowdown_fair.iloc[0]
        baseline = ranking_final[ranking_final["config_id"] == "baseline_entry"].iloc[0]
        if (
            float(candidate["delta_oos_sharpe_vs_baseline"]) > BASELINE_BEAT_MIN_DELTA
            and float(candidate["final_score"]) > float(baseline["final_score"])
        ):
            best_row = candidate

    save_best_artifacts(best_row, results_by_config[str(best_row["config_id"])])
    update_campaign_journal(best_row, conclusion)

    metadata = {
        "universe": UNIVERSE,
        "full_period": [FULL_START, FULL_END],
        "is_period": [IS_START, IS_END],
        "oos_period": [OOS_START, OOS_END],
        "scan_frequency": SCAN_FREQUENCY,
        "scan_weekday": SCAN_WEEKDAY,
        "baseline_reference": {
            "z_entry": BASELINE_Z_ENTRY,
            "z_exit": BASELINE_Z_EXIT,
            "z_stop": BASELINE_Z_STOP,
            "z_window": BASELINE_Z_WINDOW,
            "max_holding_days": BASELINE_MAX_HOLD,
            "top_n_candidates": BASELINE_TOP_N,
            "max_positions": BASELINE_MAX_POSITIONS,
            "eligibility_labels": ["ELIGIBLE"],
        },
        "best_config_id": str(best_row["config_id"]),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== FAIR COMPARISON CONCLUSION ===")
    print(conclusion)
    print("\n=== TOP RANKING ===")
    print(ranking_final[[
        "config_id",
        "z_entry",
        "oos_sharpe",
        "oos_nb_trades",
        "trade_count_gap_pct",
        "activity_zone",
        "final_score",
        "verdict_final",
    ]].head(8))
    print("\nNotebook cell saved to:", NOTEBOOK_CELL_PATH)


if __name__ == "__main__":
    main()
