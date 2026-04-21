from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.global_loop import run_global_ranking_daily_portfolio
from object.class_file import BatchConfig, StrategyParams
from utils.inline_scanner import InlineScannerConfig, build_scans_inline
from utils.scanner import ELIGIBILITY_V1_BASELINE


UNIVERSE = "sweden"
START = "2018-01-01"
END = "2025-12-31"
IS_START = "2018-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2025-12-31"

SCAN_FREQ = "W-FRI"
SCAN_SCHEDULE = "weekly"
SCAN_WEEKDAY = "FRI"

BASE_Z_ENTRY = 1.8
BASE_Z_WINDOW = 60
BASE_MAX_HOLD = 30

DEFAULT_TOP_N = 20
DEFAULT_MAX_POSITIONS = 5
DEFAULT_FEES = 0.0002

PENALTY_GAP = 0.35
PENALTY_TRADES = 0.0015

BASE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"
OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "sweden_weekly_speedfilter_campaign_2018_2025"
SCAN_CACHE = OUT_DIR / "scans" / f"{UNIVERSE}_weekly_fri.parquet"


@dataclass(frozen=True)
class Segment:
    name: str
    start_date: str
    end_date: str


SEGMENTS = [
    Segment("FULL", START, END),
    Segment("IS", IS_START, IS_END),
    Segment("OOS", OOS_START, OOS_END),
]


def _safe(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def _linked_thresholds(z_entry: float) -> tuple[float, float]:
    return (round(float(z_entry) / 3.0, 4), round(2.0 * float(z_entry), 4))


def _normalize_scans(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["scan_date"] = pd.to_datetime(d["scan_date"]).dt.normalize()
    d["asset_1"] = d["asset_1"].astype(str).str.upper()
    d["asset_2"] = d["asset_2"].astype(str).str.upper()
    d["eligibility"] = d.get("eligibility", "").astype(str).str.upper()
    d["eligibility_score"] = pd.to_numeric(d.get("eligibility_score"), errors="coerce")
    d["universe"] = UNIVERSE
    d = d.sort_values(
        ["scan_date", "asset_1", "asset_2", "eligibility_score"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(subset=["scan_date", "asset_1", "asset_2"], keep="first")
    return d.reset_index(drop=True)


def load_or_build_weekly_scans(rebuild: bool = False) -> pd.DataFrame:
    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SCAN_CACHE.exists() and not rebuild:
        scans = _normalize_scans(pd.read_parquet(SCAN_CACHE))
        if not scans.empty:
            return scans

    scan_cfg = InlineScannerConfig(
        raw_data_path=BASE_DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
        eligibility_mode=ELIGIBILITY_V1_BASELINE,
    )
    scans = build_scans_inline(
        universes=[UNIVERSE],
        start_date=START,
        end_date=END,
        freq=SCAN_FREQ,
        cfg=scan_cfg,
        print_every=20,
    )
    if scans.empty:
        raise RuntimeError("Weekly scans are empty for Sweden.")
    scans = _normalize_scans(scans)
    scans.to_parquet(SCAN_CACHE, index=False)
    return scans


def build_variant_grid() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append({"family": "baseline_entry", "name": "baseline_ref", "entry_mode": "baseline_entry"})
    for w, m in [(15, 2.0), (15, 2.5), (20, 2.0), (20, 2.5)]:
        out.append({"family": "current_speed_filter", "name": f"current_w{w}_m{m}", "entry_mode": "current_speed_filter", "entry_speed_window": w, "entry_speed_max_multiple": m})
    for cap in [1.2, 1.6, 2.0]:
        out.append({"family": "zspeed_hard_cap", "name": f"zhard_{cap}", "entry_mode": "zspeed_hard_cap", "entry_speed_hard_cap": cap, "entry_speed_window": 20})
    for span, cap in itertools.product([3, 5], [1.0, 1.3]):
        out.append({"family": "zspeed_ewma_cap", "name": f"zewma_s{span}_c{cap}", "entry_mode": "zspeed_ewma_cap", "entry_speed_ewma_span": span, "entry_speed_ewma_cap": cap})
    for window, cap in itertools.product([15, 20], [1.5, 2.0]):
        out.append({"family": "spread_speed_vol_normalized", "name": f"spnorm_w{window}_c{cap}", "entry_mode": "spread_speed_vol_normalized", "entry_spread_vol_window": window, "entry_spread_speed_vol_cap": cap})
    for window, ratio in itertools.product([3, 5], [0.85, 0.95]):
        out.append({"family": "slowdown_confirmation", "name": f"slow_w{window}_r{ratio}", "entry_mode": "slowdown_confirmation", "entry_slowdown_window": window, "entry_slowdown_ratio_max": ratio})
    return out


def build_params(cfg_row: dict[str, Any]) -> StrategyParams:
    z_exit, z_stop = _linked_thresholds(BASE_Z_ENTRY)
    params = dict(
        z_entry=BASE_Z_ENTRY,
        z_exit=z_exit,
        z_stop=z_stop,
        z_window=BASE_Z_WINDOW,
        beta_mode="static",
        fees=DEFAULT_FEES,
        top_n_candidates=DEFAULT_TOP_N,
        max_positions=DEFAULT_MAX_POSITIONS,
        max_holding_days=BASE_MAX_HOLD,
        exec_lag_days=1,
        scan_schedule=SCAN_SCHEDULE,
        scan_weekday=SCAN_WEEKDAY,
        signal_space="raw",
        selection_mode="legacy",
        selection_score_variant="baseline",
        eligibility_labels=("ELIGIBLE",),
        entry_mode=cfg_row["entry_mode"],
    )
    for k in (
        "entry_speed_window",
        "entry_speed_max_multiple",
        "entry_speed_hard_cap",
        "entry_speed_ewma_span",
        "entry_speed_ewma_cap",
        "entry_spread_vol_window",
        "entry_spread_speed_vol_cap",
        "entry_slowdown_window",
        "entry_slowdown_ratio_max",
    ):
        if k in cfg_row:
            params[k] = cfg_row[k]
    return StrategyParams(**params)


def run_one_segment(scans: pd.DataFrame, seg: Segment, cfg_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = BatchConfig(data_path=BASE_DATA_PATH, start_date=seg.start_date, end_date=seg.end_date)
    params = build_params(cfg_row)
    t0 = time.time()
    res = run_global_ranking_daily_portfolio(cfg=cfg, params=params, universes=[UNIVERSE], scans=scans)
    runtime_s = time.time() - t0

    base = {"segment": seg.name, "family": cfg_row["family"], "name": cfg_row["name"], "entry_mode": cfg_row["entry_mode"], "runtime_s": round(runtime_s, 3)}
    if not res:
        row = {**base, "ok": False, "sharpe": np.nan, "cagr": np.nan, "max_drawdown": np.nan, "nb_trades": 0, "hit_ratio": np.nan, "avg_holding_days": np.nan, "entry_candidates_considered": 0, "entry_filtered_by_mode": 0, "entry_filter_rate": np.nan, "entry_filter_reasons": ""}
        return row, row.copy()

    st = dict(res.get("stats", {}))
    tr = res.get("trades", pd.DataFrame())
    closed = tr[tr["exit_datetime"].notna()].copy() if isinstance(tr, pd.DataFrame) and not tr.empty else pd.DataFrame()
    diag = res.get("diagnostics", pd.DataFrame())
    if isinstance(diag, pd.DataFrame) and not diag.empty:
        cand = int(pd.to_numeric(diag.get("entry_candidates_considered"), errors="coerce").fillna(0).sum())
        filt = int(pd.to_numeric(diag.get("entry_filtered_by_mode"), errors="coerce").fillna(0).sum())
    else:
        cand, filt = 0, 0

    row = {
        **base,
        "ok": True,
        "sharpe": _safe(st.get("Sharpe")),
        "cagr": _safe(st.get("CAGR")),
        "max_drawdown": _safe(st.get("Max Drawdown")),
        "nb_trades": int(st.get("Nb Trades", len(tr) if isinstance(tr, pd.DataFrame) else 0)),
        "hit_ratio": float((closed["trade_return"] > 0).mean()) if len(closed) > 0 else np.nan,
        "avg_holding_days": _safe(pd.to_numeric(closed.get("duration_days"), errors="coerce").mean()) if len(closed) > 0 else np.nan,
        "entry_candidates_considered": cand,
        "entry_filtered_by_mode": filt,
        "entry_filter_rate": (filt / cand) if cand > 0 else np.nan,
        "entry_filter_reasons": str(st.get("Entry filter reasons", "")),
    }
    return row, row.copy()


def run_campaign(scans: pd.DataFrame, grid: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, cfg_row in enumerate(grid, start=1):
        print(f"[{i:02d}/{len(grid):02d}] {cfg_row['name']} ({cfg_row['entry_mode']})")
        by_seg: dict[str, dict[str, Any]] = {}
        for seg in SEGMENTS:
            seg_row, seg_diag = run_one_segment(scans, seg, cfg_row)
            by_seg[seg.name] = seg_row
            diag_rows.append(seg_diag)
        full, is_row, oos = by_seg["FULL"], by_seg["IS"], by_seg["OOS"]
        is_sh, oos_sh = _safe(is_row["sharpe"]), _safe(oos["sharpe"])
        gap = is_sh - oos_sh if np.isfinite(is_sh) and np.isfinite(oos_sh) else np.nan
        row = {
            "family": cfg_row["family"],
            "name": cfg_row["name"],
            "entry_mode": cfg_row["entry_mode"],
            "z_entry": BASE_Z_ENTRY,
            "z_window": BASE_Z_WINDOW,
            "max_holding_days": BASE_MAX_HOLD,
            "scan_freq": SCAN_FREQ,
            "scan_schedule": SCAN_SCHEDULE,
            "scan_weekday": SCAN_WEEKDAY,
            "full_sharpe": _safe(full["sharpe"]),
            "is_sharpe": is_sh,
            "oos_sharpe": oos_sh,
            "full_cagr": _safe(full["cagr"]),
            "full_max_drawdown": _safe(full["max_drawdown"]),
            "nb_trades": int(full["nb_trades"]),
            "hit_ratio": _safe(full["hit_ratio"]),
            "avg_holding_days": _safe(full["avg_holding_days"]),
            "entry_candidates_considered": int(full["entry_candidates_considered"]),
            "entry_filtered_by_mode": int(full["entry_filtered_by_mode"]),
            "entry_filter_rate": _safe(full["entry_filter_rate"]),
            "entry_filter_reasons": str(full["entry_filter_reasons"]),
            "gap_is_oos": _safe(gap),
            "gap_is_oos_abs": _safe(abs(gap) if np.isfinite(gap) else np.nan),
        }
        for k, v in cfg_row.items():
            if k not in row:
                row[k] = v
        rows.append(row)
        elapsed = max(1e-6, time.time() - t0)
        rate = i / elapsed
        eta = (len(grid) - i) / rate if rate > 0 else np.nan
        print(f"  progress={i/len(grid):.1%} rate={rate:.2f} cfg/s eta={eta/60:.1f} min")
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def score_and_label(runs: pd.DataFrame) -> pd.DataFrame:
    d = runs.copy()
    base = d[d["family"] == "baseline_entry"].sort_values(["oos_sharpe", "gap_is_oos_abs"], ascending=[False, True]).iloc[0]
    base_oos = _safe(base["oos_sharpe"])
    base_gap = _safe(base["gap_is_oos_abs"])
    trades_ref = max(1.0, _safe(base["nb_trades"]))
    d["decision_score"] = pd.to_numeric(d["oos_sharpe"], errors="coerce") - PENALTY_GAP * pd.to_numeric(d["gap_is_oos_abs"], errors="coerce") - PENALTY_TRADES * np.maximum(0.0, trades_ref - pd.to_numeric(d["nb_trades"], errors="coerce"))
    labels = []
    for r in d.itertuples(index=False):
        oos, gap, tr = _safe(r.oos_sharpe), _safe(r.gap_is_oos_abs), _safe(r.nb_trades)
        if np.isfinite(oos) and np.isfinite(gap) and np.isfinite(tr) and oos > base_oos and gap <= base_gap + 0.25 and tr >= 0.75 * trades_ref:
            labels.append("retenir")
        elif np.isfinite(oos) and np.isfinite(gap) and oos >= base_oos - 0.20 and gap <= base_gap + 0.60:
            labels.append("a_surveiller")
        else:
            labels.append("rejeter")
    d["decision_label"] = labels
    return d


def write_summary(runs: pd.DataFrame, ranking: pd.DataFrame) -> None:
    base = runs[runs["family"] == "baseline_entry"].sort_values(["oos_sharpe", "gap_is_oos_abs"], ascending=[False, True]).iloc[0]
    base_oos = _safe(base["oos_sharpe"])
    better = runs[pd.to_numeric(runs["oos_sharpe"], errors="coerce") > base_oos].copy()
    top = ranking.iloc[0] if not ranking.empty else None
    retain = ranking[ranking["decision_label"] == "retenir"].copy() if not ranking.empty else pd.DataFrame()
    if not retain.empty:
        focus = retain.sort_values(["decision_score", "oos_sharpe"], ascending=[False, False]).iloc[0]
    else:
        watch = ranking[ranking["decision_label"] == "a_surveiller"].copy() if not ranking.empty else pd.DataFrame()
        focus = watch.sort_values(["decision_score", "oos_sharpe"], ascending=[False, False]).iloc[0] if not watch.empty else top
    lines = []
    lines.append("Sweden weekly speed-filter campaign")
    lines.append(f"Universe={UNIVERSE} | FULL={START}->{END} | IS={IS_START}->{IS_END} | OOS={OOS_START}->{OOS_END}")
    lines.append(f"Scan={SCAN_FREQ} schedule={SCAN_SCHEDULE} weekday={SCAN_WEEKDAY}")
    lines.append(f"Configs={len(runs)}")
    lines.append("")
    lines.append("1) Variante qui bat baseline OOS ?")
    if better.empty:
        lines.append(f"- Non (baseline OOS={base_oos:.2f}).")
    else:
        lines.append(f"- Oui (baseline OOS={base_oos:.2f}).")
        for r in better.sort_values("oos_sharpe", ascending=False).head(6).itertuples(index=False):
            lines.append(f"  {r.name} | family={r.family} | OOS={_safe(r.oos_sharpe):.2f} | gap={_safe(r.gap_is_oos_abs):.2f} | trades={int(_safe(r.nb_trades))}")
    lines.append("")
    lines.append("2) Gain: timing ou debit de trades ?")
    if focus is None:
        lines.append("- N/A")
    else:
        d_tr = _safe(focus["nb_trades"]) - _safe(base["nb_trades"])
        d_hit = _safe(focus["hit_ratio"]) - _safe(base["hit_ratio"])
        if d_hit > 0.02 and d_tr > -0.2 * max(1.0, _safe(base.nb_trades)):
            verdict = "plutot timing/qualite"
        elif d_tr < 0 and d_hit <= 0.02:
            verdict = "plutot baisse du debit de trades"
        else:
            verdict = "mixte"
        lines.append(f"- Focus={focus['name']} vs baseline: delta_trades={d_tr:.1f}, delta_hit={d_hit:.3f} -> {verdict}")
    lines.append("")
    lines.append("3) Robustesse IS/OOS acceptable ?")
    if focus is None:
        lines.append("- N/A")
    else:
        gap = _safe(focus["gap_is_oos_abs"])
        lines.append(f"- {'Oui' if np.isfinite(gap) and gap <= 1.0 else 'Non'} (gap_abs={gap:.2f})")
    lines.append("")
    lines.append("4) Variante retenue")
    if focus is None:
        lines.append("- Aucune")
    else:
        lines.append(f"- {focus['name']} | family={focus['family']} | mode={focus['entry_mode']} | label={focus['decision_label']} | OOS={_safe(focus['oos_sharpe']):.2f}")
    (OUT_DIR / "weekly_speedfilter_campaign_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def select_focus_variant(ranking: pd.DataFrame) -> pd.Series | None:
    if ranking.empty:
        return None
    retain = ranking[ranking["decision_label"] == "retenir"].copy()
    if not retain.empty:
        return retain.sort_values(["decision_score", "oos_sharpe"], ascending=[False, False]).iloc[0]
    watch = ranking[ranking["decision_label"] == "a_surveiller"].copy()
    if not watch.empty:
        return watch.sort_values(["decision_score", "oos_sharpe"], ascending=[False, False]).iloc[0]
    return ranking.iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scans = load_or_build_weekly_scans(rebuild=False)
    print(f"Scans rows={len(scans):,} dates={scans['scan_date'].nunique()} range={scans['scan_date'].min().date()}->{scans['scan_date'].max().date()}")

    grid = build_variant_grid()
    runs, diag = run_campaign(scans, grid)
    runs = score_and_label(runs)
    runs = runs.sort_values(["decision_score", "oos_sharpe"], ascending=[False, False]).reset_index(drop=True)
    runs.to_csv(OUT_DIR / "runs_weekly_speedfilter_campaign.csv", index=False)

    by_mode = (
        runs.groupby("family", as_index=False)
        .agg(
            n_configs=("name", "size"),
            best_oos_sharpe=("oos_sharpe", "max"),
            median_oos_sharpe=("oos_sharpe", "median"),
            median_gap_abs=("gap_is_oos_abs", "median"),
            median_nb_trades=("nb_trades", "median"),
            median_hit_ratio=("hit_ratio", "median"),
            median_filter_rate=("entry_filter_rate", "median"),
            best_decision_score=("decision_score", "max"),
            retain_rate=("decision_label", lambda s: float((s == "retenir").mean())),
            watch_rate=("decision_label", lambda s: float((s == "a_surveiller").mean())),
        )
        .sort_values(["best_decision_score", "best_oos_sharpe"], ascending=False)
        .reset_index(drop=True)
    )
    by_mode.to_csv(OUT_DIR / "summary_by_entry_variant.csv", index=False)

    ranking = (
        runs.sort_values(["family", "decision_score", "oos_sharpe", "gap_is_oos_abs"], ascending=[True, False, False, True])
        .drop_duplicates("family")
        .sort_values(["decision_score", "oos_sharpe"], ascending=[False, False])
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking[["rank", "family", "name", "entry_mode", "oos_sharpe", "is_sharpe", "gap_is_oos_abs", "nb_trades", "hit_ratio", "entry_filter_rate", "decision_score", "decision_label"]].to_csv(OUT_DIR / "final_variant_ranking.csv", index=False)

    scan_diag = pd.DataFrame([{"scope": "scanner_weekly_baseline", "universe": UNIVERSE, "scan_freq": SCAN_FREQ, "scan_dates": int(scans["scan_date"].nunique()), "n_rows": int(len(scans)), "n_eligible": int((scans["eligibility"] == "ELIGIBLE").sum()), "n_watch": int((scans["eligibility"] == "WATCH").sum()), "n_out": int((scans["eligibility"] == "OUT").sum())}])
    pd.concat([diag, scan_diag], ignore_index=True, sort=False).to_csv(OUT_DIR / "filtered_trade_diagnostics.csv", index=False)

    write_summary(runs, ranking)
    focus = select_focus_variant(ranking)
    if focus is not None:
        focus_row = runs[runs["name"] == focus["name"]].iloc[0]
        focus_row.to_frame().T.to_csv(OUT_DIR / "best_config_row.csv", index=False)

    print("Saved outputs in:", OUT_DIR)
    print(" - runs_weekly_speedfilter_campaign.csv")
    print(" - summary_by_entry_variant.csv")
    print(" - filtered_trade_diagnostics.csv")
    print(" - final_variant_ranking.csv")
    print(" - weekly_speedfilter_campaign_summary.txt")
    print(" - best_config_row.csv")


if __name__ == "__main__":
    main()
