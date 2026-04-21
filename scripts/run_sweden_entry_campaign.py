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
SCAN_FREQ = "ME"

BASE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"

OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "sweden_entry_campaign_2018_2025"
SCAN_CACHE_DIR = OUT_DIR / "scans"
LEGACY_SCAN_CACHE = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "robust_cross_sectional_long_2015_2025"
    / "scans"
    / f"{UNIVERSE}.parquet"
)

ENTRY_MODES = [
    "baseline_entry",
    "entry_with_reversal_confirmation",
    "entry_vol_adjusted",
    "entry_half_life_aware",
    "entry_with_spread_speed_filter",
]
Z_ENTRY_GRID = [1.5, 1.8]
Z_WINDOW_GRID = [60, 100]
MAX_HOLD_GRID = [20, 30]

DEFAULT_TOP_N = 20
DEFAULT_MAX_POSITIONS = 5
DEFAULT_FEES = 0.0002
DEFAULT_BETA_MODE = "static"
DEFAULT_SIGNAL_SPACE = "raw"


@dataclass(frozen=True)
class Segment:
    name: str
    start_date: str
    end_date: str


SEGMENTS = [
    Segment(name="FULL", start_date=START, end_date=END),
    Segment(name="IS", start_date=IS_START, end_date=IS_END),
    Segment(name="OOS", start_date=OOS_START, end_date=OOS_END),
]


def linked_thresholds(z_entry: float) -> tuple[float, float]:
    return (round(float(z_entry) / 3.0, 4), round(2.0 * float(z_entry), 4))


def _safe_float(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _fmt(v: Any, digits: int = 3) -> str:
    x = _safe_float(v)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def normalize_scans(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"]).dt.normalize()
    out["asset_1"] = out["asset_1"].astype(str).str.upper()
    out["asset_2"] = out["asset_2"].astype(str).str.upper()
    out["eligibility"] = out.get("eligibility", "").astype(str).str.upper()
    out["universe"] = UNIVERSE
    out["eligibility_score"] = pd.to_numeric(out.get("eligibility_score"), errors="coerce")
    out = out.sort_values(
        ["scan_date", "asset_1", "asset_2", "eligibility_score"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(subset=["scan_date", "asset_1", "asset_2"], keep="first")
    return out.reset_index(drop=True)


def load_or_build_sweden_scans(*, rebuild: bool = False) -> pd.DataFrame:
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_fp = SCAN_CACHE_DIR / f"{UNIVERSE}.parquet"

    if cache_fp.exists() and not rebuild:
        try:
            df_cached = normalize_scans(pd.read_parquet(cache_fp))
            if not df_cached.empty:
                return df_cached
        except Exception:
            pass

    if LEGACY_SCAN_CACHE.exists() and not rebuild:
        df_legacy = normalize_scans(pd.read_parquet(LEGACY_SCAN_CACHE))
        if not df_legacy.empty:
            df_legacy.to_parquet(cache_fp, index=False)
            return df_legacy

    inline_cfg = InlineScannerConfig(
        raw_data_path=BASE_DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
        eligibility_mode=ELIGIBILITY_V1_BASELINE,
    )

    df = build_scans_inline(
        universes=[UNIVERSE],
        start_date=START,
        end_date=END,
        freq=SCAN_FREQ,
        cfg=inline_cfg,
        print_every=12,
    )
    if df.empty:
        raise RuntimeError("Scanner output is empty for Sweden.")

    out = normalize_scans(df)
    out.to_parquet(cache_fp, index=False)
    return out


def scanner_reject_diagnostics(scans: pd.DataFrame) -> pd.DataFrame:
    d = scans.copy()
    d["scan_date"] = pd.to_datetime(d["scan_date"]).dt.normalize()
    d = d[(d["scan_date"] >= pd.Timestamp(START)) & (d["scan_date"] <= pd.Timestamp(END))].copy()
    if d.empty:
        return pd.DataFrame()

    reject_cols = [
        "reject_corr_insufficient_count",
        "reject_adf_invalid_count",
        "reject_eg_invalid_count",
        "reject_half_life_too_high_count",
        "reject_beta_instability_count",
        "reject_insufficient_data_count",
        "reject_technical_exception_count",
    ]
    payload: dict[str, Any] = {
        "scope": "scanner_baseline",
        "universe": UNIVERSE,
        "scan_dates": int(d["scan_date"].nunique()),
        "rows": int(len(d)),
        "n_eligible": int((d["eligibility"] == "ELIGIBLE").sum()),
        "n_watch": int((d["eligibility"] == "WATCH").sum()),
        "n_out": int((d["eligibility"] == "OUT").sum()),
    }
    for c in reject_cols:
        if c in d.columns:
            payload[c] = int(pd.to_numeric(d[c], errors="coerce").fillna(0.0).sum())

    return pd.DataFrame([payload])


def build_strategy_params(
    *,
    entry_mode: str,
    z_entry: float,
    z_window: int,
    max_holding_days: int,
) -> StrategyParams:
    z_exit, z_stop = linked_thresholds(z_entry)
    return StrategyParams(
        z_entry=float(z_entry),
        z_exit=float(z_exit),
        z_stop=float(z_stop),
        z_window=int(z_window),
        beta_mode=DEFAULT_BETA_MODE,
        fees=DEFAULT_FEES,
        top_n_candidates=DEFAULT_TOP_N,
        max_positions=DEFAULT_MAX_POSITIONS,
        max_holding_days=int(max_holding_days),
        signal_space=DEFAULT_SIGNAL_SPACE,
        selection_mode="legacy",
        selection_score_variant="baseline",
        eligibility_labels=("ELIGIBLE",),
        entry_mode=str(entry_mode),
        # Mode 3 formula:
        # z_entry_eff(t)=z_entry*clip(sigma_recent/sigma_hist,1,entry_vol_scale_cap)
        # sigma_recent=std(Delta spread over 20 bars), sigma_hist=std(Delta spread over 120 bars)
        entry_vol_recent_window=20,
        entry_vol_hist_window=120,
        entry_vol_scale_cap=1.5,
        # Mode 4: stricter than scanner baseline (scanner max=100).
        entry_half_life_max_at_entry=60.0,
        # Mode 5: reject if directed widening speed is too abrupt.
        entry_speed_window=20,
        entry_speed_max_multiple=2.5,
    )


def run_one_segment(
    *,
    scans: pd.DataFrame,
    segment: Segment,
    entry_mode: str,
    z_entry: float,
    z_window: int,
    max_holding_days: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = BatchConfig(
        data_path=BASE_DATA_PATH,
        start_date=segment.start_date,
        end_date=segment.end_date,
    )
    params = build_strategy_params(
        entry_mode=entry_mode,
        z_entry=z_entry,
        z_window=z_window,
        max_holding_days=max_holding_days,
    )

    t0 = time.time()
    res = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=[UNIVERSE],
        scans=scans,
    )
    runtime_s = time.time() - t0

    base = {
        "segment": segment.name,
        "entry_mode": entry_mode,
        "z_entry": float(z_entry),
        "z_window": int(z_window),
        "max_holding_days": int(max_holding_days),
        "runtime_s": round(float(runtime_s), 3),
    }
    if not res:
        row = {
            **base,
            "ok": False,
            "sharpe": np.nan,
            "annualized_return": np.nan,
            "max_drawdown": np.nan,
            "hit_ratio": np.nan,
            "avg_holding_days": np.nan,
            "nb_trades": 0,
            "nb_entries_filtered_by_entry_mode": 0,
            "entry_candidates_considered": 0,
            "entry_filter_rate": np.nan,
            "entry_filter_reasons": "",
        }
        return row, row.copy()

    stats = dict(res.get("stats", {}))
    trades = res.get("trades", pd.DataFrame())
    closed = trades[trades["exit_datetime"].notna()].copy() if isinstance(trades, pd.DataFrame) and not trades.empty else pd.DataFrame()

    diagnostics = res.get("diagnostics", pd.DataFrame())
    if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
        entry_candidates = int(pd.to_numeric(diagnostics.get("entry_candidates_considered"), errors="coerce").fillna(0.0).sum())
        entry_filtered = int(pd.to_numeric(diagnostics.get("entry_filtered_by_mode"), errors="coerce").fillna(0.0).sum())
    else:
        entry_candidates = 0
        entry_filtered = 0
    entry_filter_rate = (entry_filtered / entry_candidates) if entry_candidates > 0 else np.nan

    row = {
        **base,
        "ok": True,
        "sharpe": _safe_float(stats.get("Sharpe")),
        "annualized_return": _safe_float(stats.get("CAGR")),
        "max_drawdown": _safe_float(stats.get("Max Drawdown")),
        "hit_ratio": float((closed["trade_return"] > 0).mean()) if len(closed) > 0 else np.nan,
        "avg_holding_days": _safe_float(pd.to_numeric(closed.get("duration_days"), errors="coerce").mean()) if len(closed) > 0 else np.nan,
        "nb_trades": int(stats.get("Nb Trades", len(trades) if isinstance(trades, pd.DataFrame) else 0)),
        "nb_entries_filtered_by_entry_mode": int(stats.get("Entries filtered by entry mode", entry_filtered)),
        "entry_candidates_considered": int(entry_candidates),
        "entry_filter_rate": _safe_float(entry_filter_rate),
        "entry_filter_reasons": str(stats.get("Entry filter reasons", "")),
    }

    diag = {
        **base,
        "scope": "entry_mode_filter",
        "entry_candidates_considered": int(entry_candidates),
        "entry_filtered_by_mode": int(entry_filtered),
        "entry_filter_rate": _safe_float(entry_filter_rate),
        "entry_filter_reasons": str(stats.get("Entry filter reasons", "")),
        "nb_trades": int(row["nb_trades"]),
    }
    return row, diag


def run_campaign(scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combos = list(itertools.product(ENTRY_MODES, Z_ENTRY_GRID, Z_WINDOW_GRID, MAX_HOLD_GRID))
    total = len(combos)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    t0 = time.time()

    for i, (entry_mode, z_entry, z_window, max_hold) in enumerate(combos, start=1):
        per_segment: dict[str, dict[str, Any]] = {}
        print(
            f"[{i:02d}/{total:02d}] mode={entry_mode} z_entry={z_entry} z_window={z_window} max_hold={max_hold}"
        )

        for seg in SEGMENTS:
            seg_row, seg_diag = run_one_segment(
                scans=scans,
                segment=seg,
                entry_mode=entry_mode,
                z_entry=z_entry,
                z_window=z_window,
                max_holding_days=max_hold,
            )
            per_segment[seg.name] = seg_row
            diag_rows.append(seg_diag)

        full = per_segment["FULL"]
        is_row = per_segment["IS"]
        oos = per_segment["OOS"]

        is_sharpe = _safe_float(is_row.get("sharpe"))
        oos_sharpe = _safe_float(oos.get("sharpe"))
        gap = is_sharpe - oos_sharpe if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe) else np.nan

        rows.append(
            {
                "entry_mode": entry_mode,
                "z_entry": float(z_entry),
                "z_window": int(z_window),
                "max_holding_days": int(max_hold),
                "full_sample_sharpe": _safe_float(full.get("sharpe")),
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "annualized_return": _safe_float(full.get("annualized_return")),
                "max_drawdown": _safe_float(full.get("max_drawdown")),
                "hit_ratio": _safe_float(full.get("hit_ratio")),
                "avg_holding_days": _safe_float(full.get("avg_holding_days")),
                "nb_trades": int(full.get("nb_trades", 0)),
                "nb_entries_filtered_by_entry_mode": int(full.get("nb_entries_filtered_by_entry_mode", 0)),
                "entry_candidates_considered": int(full.get("entry_candidates_considered", 0)),
                "entry_filter_rate": _safe_float(full.get("entry_filter_rate")),
                "gap_is_oos": _safe_float(gap),
                "gap_is_oos_abs": _safe_float(abs(gap) if np.isfinite(gap) else np.nan),
                "entry_filter_reasons": str(full.get("entry_filter_reasons", "")),
                "full_ok": bool(full.get("ok", False)),
                "is_ok": bool(is_row.get("ok", False)),
                "oos_ok": bool(oos.get("ok", False)),
            }
        )

        elapsed = max(1e-6, time.time() - t0)
        rate = i / elapsed
        eta = (total - i) / rate if rate > 0 else np.nan
        print(f"  progress={i/total:.1%} rate={rate:.2f} cfg/s eta={eta/60:.1f} min")

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def best_by_entry_mode(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    d = runs.copy()
    for c in ("oos_sharpe", "gap_is_oos_abs", "full_sample_sharpe", "nb_trades"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(
        ["entry_mode", "oos_sharpe", "gap_is_oos_abs", "full_sample_sharpe", "nb_trades"],
        ascending=[True, False, True, False, False],
    )
    return d.drop_duplicates(subset=["entry_mode"], keep="first").reset_index(drop=True)


def aggregate_by_entry_mode(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    d = runs.copy()
    return (
        d.groupby("entry_mode", as_index=False)
        .agg(
            n_runs=("entry_mode", "size"),
            median_full_sample_sharpe=("full_sample_sharpe", "median"),
            median_is_sharpe=("is_sharpe", "median"),
            median_oos_sharpe=("oos_sharpe", "median"),
            best_oos_sharpe=("oos_sharpe", "max"),
            median_gap_is_oos_abs=("gap_is_oos_abs", "median"),
            median_nb_trades=("nb_trades", "median"),
            median_hit_ratio=("hit_ratio", "median"),
            median_avg_holding_days=("avg_holding_days", "median"),
            median_entry_filter_rate=("entry_filter_rate", "median"),
            median_nb_entries_filtered=("nb_entries_filtered_by_entry_mode", "median"),
        )
        .sort_values(["best_oos_sharpe", "median_oos_sharpe"], ascending=False)
        .reset_index(drop=True)
    )


def write_summary(
    *,
    out_dir: Path,
    runs: pd.DataFrame,
    best: pd.DataFrame,
    agg: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("Sweden Entry Campaign (baseline eligibility fixed)")
    lines.append("")
    lines.append(f"Universe: {UNIVERSE}")
    lines.append(f"Period: {START} -> {END}")
    lines.append(f"IS: {IS_START} -> {IS_END}")
    lines.append(f"OOS: {OOS_START} -> {OOS_END}")
    lines.append(f"Runs: {len(runs)}")
    lines.append("")
    lines.append("Mode 3 formula (entry_vol_adjusted):")
    lines.append("z_entry_eff(t) = z_entry * clip(sigma_recent/sigma_hist, 1, 1.5)")
    lines.append("sigma_recent = std(Delta spread over last 20 bars)")
    lines.append("sigma_hist = std(Delta spread over last 120 bars)")
    lines.append("")
    lines.append("Mode 4 rule (entry_half_life_aware):")
    lines.append("Entry allowed only if scanner 6m_half_life <= 60 at the latest scan snapshot.")
    lines.append("")

    baseline_best = best[best["entry_mode"] == "baseline_entry"].copy()
    baseline_oos = _safe_float(baseline_best["oos_sharpe"].iloc[0]) if not baseline_best.empty else np.nan

    lines.append("1. Est-ce que certains modes d'entree ameliorent reellement l'OOS sur Sweden ?")
    if best.empty or not np.isfinite(baseline_oos):
        lines.append("- Resultat: impossible a conclure (baseline manquante ou OOS non calcule).")
    else:
        improved = best[pd.to_numeric(best["oos_sharpe"], errors="coerce") > baseline_oos].copy()
        if improved.empty:
            lines.append(f"- Non: aucun mode n'a depasse la baseline OOS ({_fmt(baseline_oos, 2)}).")
        else:
            lines.append(f"- Oui: baseline OOS={_fmt(baseline_oos, 2)}; modes au-dessus:")
            for r in improved.itertuples(index=False):
                lines.append(f"  - {r.entry_mode}: OOS={_fmt(r.oos_sharpe, 2)} (gap_abs={_fmt(r.gap_is_oos_abs, 2)})")

    lines.append("")
    lines.append("2. Est-ce que le gain vient d'une meilleure qualite de trades ou juste d'une baisse du nombre de trades ?")
    if best.empty or baseline_best.empty:
        lines.append("- Analyse indisponible (best table vide).")
    else:
        comp = best[best["entry_mode"] != "baseline_entry"].copy()
        if comp.empty:
            lines.append("- Aucun mode alternatif valide pour comparer.")
        else:
            comp = comp.sort_values(["oos_sharpe", "gap_is_oos_abs"], ascending=[False, True]).reset_index(drop=True)
            top_alt = comp.iloc[0]
            base = baseline_best.iloc[0]
            trades_delta = _safe_float(top_alt["nb_trades"]) - _safe_float(base["nb_trades"])
            hit_delta = _safe_float(top_alt["hit_ratio"]) - _safe_float(base["hit_ratio"])
            if np.isfinite(trades_delta) and np.isfinite(hit_delta):
                trade_read = "baisse des trades" if trades_delta < 0 else "hausse des trades"
                if hit_delta > 0.02 and trades_delta > -0.30 * max(1.0, _safe_float(base["nb_trades"])):
                    verdict = "plutot qualite de trades"
                elif trades_delta < 0 and hit_delta <= 0.02:
                    verdict = "plutot reduction du nombre de trades"
                else:
                    verdict = "mixte (qualite + quantite)"
                lines.append(
                    f"- Top alt={top_alt['entry_mode']} vs baseline: delta_trades={_fmt(trades_delta, 1)}, "
                    f"delta_hit_ratio={_fmt(hit_delta, 3)} -> {verdict} ({trade_read})."
                )
            else:
                lines.append("- Donnees insuffisantes pour attribuer la source du gain.")

    lines.append("")
    lines.append("3. Quel mode semble le plus robuste et le plus simple a conserver pour la suite ?")
    if best.empty:
        lines.append("- Aucun candidat robuste identifiable (best table vide).")
    else:
        complexity_rank = {
            "baseline_entry": 0,
            "entry_with_reversal_confirmation": 1,
            "entry_vol_adjusted": 2,
            "entry_half_life_aware": 2,
            "entry_with_spread_speed_filter": 2,
        }
        cand = best.copy()
        cand["oos_sharpe"] = pd.to_numeric(cand["oos_sharpe"], errors="coerce")
        cand["gap_is_oos_abs"] = pd.to_numeric(cand["gap_is_oos_abs"], errors="coerce")
        cand["complexity_rank"] = cand["entry_mode"].map(complexity_rank).fillna(9).astype(int)
        cand["robust_score"] = cand["oos_sharpe"] - 0.5 * cand["gap_is_oos_abs"]
        cand = cand.sort_values(["robust_score", "complexity_rank", "nb_trades"], ascending=[False, True, False])
        rec = cand.iloc[0]
        lines.append(
            f"- Recommande: {rec['entry_mode']} "
            f"(OOS={_fmt(rec['oos_sharpe'], 2)}, gap_abs={_fmt(rec['gap_is_oos_abs'], 2)}, "
            f"trades={int(_safe_float(rec['nb_trades'])) if np.isfinite(_safe_float(rec['nb_trades'])) else 0})."
        )

    lines.append("")
    if not agg.empty:
        lines.append("Aggregate by entry mode:")
        for r in agg.itertuples(index=False):
            lines.append(
                f"- {r.entry_mode}: best_oos={_fmt(r.best_oos_sharpe, 2)} "
                f"med_oos={_fmt(r.median_oos_sharpe, 2)} med_gap={_fmt(r.median_gap_is_oos_abs, 2)} "
                f"med_trades={_fmt(r.median_nb_trades, 1)}"
            )

    (out_dir / "summary_entry_campaign.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading baseline Sweden scans...")
    scans = load_or_build_sweden_scans(rebuild=False)
    print(
        f"Scans loaded: rows={len(scans):,} dates={scans['scan_date'].nunique()} "
        f"range={scans['scan_date'].min().date()}->{scans['scan_date'].max().date()}"
    )

    runs, entry_diag = run_campaign(scans=scans)
    runs = runs.sort_values(["entry_mode", "z_entry", "z_window", "max_holding_days"]).reset_index(drop=True)
    runs.to_csv(OUT_DIR / "runs_entry_campaign.csv", index=False)

    best = best_by_entry_mode(runs)
    best.to_csv(OUT_DIR / "best_by_entry_mode.csv", index=False)

    agg = aggregate_by_entry_mode(runs)
    agg.to_csv(OUT_DIR / "entry_mode_aggregate.csv", index=False)

    scanner_diag = scanner_reject_diagnostics(scans)
    reject_or_filter_diag = pd.concat([entry_diag, scanner_diag], ignore_index=True, sort=False)
    reject_or_filter_diag.to_csv(OUT_DIR / "reject_or_filter_diagnostics.csv", index=False)

    write_summary(out_dir=OUT_DIR, runs=runs, best=best, agg=agg)

    print("Saved outputs in:", OUT_DIR)
    print(" - runs_entry_campaign.csv")
    print(" - best_by_entry_mode.csv")
    print(" - entry_mode_aggregate.csv")
    print(" - reject_or_filter_diagnostics.csv")
    print(" - summary_entry_campaign.txt")


if __name__ == "__main__":
    main()

