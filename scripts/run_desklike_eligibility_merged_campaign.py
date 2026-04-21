from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.global_loop import run_global_ranking_daily_portfolio
from object.class_file import BatchConfig, StrategyParams
from utils.inline_scanner import InlineScannerConfig, build_scans_inline
from utils.scanner import (
    ELIGIBILITY_MODES,
    ELIGIBILITY_V1_BASELINE,
    ELIGIBILITY_V2_HARD_12M_PLUS_SHORT,
    ELIGIBILITY_V3_HARD_12M_6M,
    ELIGIBILITY_V4_CONTINUOUS_DISTANCE_SCORE,
    ELIGIBILITY_V5_STABILITY_FOCUSED,
)
from config.params import SCANNER_THRESHOLDS, SCANNER_WEIGHTS


START = "2015-12-31"
END = "2025-12-31"
SCAN_FREQ = "ME"

BASE_UNIVERSES = ["france", "sweden", "italy", "germany"]

MERGED_CASES = [
    ("france_sweden", ["france", "sweden"]),
    ("france_germany", ["france", "germany"]),
    ("france_italy", ["france", "italy"]),
    ("sweden_germany", ["sweden", "germany"]),
    ("france_sweden_germany", ["france", "sweden", "germany"]),
]

Z_ENTRY_GRID = [1.2, 1.5, 1.8]
Z_WINDOW_GRID = [40, 60, 80, 100]
MAX_HOLD_GRID = [15, 20, 25, 30]
PASS2_PARAM_GRID = [
    (1.2, 40, 15),
    (1.2, 80, 25),
    (1.5, 60, 20),
    (1.5, 100, 30),
    (1.8, 40, 25),
    (1.8, 100, 30),
]

ROBUST_FULL_SHARPE_MIN = 1.0
ROBUST_OOS_SHARPE_MIN = 0.8
ROBUST_GAP_MAX = 0.35

DEFAULT_TOP_N = 20
DEFAULT_MAX_POSITIONS = 5
DEFAULT_FEES = 0.0002
DEFAULT_BETA_MODE = "static"
DEFAULT_SIGNAL_PCA_WINDOW = 252
DEFAULT_SIGNAL_PCA_COMPONENTS = 3
DEFAULT_SIGNAL_PCA_MIN_ASSETS = 10
BASELINE_BETA_STD_PENALTY_FILL = 0.35
BASELINE_MISSING_PENALTY = 0.25
PASS3_TOP_MODE_RANK = 2
PASS3_TOP_PARAMS_PER_MODE_RANK = 2

BASE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"
OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "desklike_eligibility_merged_campaign_2015_2025"
SCAN_CACHE_DIR = OUT_DIR / "scans_v2_base"

SPLITS = [
    {
        "split_id": "split1",
        "is_start": "2016-01-01",
        "is_end": "2019-12-31",
        "oos_start": "2020-01-01",
        "oos_end": "2021-12-31",
    },
    {
        "split_id": "split2",
        "is_start": "2016-01-01",
        "is_end": "2021-12-31",
        "oos_start": "2022-01-01",
        "oos_end": "2023-12-31",
    },
    {
        "split_id": "split3",
        "is_start": "2016-01-01",
        "is_end": "2023-12-31",
        "oos_start": "2024-01-01",
        "oos_end": "2025-12-31",
    },
]


@dataclass(frozen=True)
class UniverseCase:
    name: str
    members: tuple[str, ...]
    kind: str  # single | merged


@dataclass(frozen=True)
class SegmentConfig:
    segment: str
    split_id: str
    split_kind: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class RankingConfig:
    name: str
    selection_mode: str
    selection_score_variant: str
    selection_winsor_quantile: float = 0.0
    selection_stability_penalty: float = 0.0


def _safe_nanquantile(x: np.ndarray, q: float, default: float = np.nan) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanquantile(arr, q))


def linked_thresholds(z_entry: float) -> tuple[float, float]:
    return (round(float(z_entry) / 3.0, 4), round(2.0 * float(z_entry), 4))


def build_segments(include_splits: bool = True) -> list[SegmentConfig]:
    segs = [
        SegmentConfig(
            segment="FULL",
            split_id="full",
            split_kind="FULL",
            start_date=START,
            end_date=END,
        )
    ]
    if not include_splits:
        return segs
    for s in SPLITS:
        sid = str(s["split_id"])
        segs.append(
            SegmentConfig(
                segment=f"{sid.upper()}_IS",
                split_id=sid,
                split_kind="IS",
                start_date=str(s["is_start"]),
                end_date=str(s["is_end"]),
            )
        )
        segs.append(
            SegmentConfig(
                segment=f"{sid.upper()}_OOS",
                split_id=sid,
                split_kind="OOS",
                start_date=str(s["oos_start"]),
                end_date=str(s["oos_end"]),
            )
        )
    return segs


def build_cases() -> list[UniverseCase]:
    out: list[UniverseCase] = []
    for u in BASE_UNIVERSES:
        out.append(UniverseCase(name=u, members=(u,), kind="single"))
    for name, members in MERGED_CASES:
        out.append(UniverseCase(name=name, members=tuple(members), kind="merged"))
    return out


def build_rankings() -> list[RankingConfig]:
    return [
        RankingConfig(
            name="ranking_current_legacy",
            selection_mode="legacy",
            selection_score_variant="baseline",
        ),
        RankingConfig(
            name="ranking_composite_adjusted",
            selection_mode="composite_quality",
            selection_score_variant="baseline",
        ),
        RankingConfig(
            name="ranking_percentile",
            selection_mode="composite_quality",
            selection_score_variant="rank_percentile",
        ),
        RankingConfig(
            name="ranking_robust_zscore",
            selection_mode="composite_quality",
            selection_score_variant="robust_zscore",
            selection_winsor_quantile=0.05,
        ),
        RankingConfig(
            name="ranking_stability_penalty",
            selection_mode="composite_quality",
            selection_score_variant="rank_stability_penalty",
            selection_stability_penalty=0.35,
        ),
        RankingConfig(
            name="ranking_missing_safe",
            selection_mode="composite_quality",
            selection_score_variant="missing_safe",
        ),
    ]


def normalize_scan_df(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    d = df.copy()
    d["scan_date"] = pd.to_datetime(d["scan_date"]).dt.normalize()
    d["asset_1"] = d["asset_1"].astype(str).str.upper()
    d["asset_2"] = d["asset_2"].astype(str).str.upper()
    d["universe"] = str(universe).strip().lower()
    d["eligibility"] = d.get("eligibility", "OUT").astype(str).str.upper()
    d["eligibility_score"] = pd.to_numeric(d.get("eligibility_score"), errors="coerce")
    d["n_valid_windows"] = pd.to_numeric(d.get("n_valid_windows"), errors="coerce").fillna(0.0)

    for c in (
        "reject_corr_insufficient_count",
        "reject_adf_invalid_count",
        "reject_eg_invalid_count",
        "reject_half_life_too_high_count",
        "reject_beta_instability_count",
        "reject_insufficient_data_count",
        "reject_technical_exception_count",
        "missing_metric_count",
        "window_exception_count",
    ):
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).astype(int)

    d = d.sort_values(
        ["scan_date", "asset_1", "asset_2", "universe"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    d = d.drop_duplicates(subset=["scan_date", "asset_1", "asset_2", "universe"], keep="first").reset_index(drop=True)
    return d


def load_or_build_scan(universe: str, *, rebuild: bool = False) -> pd.DataFrame:
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = SCAN_CACHE_DIR / f"{universe}.parquet"

    if fp.exists() and not rebuild:
        try:
            cached = pd.read_parquet(fp)
            cached = normalize_scan_df(cached, universe)
            needed = {"3m_window_valid", "6m_window_valid", "12m_window_valid", "validity_distance_score"}
            if needed.issubset(set(cached.columns)):
                return cached
        except Exception:
            pass

    cfg = InlineScannerConfig(
        raw_data_path=BASE_DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
        eligibility_mode=ELIGIBILITY_V2_HARD_12M_PLUS_SHORT,
    )

    df = build_scans_inline(
        universes=[universe],
        start_date=START,
        end_date=END,
        freq=SCAN_FREQ,
        cfg=cfg,
    )
    if df.empty:
        return pd.DataFrame()

    out = normalize_scan_df(df, universe)
    out.to_parquet(fp, index=False)
    return out


def _compute_mode_score(
    d: pd.DataFrame,
    mode: str,
) -> tuple[pd.Series, pd.Series]:
    n_windows = 3.0
    v3 = d.get("3m_window_valid", pd.Series(False, index=d.index)).astype(bool)
    v6 = d.get("6m_window_valid", pd.Series(False, index=d.index)).astype(bool)
    v12 = d.get("12m_window_valid", pd.Series(False, index=d.index)).astype(bool)

    n_valid = pd.to_numeric(d.get("n_valid_windows"), errors="coerce").fillna(0.0)
    valid_ratio = (n_valid / n_windows).clip(lower=0.0, upper=1.0)
    validity_distance = pd.to_numeric(d.get("validity_distance_score"), errors="coerce").fillna(-1.0)
    stability = pd.to_numeric(d.get("stability_score"), errors="coerce").fillna(-1.0)
    beta_std = pd.to_numeric(d.get("beta_std"), errors="coerce")

    if mode == ELIGIBILITY_V1_BASELINE:
        corr_12m = pd.to_numeric(d.get("12m_corr"), errors="coerce").abs()
        hl_6m = pd.to_numeric(d.get("6m_half_life"), errors="coerce")
        corr_for_score = corr_12m.fillna(0.0)
        hl_for_score = hl_6m.fillna(float(SCANNER_THRESHOLDS["half_life_max"]))
        beta_for_score = beta_std.fillna(BASELINE_BETA_STD_PENALTY_FILL)
        missing_count = (
            corr_12m.isna().astype(float)
            + hl_6m.isna().astype(float)
            + beta_std.isna().astype(float)
        )
        score = (
            SCANNER_WEIGHTS["n_valid"] * n_valid
            + SCANNER_WEIGHTS["corr_12m"] * corr_for_score
            + SCANNER_WEIGHTS["half_life_6m"] * hl_for_score
            + SCANNER_WEIGHTS["beta_stability"] * beta_for_score
            - BASELINE_MISSING_PENALTY * missing_count
        )
        label = np.where(n_valid >= 2.0, "ELIGIBLE", np.where(n_valid == 1.0, "WATCH", "OUT"))
        return pd.Series(label, index=d.index), pd.to_numeric(score, errors="coerce").fillna(-np.inf)

    if mode == ELIGIBILITY_V2_HARD_12M_PLUS_SHORT:
        short_ok = v3 | v6
        score = 0.55 * valid_ratio + 0.30 * validity_distance + 0.15 * stability
        label = np.where(v12 & short_ok, "ELIGIBLE", np.where(v12 | short_ok, "WATCH", "OUT"))
        return pd.Series(label, index=d.index), pd.Series(score, index=d.index, dtype=float)

    if mode == ELIGIBILITY_V3_HARD_12M_6M:
        score = 0.60 * valid_ratio + 0.25 * validity_distance + 0.15 * stability
        label = np.where(v12 & v6, "ELIGIBLE", np.where(v12 | v6, "WATCH", "OUT"))
        return pd.Series(label, index=d.index), pd.Series(score, index=d.index, dtype=float)

    if mode == ELIGIBILITY_V5_STABILITY_FOCUSED:
        beta_ok = beta_std.notna() & (beta_std <= 0.35)
        score = 0.45 * valid_ratio + 0.20 * validity_distance + 0.35 * stability
        eligible = v12 & v6 & beta_ok & (stability >= 0.0)
        watch = (~eligible) & ((v12 & v6) | (v12 & beta_ok))
        label = np.where(eligible, "ELIGIBLE", np.where(watch, "WATCH", "OUT"))
        return pd.Series(label, index=d.index), pd.Series(score, index=d.index, dtype=float)

    # V4 dynamic score by date+country bucket.
    score = 0.50 * validity_distance + 0.25 * valid_ratio + 0.25 * stability
    score = pd.Series(score, index=d.index, dtype=float)
    if d.empty:
        return pd.Series(dtype=str), score

    if "scan_date" in d.columns and "universe" in d.columns:
        thr = score.groupby([d["scan_date"], d["universe"]], sort=False).transform(
            lambda s: _safe_nanquantile(s.to_numpy(dtype=float), 0.65, default=np.nan)
        )
    else:
        q = _safe_nanquantile(score.to_numpy(dtype=float), 0.65, default=np.nan)
        thr = pd.Series(q, index=d.index, dtype=float)
    thr = pd.to_numeric(thr, errors="coerce")
    fallback_thr = _safe_nanquantile(score.to_numpy(dtype=float), 0.65, default=np.nan)
    thr = thr.fillna(fallback_thr)

    eligible = v12 & (score >= thr)
    watch = (~eligible) & (score >= (thr - 0.10))
    label = np.where(eligible, "ELIGIBLE", np.where(watch, "WATCH", "OUT"))
    return pd.Series(label, index=d.index), score


def apply_eligibility_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in set(ELIGIBILITY_MODES):
        raise ValueError(f"Unsupported mode={mode}")

    d = df.copy()
    label, score = _compute_mode_score(d, mode=mode)
    d["eligibility_mode"] = mode
    d["gating_label"] = label.astype(str)
    d["gating_score"] = pd.to_numeric(score, errors="coerce")
    d["eligibility"] = d["gating_label"]
    d["eligibility_score"] = d["gating_score"]
    d["ranking_score"] = d["gating_score"]
    return d


def build_asset_to_universes() -> dict[str, set[str]]:
    reg = pd.read_csv(ASSET_REGISTRY_PATH, usecols=["category_id", "asset"])
    reg["category_id"] = reg["category_id"].astype(str).str.lower()
    reg["asset"] = reg["asset"].astype(str).str.upper()
    out: dict[str, set[str]] = {}
    for r in reg.itertuples(index=False):
        out.setdefault(str(r.asset), set()).add(str(r.category_id))
    return out


def ensure_no_cross_country_pairs(
    context_name: str,
    context_members: tuple[str, ...],
    scan_df: pd.DataFrame,
    asset_to_universes: dict[str, set[str]],
) -> None:
    if scan_df.empty:
        return
    d = scan_df.copy()
    d["universe"] = d["universe"].astype(str).str.lower()
    allowed = set(context_members)

    bad_universe = d[~d["universe"].isin(allowed)]
    if not bad_universe.empty:
        raise RuntimeError(f"[{context_name}] Found rows outside allowed universes: {len(bad_universe)}")

    def _asset_ok(asset: str, univ: str) -> bool:
        return univ in asset_to_universes.get(str(asset).upper(), set())

    m1 = d.apply(lambda r: _asset_ok(str(r["asset_1"]), str(r["universe"])), axis=1)
    m2 = d.apply(lambda r: _asset_ok(str(r["asset_2"]), str(r["universe"])), axis=1)
    bad = d[~(m1 & m2)]
    if not bad.empty:
        raise RuntimeError(f"[{context_name}] Found {len(bad)} rows with cross-country asset assignment.")


def build_params(
    ranking: RankingConfig,
    z_entry: float,
    z_window: int,
    max_hold: int,
    n_universes_in_context: int,
) -> StrategyParams:
    z_exit, z_stop = linked_thresholds(z_entry)
    per_universe_cap = 0
    if int(n_universes_in_context) > 1:
        per_universe_cap = int(np.ceil(DEFAULT_TOP_N / float(n_universes_in_context)))

    return StrategyParams(
        z_entry=float(z_entry),
        z_exit=float(z_exit),
        z_stop=float(z_stop),
        z_window=int(z_window),
        beta_mode=DEFAULT_BETA_MODE,
        fees=DEFAULT_FEES,
        top_n_candidates=DEFAULT_TOP_N,
        max_positions=DEFAULT_MAX_POSITIONS,
        max_holding_days=int(max_hold),
        signal_space="idio_pca",
        selection_mode=ranking.selection_mode,
        selection_score_variant=ranking.selection_score_variant,
        selection_winsor_quantile=float(ranking.selection_winsor_quantile),
        selection_stability_penalty=float(ranking.selection_stability_penalty),
        max_pairs_per_asset=2,
        max_pairs_per_universe=int(per_universe_cap),
        pair_return_cap=0.05,
        trade_return_isolated_cap=0.20,
        pca_signal_window=DEFAULT_SIGNAL_PCA_WINDOW,
        pca_signal_components=DEFAULT_SIGNAL_PCA_COMPONENTS,
        pca_signal_min_assets=DEFAULT_SIGNAL_PCA_MIN_ASSETS,
    )


def run_one(
    case: UniverseCase,
    scans_df: pd.DataFrame,
    eligibility_mode: str,
    ranking: RankingConfig,
    z_entry: float,
    z_window: int,
    max_hold: int,
    segment: SegmentConfig,
    pass_stage: str,
) -> dict[str, Any]:
    cfg = BatchConfig(
        data_path=BASE_DATA_PATH,
        start_date=segment.start_date,
        end_date=segment.end_date,
    )
    params = build_params(
        ranking=ranking,
        z_entry=z_entry,
        z_window=z_window,
        max_hold=max_hold,
        n_universes_in_context=len(case.members),
    )
    t0 = time.time()
    res = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=list(case.members),
        scans=scans_df,
    )
    runtime_s = time.time() - t0

    row: dict[str, Any] = {
        "pass_stage": pass_stage,
        "context_name": case.name,
        "context_kind": case.kind,
        "context_members": ",".join(case.members),
        "segment": segment.segment,
        "split_id": segment.split_id,
        "split_kind": segment.split_kind,
        "start_date": segment.start_date,
        "end_date": segment.end_date,
        "eligibility_mode": eligibility_mode,
        "ranking_name": ranking.name,
        "selection_mode": ranking.selection_mode,
        "selection_score_variant": ranking.selection_score_variant,
        "z_entry": float(z_entry),
        "z_exit": float(params.z_exit),
        "z_stop": float(params.z_stop),
        "z_window": int(z_window),
        "max_hold": int(max_hold),
        "max_pairs_per_asset": int(params.max_pairs_per_asset),
        "max_pairs_per_universe": int(params.max_pairs_per_universe),
        "runtime_s": round(float(runtime_s), 3),
    }

    if not res:
        row["ok"] = False
        return row

    st = dict(res.get("stats", {}))
    tr = res.get("trades", pd.DataFrame())
    tr_closed = tr[tr["exit_datetime"].notna()].copy() if isinstance(tr, pd.DataFrame) and not tr.empty else pd.DataFrame()

    row.update(
        {
            "ok": True,
            "final_equity": st.get("Final Equity"),
            "sharpe": st.get("Sharpe"),
            "cagr": st.get("CAGR"),
            "max_drawdown": st.get("Max Drawdown"),
            "nb_trades": st.get("Nb Trades"),
            "closed_trades": int(len(tr_closed)),
            "hit_ratio": float((tr_closed["trade_return"] > 0).mean()) if len(tr_closed) > 0 else np.nan,
            "avg_trade_return": float(tr_closed["trade_return"].mean()) if len(tr_closed) > 0 else np.nan,
            "anomaly_flag": bool(st.get("Anomaly flag", False)),
            "anomaly_reasons": st.get("Anomaly reasons", ""),
        }
    )
    return row


def build_context_scans(
    scans_by_mode_universe: dict[str, dict[str, pd.DataFrame]],
    cases: list[UniverseCase],
) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for mode, by_univ in scans_by_mode_universe.items():
        out[mode] = {}
        for case in cases:
            frames = [by_univ[u] for u in case.members if u in by_univ]
            if not frames:
                out[mode][case.name] = pd.DataFrame()
                continue
            d = pd.concat(frames, ignore_index=True)
            d = d.sort_values(["scan_date", "universe", "asset_1", "asset_2"]).reset_index(drop=True)
            out[mode][case.name] = d
    return out


def run_matrix(
    *,
    pass_stage: str,
    cases: list[UniverseCase],
    context_scans: dict[str, dict[str, pd.DataFrame]],
    eligibility_modes: list[str],
    rankings: list[RankingConfig],
    z_entries: list[float],
    z_windows: list[int],
    max_holds: list[int],
    segments: list[SegmentConfig],
    param_grid: list[tuple[float, int, int]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combos = (
        [(float(ze), int(zw), int(mh)) for ze in z_entries for zw in z_windows for mh in max_holds]
        if param_grid is None
        else [(float(ze), int(zw), int(mh)) for (ze, zw, mh) in param_grid]
    )
    total = (
        len(cases)
        * len(eligibility_modes)
        * len(rankings)
        * len(combos)
        * len(segments)
    )
    done = 0
    t0 = time.time()

    for case in cases:
        for mode in eligibility_modes:
            scans_df = context_scans.get(mode, {}).get(case.name, pd.DataFrame())
            if scans_df.empty:
                continue
            for ranking in rankings:
                for (z_entry, z_window, max_hold) in combos:
                    for segment in segments:
                        done += 1
                        row = run_one(
                            case=case,
                            scans_df=scans_df,
                            eligibility_mode=mode,
                            ranking=ranking,
                            z_entry=z_entry,
                            z_window=z_window,
                            max_hold=max_hold,
                            segment=segment,
                            pass_stage=pass_stage,
                        )
                        rows.append(row)
                        if (done % 25 == 0) or (done == total):
                            elapsed = max(1e-6, time.time() - t0)
                            rate = done / elapsed
                            eta = (total - done) / max(rate, 1e-9)
                            print(
                                f"[{pass_stage}] {done}/{total} ({done/total:.1%}) | "
                                f"{rate:.2f} runs/s | ETA {eta/60:.1f} min"
                            )
    return pd.DataFrame(rows)


def select_pass3_configs(pass2_runs: pd.DataFrame) -> pd.DataFrame:
    if pass2_runs.empty:
        return pd.DataFrame()
    d = pass2_runs[(pass2_runs["ok"] == True) & (pass2_runs["segment"] == "FULL")].copy()
    if d.empty:
        return pd.DataFrame()

    by_mode_rank = (
        d.groupby(["eligibility_mode", "ranking_name"], as_index=False)
        .agg(
            median_sharpe=("sharpe", "median"),
            mean_sharpe=("sharpe", "mean"),
            bad_rate=("sharpe", lambda s: float((pd.to_numeric(s, errors="coerce") < 0).mean())),
        )
        .sort_values(["median_sharpe", "mean_sharpe", "bad_rate"], ascending=[False, False, True])
    )
    top_mode_rank = by_mode_rank.head(PASS3_TOP_MODE_RANK)
    if top_mode_rank.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for r in top_mode_rank.itertuples(index=False):
        sub = d[
            (d["eligibility_mode"] == r.eligibility_mode)
            & (d["ranking_name"] == r.ranking_name)
        ].copy()
        if sub.empty:
            continue
        best_params = (
            sub.groupby(["z_entry", "z_window", "max_hold"], as_index=False)
            .agg(
                median_sharpe=("sharpe", "median"),
                mean_sharpe=("sharpe", "mean"),
            )
            .sort_values(["median_sharpe", "mean_sharpe"], ascending=False)
            .head(PASS3_TOP_PARAMS_PER_MODE_RANK)
        )
        best_params["eligibility_mode"] = r.eligibility_mode
        best_params["ranking_name"] = r.ranking_name
        rows.append(best_params)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["eligibility_mode", "ranking_name", "z_entry", "z_window", "max_hold"])
    return out.reset_index(drop=True)


def run_selected_configs(
    *,
    pass_stage: str,
    selected_configs: pd.DataFrame,
    rankings_by_name: dict[str, RankingConfig],
    cases: list[UniverseCase],
    context_scans: dict[str, dict[str, pd.DataFrame]],
    segments: list[SegmentConfig],
) -> pd.DataFrame:
    if selected_configs.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    total = len(selected_configs) * len(cases) * len(segments)
    done = 0
    t0 = time.time()
    for cfg in selected_configs.itertuples(index=False):
        mode = str(cfg.eligibility_mode)
        ranking = rankings_by_name.get(str(cfg.ranking_name))
        if ranking is None:
            continue
        for case in cases:
            scans_df = context_scans.get(mode, {}).get(case.name, pd.DataFrame())
            if scans_df.empty:
                continue
            for segment in segments:
                done += 1
                row = run_one(
                    case=case,
                    scans_df=scans_df,
                    eligibility_mode=mode,
                    ranking=ranking,
                    z_entry=float(cfg.z_entry),
                    z_window=int(cfg.z_window),
                    max_hold=int(cfg.max_hold),
                    segment=segment,
                    pass_stage=pass_stage,
                )
                rows.append(row)
                if (done % 50 == 0) or (done == total):
                    elapsed = max(1e-6, time.time() - t0)
                    rate = done / elapsed
                    eta = (total - done) / max(rate, 1e-9)
                    print(
                        f"[{pass_stage}] {done}/{total} ({done/total:.1%}) | "
                        f"{rate:.2f} runs/s | ETA {eta/60:.1f} min"
                    )
    return pd.DataFrame(rows)


def build_stability_table(runs: pd.DataFrame) -> pd.DataFrame:
    d = runs[(runs["ok"] == True)].copy()
    if d.empty:
        return pd.DataFrame()

    key = ["context_name", "context_kind", "eligibility_mode", "ranking_name", "z_entry", "z_window", "max_hold"]
    rows: list[dict[str, Any]] = []
    for k, g in d.groupby(key, as_index=False):
        row = {
            "context_name": k[0],
            "context_kind": k[1],
            "eligibility_mode": k[2],
            "ranking_name": k[3],
            "z_entry": float(k[4]),
            "z_window": int(k[5]),
            "max_hold": int(k[6]),
        }
        full = g[g["segment"] == "FULL"]
        row["full_sharpe"] = float(full["sharpe"].median()) if not full.empty else np.nan
        row["full_cagr"] = float(full["cagr"].median()) if not full.empty else np.nan
        row["full_maxdd"] = float(full["max_drawdown"].median()) if not full.empty else np.nan
        row["full_trades"] = float(full["nb_trades"].median()) if not full.empty else np.nan
        row["full_anomaly_rate"] = float(full["anomaly_flag"].mean()) if not full.empty else np.nan

        split_pass_flags: list[bool] = []
        oos_vals: list[float] = []
        gap_vals: list[float] = []
        for s in SPLITS:
            sid = str(s["split_id"])
            seg_is = f"{sid.upper()}_IS"
            seg_oos = f"{sid.upper()}_OOS"
            is_sh = float(g.loc[g["segment"] == seg_is, "sharpe"].median()) if (g["segment"] == seg_is).any() else np.nan
            oos_sh = float(g.loc[g["segment"] == seg_oos, "sharpe"].median()) if (g["segment"] == seg_oos).any() else np.nan
            gap = abs(is_sh - oos_sh) if np.isfinite(is_sh) and np.isfinite(oos_sh) else np.nan
            row[f"{sid}_is_sharpe"] = is_sh
            row[f"{sid}_oos_sharpe"] = oos_sh
            row[f"{sid}_gap_abs"] = gap

            pass_split = bool(
                np.isfinite(oos_sh)
                and np.isfinite(gap)
                and (oos_sh >= ROBUST_OOS_SHARPE_MIN)
                and (gap <= ROBUST_GAP_MAX)
            )
            row[f"{sid}_pass"] = pass_split
            split_pass_flags.append(pass_split)
            oos_vals.append(oos_sh)
            gap_vals.append(gap)

        oos_arr = np.asarray(oos_vals, dtype=float)
        gap_arr = np.asarray(gap_vals, dtype=float)
        oos_finite = oos_arr[np.isfinite(oos_arr)]
        gap_finite = gap_arr[np.isfinite(gap_arr)]
        row["oos_min_sharpe"] = float(np.min(oos_finite)) if oos_finite.size > 0 else np.nan
        row["max_gap_abs"] = float(np.max(gap_finite)) if gap_finite.size > 0 else np.nan
        row["split_pass_count"] = int(sum(1 for x in split_pass_flags if x))
        row["full_pass"] = bool(np.isfinite(row["full_sharpe"]) and row["full_sharpe"] >= ROBUST_FULL_SHARPE_MIN)
        row["robust_multi_split"] = bool(row["full_pass"] and all(split_pass_flags))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["context_name", "eligibility_mode", "ranking_name", "full_sharpe"],
        ascending=[True, True, True, False],
    )


def annotate_local_clusters(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability
    out = stability.copy()
    out["cluster_size"] = 0
    out["cluster_median_full_sharpe"] = np.nan

    ze_levels = sorted(set(float(x) for x in out["z_entry"].dropna().unique().tolist()))
    zw_levels = sorted(set(int(x) for x in out["z_window"].dropna().unique().tolist()))
    mh_levels = sorted(set(int(x) for x in out["max_hold"].dropna().unique().tolist()))

    ze_idx = {v: i for i, v in enumerate(ze_levels)}
    zw_idx = {v: i for i, v in enumerate(zw_levels)}
    mh_idx = {v: i for i, v in enumerate(mh_levels)}

    for (ctx, mode, rank), g in out.groupby(["context_name", "eligibility_mode", "ranking_name"]):
        robust = g[g["robust_multi_split"] == True].copy()
        if robust.empty:
            continue
        robust = robust.assign(
            ze_i=robust["z_entry"].map(ze_idx),
            zw_i=robust["z_window"].map(zw_idx),
            mh_i=robust["max_hold"].map(mh_idx),
        )
        for r in robust.itertuples(index=False):
            neigh = robust[
                (robust["ze_i"].sub(int(r.ze_i)).abs() <= 1)
                & (robust["zw_i"].sub(int(r.zw_i)).abs() <= 1)
                & (robust["mh_i"].sub(int(r.mh_i)).abs() <= 1)
            ]
            m = (
                (out["context_name"] == ctx)
                & (out["eligibility_mode"] == mode)
                & (out["ranking_name"] == rank)
                & (out["z_entry"] == float(r.z_entry))
                & (out["z_window"] == int(r.z_window))
                & (out["max_hold"] == int(r.max_hold))
            )
            out.loc[m, "cluster_size"] = int(len(neigh))
            out.loc[m, "cluster_median_full_sharpe"] = float(neigh["full_sharpe"].median()) if not neigh.empty else np.nan

    out["stable_cluster"] = (out["robust_multi_split"] == True) & (pd.to_numeric(out["cluster_size"], errors="coerce") >= 3)
    return out


def build_reject_reasons_summary(scans_by_mode_universe: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, by_u in scans_by_mode_universe.items():
        for u, d in by_u.items():
            if d.empty:
                continue
            n_pairs = int(len(d))
            rows.append(
                {
                    "eligibility_mode": mode,
                    "universe": u,
                    "n_pairs": n_pairs,
                    "n_eligible": int((d["eligibility"] == "ELIGIBLE").sum()),
                    "n_watch": int((d["eligibility"] == "WATCH").sum()),
                    "n_out": int((d["eligibility"] == "OUT").sum()),
                    "reject_corr_insufficient": int(pd.to_numeric(d["reject_corr_insufficient_count"], errors="coerce").fillna(0).sum()),
                    "reject_adf_invalid": int(pd.to_numeric(d["reject_adf_invalid_count"], errors="coerce").fillna(0).sum()),
                    "reject_eg_invalid": int(pd.to_numeric(d["reject_eg_invalid_count"], errors="coerce").fillna(0).sum()),
                    "reject_half_life_too_high": int(pd.to_numeric(d["reject_half_life_too_high_count"], errors="coerce").fillna(0).sum()),
                    "reject_beta_instability": int(pd.to_numeric(d["reject_beta_instability_count"], errors="coerce").fillna(0).sum()),
                    "reject_insufficient_data": int(pd.to_numeric(d["reject_insufficient_data_count"], errors="coerce").fillna(0).sum()),
                    "reject_technical_exception": int(pd.to_numeric(d["reject_technical_exception_count"], errors="coerce").fillna(0).sum()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["eligibility_mode", "universe"]).reset_index(drop=True)
    return out


def build_eligibility_diagnostics_by_date(scans_by_mode_universe: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mode, by_u in scans_by_mode_universe.items():
        for u, d in by_u.items():
            if d.empty:
                continue
            x = d.copy()
            x["scan_date"] = pd.to_datetime(x["scan_date"]).dt.normalize()
            by_date = (
                x.groupby("scan_date", as_index=False)
                .agg(
                    n_pairs=("asset_1", "size"),
                    n_eligible=("eligibility", lambda s: int((s == "ELIGIBLE").sum())),
                    n_watch=("eligibility", lambda s: int((s == "WATCH").sum())),
                    n_out=("eligibility", lambda s: int((s == "OUT").sum())),
                    avg_gating_score=("gating_score", "mean"),
                    med_gating_score=("gating_score", "median"),
                    reject_corr_insufficient=("reject_corr_insufficient_count", "sum"),
                    reject_adf_invalid=("reject_adf_invalid_count", "sum"),
                    reject_eg_invalid=("reject_eg_invalid_count", "sum"),
                    reject_half_life_too_high=("reject_half_life_too_high_count", "sum"),
                    reject_beta_instability=("reject_beta_instability_count", "sum"),
                    reject_insufficient_data=("reject_insufficient_data_count", "sum"),
                    reject_technical_exception=("reject_technical_exception_count", "sum"),
                )
            )
            by_date["eligibility_mode"] = mode
            by_date["universe"] = u
            frames.append(by_date)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_pair_level_reject_diagnostics(scans_by_mode_universe: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    keep_cols = [
        "scan_date",
        "asset_1",
        "asset_2",
        "universe",
        "eligibility_mode",
        "eligibility",
        "gating_score",
        "validity_distance_score",
        "stability_score",
        "n_valid_windows",
        "missing_metric_count",
        "window_exception_count",
        "reject_reason_primary",
        "3m_reject_reason",
        "6m_reject_reason",
        "12m_reject_reason",
        "reject_corr_insufficient_count",
        "reject_adf_invalid_count",
        "reject_eg_invalid_count",
        "reject_half_life_too_high_count",
        "reject_beta_instability_count",
        "reject_insufficient_data_count",
        "reject_technical_exception_count",
    ]
    for mode, by_u in scans_by_mode_universe.items():
        for u, d in by_u.items():
            if d.empty:
                continue
            x = d.copy()
            x["scan_date"] = pd.to_datetime(x["scan_date"]).dt.normalize()
            x["eligibility_mode"] = mode
            x["universe"] = u
            existing = [c for c in keep_cols if c in x.columns]
            frames.append(x[existing])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["eligibility_mode", "universe", "scan_date", "asset_1", "asset_2"]).reset_index(drop=True)


def summarize_best_tables(stability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if stability.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    s = stability.copy()
    s["has_complete_oos"] = np.isfinite(pd.to_numeric(s["oos_min_sharpe"], errors="coerce")) & np.isfinite(
        pd.to_numeric(s["max_gap_abs"], errors="coerce")
    )

    best_context = (
        s.sort_values(
            ["context_name", "has_complete_oos", "robust_multi_split", "stable_cluster", "oos_min_sharpe", "full_sharpe"],
            ascending=[True, False, False, False, False, False],
        )
        .drop_duplicates(subset=["context_name"], keep="first")
        .reset_index(drop=True)
    )

    best_mode = (
        s.groupby("eligibility_mode", as_index=False)
        .agg(
            n_configs=("full_sharpe", "size"),
            median_full_sharpe=("full_sharpe", "median"),
            median_oos_min_sharpe=("oos_min_sharpe", "median"),
            median_max_gap=("max_gap_abs", "median"),
            complete_oos_rate=("has_complete_oos", "mean"),
            robust_rate=("robust_multi_split", "mean"),
            stable_cluster_rate=("stable_cluster", "mean"),
        )
        .sort_values(["robust_rate", "stable_cluster_rate", "complete_oos_rate", "median_oos_min_sharpe"], ascending=False)
    )

    merged = s[s["context_kind"] == "merged"].copy()
    merged = merged.sort_values(
        ["context_name", "has_complete_oos", "robust_multi_split", "stable_cluster", "oos_min_sharpe", "full_sharpe"],
        ascending=[True, False, False, False, False, False],
    )

    strict = s[
        (s["full_sharpe"] >= ROBUST_FULL_SHARPE_MIN)
        & (s["oos_min_sharpe"] >= ROBUST_OOS_SHARPE_MIN)
        & (s["max_gap_abs"] <= ROBUST_GAP_MAX)
    ].copy()
    strict = strict.sort_values(["stable_cluster", "full_sharpe", "oos_min_sharpe"], ascending=[False, False, False])
    return best_context, best_mode, merged, strict


def build_multiple_testing_diagnostics(runs: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    key = ["context_name", "eligibility_mode", "ranking_name", "z_entry", "z_window", "max_hold"]
    tested = runs[runs["segment"] == "FULL"][key].drop_duplicates()
    n_tests_total = int(len(tested))
    strict = stability[
        (stability["full_sharpe"] >= ROBUST_FULL_SHARPE_MIN)
        & (stability["oos_min_sharpe"] >= ROBUST_OOS_SHARPE_MIN)
        & (stability["max_gap_abs"] <= ROBUST_GAP_MAX)
    ][key].drop_duplicates()
    n_strict = int(len(strict))

    rows = [
        {
            "scope": "global",
            "n_tests": n_tests_total,
            "n_strict": n_strict,
            "strict_hit_rate": (n_strict / n_tests_total) if n_tests_total > 0 else np.nan,
        }
    ]
    for ctx, g in tested.groupby("context_name", as_index=False):
        ctx_strict = strict[strict["context_name"] == ctx].copy()
        rows.append(
            {
                "scope": str(ctx),
                "n_tests": int(len(g)),
                "n_strict": int(len(ctx_strict)),
                "strict_hit_rate": (len(ctx_strict) / len(g)) if len(g) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_campaign_summary(
    out_dir: Path,
    runs: pd.DataFrame,
    best_context: pd.DataFrame,
    best_mode: pd.DataFrame,
    strict: pd.DataFrame,
    mt_diag: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("Desk-like Eligibility + Merged Universe Campaign (2015-2025)")
    lines.append("")
    lines.append(f"Date range: {START} -> {END}")
    lines.append(f"Total runs: {len(runs)}")
    lines.append(f"Contexts tested: {runs['context_name'].nunique() if not runs.empty else 0}")
    lines.append(f"Eligibility modes tested: {runs['eligibility_mode'].nunique() if not runs.empty else 0}")
    lines.append(f"Ranking modes tested: {runs['ranking_name'].nunique() if not runs.empty else 0}")
    lines.append("")

    if not best_context.empty:
        lines.append("Best by country/universe:")
        for r in best_context.itertuples(index=False):
            lines.append(
                f"- {r.context_name} | {r.eligibility_mode} | {r.ranking_name} | "
                f"z_entry={r.z_entry} z_window={r.z_window} max_hold={r.max_hold} | "
                f"full={r.full_sharpe:.2f} oos_min={r.oos_min_sharpe:.2f} gap={r.max_gap_abs:.2f}"
            )
    else:
        lines.append("Best by country/universe: <none>")
    lines.append("")

    if not best_mode.empty:
        lines.append("Eligibility mode leaderboard (median stats):")
        for r in best_mode.itertuples(index=False):
            lines.append(
                f"- {r.eligibility_mode}: full_med={r.median_full_sharpe:.2f}, "
                f"oos_min_med={r.median_oos_min_sharpe:.2f}, robust_rate={r.robust_rate:.2%}, "
                f"stable_cluster_rate={r.stable_cluster_rate:.2%}"
            )
    else:
        lines.append("Eligibility mode leaderboard: <none>")
    lines.append("")

    n_strict = int(len(strict))
    lines.append(
        f"Strict objective configs (full>=1.0, each OOS>=0.8, gap<=0.35): {n_strict}"
    )
    if n_strict > 0:
        for r in strict.head(12).itertuples(index=False):
            lines.append(
                f"- {r.context_name} | {r.eligibility_mode} | {r.ranking_name} | "
                f"z_entry={r.z_entry} z_window={r.z_window} max_hold={r.max_hold} | "
                f"full={r.full_sharpe:.2f} oos_min={r.oos_min_sharpe:.2f} gap={r.max_gap_abs:.2f} "
                f"cluster={int(r.cluster_size)}"
            )
    lines.append("")

    if not mt_diag.empty:
        global_row = mt_diag[mt_diag["scope"] == "global"]
        if not global_row.empty:
            g = global_row.iloc[0]
            lines.append(
                f"Multiple testing diagnostics: n_tests={int(g['n_tests'])}, "
                f"n_strict={int(g['n_strict'])}, strict_hit_rate={float(g['strict_hit_rate']):.3%}"
            )
        else:
            lines.append("Multiple testing diagnostics: <none>")

    (out_dir / "campaign_summary_new.txt").write_text("\n".join(lines), encoding="utf-8")


def _read_csv_if_exists(fp: Path) -> pd.DataFrame:
    if not fp.exists() or fp.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Building base scans...")
    scans_base: dict[str, pd.DataFrame] = {}
    for u in BASE_UNIVERSES:
        s = load_or_build_scan(u, rebuild=False)
        if s.empty:
            raise RuntimeError(f"Empty scan for universe={u}")
        scans_base[u] = s
        print(f"  {u}: rows={len(s):,} dates={s['scan_date'].nunique()}")

    print("Applying eligibility modes...")
    scans_by_mode_universe: dict[str, dict[str, pd.DataFrame]] = {}
    for mode in ELIGIBILITY_MODES:
        scans_by_mode_universe[mode] = {}
        for u, d in scans_base.items():
            scans_by_mode_universe[mode][u] = apply_eligibility_mode(d, mode=mode)

    reject_summary = build_reject_reasons_summary(scans_by_mode_universe)
    reject_summary.to_csv(OUT_DIR / "reject_reasons_summary.csv", index=False)

    diag_by_date = build_eligibility_diagnostics_by_date(scans_by_mode_universe)
    diag_by_date.to_csv(OUT_DIR / "eligibility_diagnostics_by_date.csv", index=False)
    pair_diag = build_pair_level_reject_diagnostics(scans_by_mode_universe)
    pair_diag.to_csv(OUT_DIR / "pair_level_reject_diagnostics.csv", index=False)

    cases = build_cases()
    asset_to_universes = build_asset_to_universes()
    context_scans = build_context_scans(scans_by_mode_universe, cases)

    print("Validating merged-universe integrity (no cross-country pairs)...")
    for mode in ELIGIBILITY_MODES:
        for case in cases:
            scan_df = context_scans[mode].get(case.name, pd.DataFrame())
            ensure_no_cross_country_pairs(
                context_name=f"{case.name}:{mode}",
                context_members=case.members,
                scan_df=scan_df,
                asset_to_universes=asset_to_universes,
            )

    rankings = build_rankings()
    rankings_by_name = {r.name: r for r in rankings}
    pass2_rankings = [r for r in rankings if r.name in {"ranking_current_legacy", "ranking_stability_penalty", "ranking_missing_safe"}]

    pass1_fp = OUT_DIR / "pass1_runs.csv"
    pass2_fp = OUT_DIR / "pass2_runs.csv"
    pass3_fp = OUT_DIR / "pass3_runs.csv"
    pass3_sel_fp = OUT_DIR / "pass3_selected_configs.csv"

    pass1_runs = _read_csv_if_exists(pass1_fp)
    pass1_cases = [c for c in cases if c.name in {"france", "france_sweden"}]
    pass1_segments = build_segments(include_splits=False)
    if pass1_runs.empty:
        print("Running PASS1...")
        pass1_runs = run_matrix(
            pass_stage="PASS1_VALIDATION",
            cases=pass1_cases,
            context_scans=context_scans,
            eligibility_modes=list(ELIGIBILITY_MODES),
            rankings=rankings,
            z_entries=[1.5],
            z_windows=[60],
            max_holds=[20],
            segments=pass1_segments,
        )
        pass1_runs.to_csv(pass1_fp, index=False)
    else:
        print(f"PASS1 resumed from cache: {pass1_fp} ({len(pass1_runs)} rows)")

    pass2_runs = _read_csv_if_exists(pass2_fp)
    pass2_cases = [c for c in cases if c.kind == "single" or c.name in {"france_sweden", "france_germany"}]
    pass2_segments = build_segments(include_splits=False)
    if pass2_runs.empty:
        print("Running PASS2...")
        pass2_runs = run_matrix(
            pass_stage="PASS2_SCREENING",
            cases=pass2_cases,
            context_scans=context_scans,
            eligibility_modes=list(ELIGIBILITY_MODES),
            rankings=pass2_rankings,
            z_entries=[],
            z_windows=[],
            max_holds=[],
            segments=pass2_segments,
            param_grid=PASS2_PARAM_GRID,
        )
        pass2_runs.to_csv(pass2_fp, index=False)
    else:
        print(f"PASS2 resumed from cache: {pass2_fp} ({len(pass2_runs)} rows)")

    selected_pass3 = _read_csv_if_exists(pass3_sel_fp)
    if selected_pass3.empty:
        selected_pass3 = select_pass3_configs(pass2_runs)
        selected_pass3.to_csv(pass3_sel_fp, index=False)
    else:
        print(f"PASS3 config selection resumed from cache: {pass3_sel_fp} ({len(selected_pass3)} rows)")

    if selected_pass3.empty:
        print("PASS3 selection empty, using fallback defaults.")
        selected_pass3 = pd.DataFrame(
            [
                {
                    "eligibility_mode": ELIGIBILITY_V2_HARD_12M_PLUS_SHORT,
                    "ranking_name": "ranking_missing_safe",
                    "z_entry": 1.5,
                    "z_window": 60,
                    "max_hold": 20,
                },
                {
                    "eligibility_mode": ELIGIBILITY_V5_STABILITY_FOCUSED,
                    "ranking_name": "ranking_stability_penalty",
                    "z_entry": 1.5,
                    "z_window": 80,
                    "max_hold": 20,
                },
            ]
        )
        selected_pass3.to_csv(pass3_sel_fp, index=False)

    pass3_runs = _read_csv_if_exists(pass3_fp)
    if pass3_runs.empty:
        print("Running PASS3...")
        pass3_runs = run_selected_configs(
            pass_stage="PASS3_FOCUS",
            selected_configs=selected_pass3,
            rankings_by_name=rankings_by_name,
            cases=cases,
            context_scans=context_scans,
            segments=build_segments(include_splits=True),
        )
        pass3_runs.to_csv(pass3_fp, index=False)
    else:
        print(f"PASS3 resumed from cache: {pass3_fp} ({len(pass3_runs)} rows)")

    runs_all = pd.concat([pass1_runs, pass2_runs, pass3_runs], ignore_index=True)
    runs_all.to_csv(OUT_DIR / "runs_new_campaign.csv", index=False)

    stability = annotate_local_clusters(build_stability_table(runs_all))
    stability.to_csv(OUT_DIR / "stable_clusters_new_campaign.csv", index=False)

    best_context, best_mode, merged_results, strict = summarize_best_tables(stability)
    best_context.to_csv(OUT_DIR / "best_by_country_or_universe.csv", index=False)
    best_mode.to_csv(OUT_DIR / "best_by_eligibility_mode.csv", index=False)
    merged_results.to_csv(OUT_DIR / "merged_universe_results.csv", index=False)
    strict.to_csv(OUT_DIR / "strict_objective_configs.csv", index=False)

    mt_diag = build_multiple_testing_diagnostics(runs_all, stability)
    mt_diag.to_csv(OUT_DIR / "multiple_testing_diagnostics.csv", index=False)

    write_campaign_summary(
        out_dir=OUT_DIR,
        runs=runs_all,
        best_context=best_context,
        best_mode=best_mode,
        strict=strict,
        mt_diag=mt_diag,
    )

    print("Saved outputs in:", OUT_DIR)
    print("runs_new_campaign.csv:", OUT_DIR / "runs_new_campaign.csv")
    print("stable_clusters_new_campaign.csv:", OUT_DIR / "stable_clusters_new_campaign.csv")
    print("strict_objective_configs.csv:", OUT_DIR / "strict_objective_configs.csv")


if __name__ == "__main__":
    main()
