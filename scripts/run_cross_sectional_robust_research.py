from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.global_loop import run_global_ranking_daily_portfolio
from object.class_file import BatchConfig, StrategyParams
from utils.inline_scanner import InlineScannerConfig, build_scans_inline


START = "2015-12-31"
END = "2025-12-31"
FULL_START = START
FULL_END = END
SCAN_FREQ = "ME"

DEFAULT_TOP_N = 20
DEFAULT_MAX_POSITIONS = 5
DEFAULT_FEES = 0.0002
DEFAULT_BETA_MODE = "static"
DEFAULT_SIGNAL_PCA_WINDOW = 252
DEFAULT_SIGNAL_PCA_COMPONENTS = 3
DEFAULT_SIGNAL_PCA_MIN_ASSETS = 10

TARGET_UNIVERSES = ["france", "sweden", "italy", "germany"]
BACKUP_UNIVERSES = ["spain", "netherlands", "switzerland", "finland", "belgium", "uk", "denmark", "norway"]

Z_ENTRY_GRID = [1.2, 1.5, 1.8]
Z_WINDOW_GRID = [40, 60, 80, 100]
MAX_HOLD_GRID = [15, 20, 25, 30]

ABLATION_Z_ENTRY_GRID = [1.2, 1.5, 1.8]
ABLATION_Z_WINDOW_GRID = [40, 60, 80]
ABLATION_MAX_HOLD_GRID = [20, 30]

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

ROBUST_FULL_SHARPE_MIN = 1.0
ROBUST_OOS_SHARPE_MIN = 0.8
ROBUST_GAP_MAX = 0.35
ROBUST_MIN_CLUSTER_SIZE = 3
ROBUST_MIN_TRADES = 50

BASE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"
OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "robust_cross_sectional_long_2015_2025"
SCAN_CACHE_DIR = OUT_DIR / "scans"
HEATMAP_DIR = OUT_DIR / "heatmaps"


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    signal_space: str
    selection_mode: str
    selection_score_variant: str = "baseline"
    selection_winsor_quantile: float = 0.0
    selection_stability_penalty: float = 0.0
    max_pairs_per_asset: int = 0
    min_corr_12m: float | None = None
    max_half_life_6m: float | None = None
    max_beta_std: float | None = None
    min_n_valid_windows: int | None = None
    pair_return_cap: float | None = None
    trade_return_isolated_cap: float | None = None
    portfolio_vol_target: float | None = None
    portfolio_vol_lookback: int = 20
    portfolio_vol_max_scale: float = 1.0
    notes: str = ""


@dataclass(frozen=True)
class SegmentConfig:
    segment: str
    split_id: str
    split_kind: str
    start_date: str
    end_date: str


def linked_thresholds(z_entry: float) -> tuple[float, float]:
    return (round(float(z_entry) / 3.0, 4), round(2.0 * float(z_entry), 4))


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


def universe_asset_counts() -> pd.DataFrame:
    reg = pd.read_csv(ASSET_REGISTRY_PATH, usecols=["category_id", "asset"])
    counts = (
        reg.groupby("category_id", as_index=False)["asset"]
        .nunique()
        .rename(columns={"asset": "n_assets", "category_id": "universe"})
        .sort_values("n_assets", ascending=False)
    )
    counts["universe"] = counts["universe"].astype(str).str.lower()
    return counts


def select_universes(min_assets: int = 12, min_required: int = 4) -> list[str]:
    counts = universe_asset_counts()
    avail = counts[counts["n_assets"] >= int(min_assets)]["universe"].tolist()

    selected: list[str] = []
    for u in TARGET_UNIVERSES:
        if u in avail:
            selected.append(u)

    for u in BACKUP_UNIVERSES:
        if len(selected) >= min_required:
            break
        if u in avail and u not in selected:
            selected.append(u)

    if len(selected) < min_required:
        for u in avail:
            if u not in selected:
                selected.append(u)
            if len(selected) >= min_required:
                break

    return selected


def build_segment_list() -> list[SegmentConfig]:
    segs = [
        SegmentConfig(
            segment="FULL",
            split_id="full",
            split_kind="FULL",
            start_date=FULL_START,
            end_date=FULL_END,
        )
    ]
    for s in SPLITS:
        sid = s["split_id"]
        segs.append(
            SegmentConfig(
                segment=f"{sid.upper()}_IS",
                split_id=sid,
                split_kind="IS",
                start_date=s["is_start"],
                end_date=s["is_end"],
            )
        )
        segs.append(
            SegmentConfig(
                segment=f"{sid.upper()}_OOS",
                split_id=sid,
                split_kind="OOS",
                start_date=s["oos_start"],
                end_date=s["oos_end"],
            )
        )
    return segs


def load_or_build_scan(universe: str, inline_cfg: InlineScannerConfig, rebuild: bool = False) -> pd.DataFrame:
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = SCAN_CACHE_DIR / f"{universe}.parquet"

    if fp.exists() and not rebuild:
        try:
            df = pd.read_parquet(fp)
            df = normalize_scans(df, universe)
            dmin = pd.to_datetime(df["scan_date"]).min()
            dmax = pd.to_datetime(df["scan_date"]).max()
            if pd.notna(dmin) and pd.notna(dmax):
                if dmin <= pd.Timestamp(START) and dmax >= pd.Timestamp(END):
                    return df
        except Exception:
            pass

    df = build_scans_inline(
        universes=[universe],
        start_date=START,
        end_date=END,
        freq=SCAN_FREQ,
        cfg=inline_cfg,
    )
    if df.empty:
        return pd.DataFrame()

    df = normalize_scans(df, universe)
    df.to_parquet(fp, index=False)
    return df


def build_scans_for_universes(universes: list[str], rebuild: bool = False) -> dict[str, pd.DataFrame]:
    inline_cfg = InlineScannerConfig(
        raw_data_path=BASE_DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
    )
    out: dict[str, pd.DataFrame] = {}
    for u in universes:
        df = load_or_build_scan(u, inline_cfg=inline_cfg, rebuild=rebuild)
        if df.empty:
            continue
        out[u] = df
    return out


def build_strategy_params(family: FamilyConfig, z_entry: float, z_window: int, max_hold: int) -> StrategyParams:
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
        max_holding_days=int(max_hold),
        signal_space=family.signal_space,
        selection_mode=family.selection_mode,
        selection_score_variant=family.selection_score_variant,
        selection_winsor_quantile=float(family.selection_winsor_quantile),
        selection_stability_penalty=float(family.selection_stability_penalty),
        max_pairs_per_asset=int(family.max_pairs_per_asset),
        min_corr_12m=family.min_corr_12m,
        max_half_life_6m=family.max_half_life_6m,
        max_beta_std=family.max_beta_std,
        min_n_valid_windows=family.min_n_valid_windows,
        pair_return_cap=family.pair_return_cap,
        trade_return_isolated_cap=family.trade_return_isolated_cap,
        portfolio_vol_target=family.portfolio_vol_target,
        portfolio_vol_lookback=int(family.portfolio_vol_lookback),
        portfolio_vol_max_scale=float(family.portfolio_vol_max_scale),
        pca_signal_window=DEFAULT_SIGNAL_PCA_WINDOW,
        pca_signal_components=DEFAULT_SIGNAL_PCA_COMPONENTS,
        pca_signal_min_assets=DEFAULT_SIGNAL_PCA_MIN_ASSETS,
    )


def run_one(
    universe: str,
    scans: pd.DataFrame,
    family: FamilyConfig,
    z_entry: float,
    z_window: int,
    max_hold: int,
    segment: SegmentConfig,
) -> dict[str, Any]:
    cfg = BatchConfig(
        data_path=BASE_DATA_PATH,
        start_date=segment.start_date,
        end_date=segment.end_date,
    )
    params = build_strategy_params(family, z_entry=z_entry, z_window=z_window, max_hold=max_hold)

    t0 = time.time()
    res = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=[universe],
        scans=scans,
    )
    runtime_s = time.time() - t0

    row: dict[str, Any] = {
        "universe": universe,
        "segment": segment.segment,
        "split_id": segment.split_id,
        "split_kind": segment.split_kind,
        "start_date": segment.start_date,
        "end_date": segment.end_date,
        "family": family.name,
        "signal_space": family.signal_space,
        "selection_mode": family.selection_mode,
        "selection_score_variant": family.selection_score_variant,
        "selection_winsor_quantile": family.selection_winsor_quantile,
        "selection_stability_penalty": family.selection_stability_penalty,
        "z_entry": float(z_entry),
        "z_exit": float(params.z_exit),
        "z_stop": float(params.z_stop),
        "z_window": int(z_window),
        "max_hold": int(max_hold),
        "max_pairs_per_asset": int(family.max_pairs_per_asset),
        "min_corr_12m": family.min_corr_12m,
        "max_half_life_6m": family.max_half_life_6m,
        "max_beta_std": family.max_beta_std,
        "min_n_valid_windows": family.min_n_valid_windows,
        "pair_return_cap": family.pair_return_cap,
        "trade_return_isolated_cap": family.trade_return_isolated_cap,
        "portfolio_vol_target": family.portfolio_vol_target,
        "portfolio_vol_lookback": family.portfolio_vol_lookback,
        "portfolio_vol_max_scale": family.portfolio_vol_max_scale,
        "notes": family.notes,
        "runtime_s": round(float(runtime_s), 3),
    }

    if not res:
        row.update({"ok": False})
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
            "max_abs_daily_return": st.get("Max abs daily return"),
            "p99_abs_daily_return": st.get("P99 abs daily return"),
            "max_abs_pair_mtm_raw": st.get("Max abs pair mtm raw"),
            "max_vol_scale": st.get("Max vol scale"),
            "anomaly_flag": bool(st.get("Anomaly flag", False)),
            "anomaly_reasons": st.get("Anomaly reasons", ""),
        }
    )
    return row


def run_campaign(
    campaign_name: str,
    universes: list[str],
    scans_by_universe: dict[str, pd.DataFrame],
    families: list[FamilyConfig],
    z_entry_grid: list[float],
    z_window_grid: list[int],
    max_hold_grid: list[int],
    segments: list[SegmentConfig],
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    total = (
        len(universes)
        * len(families)
        * len(segments)
        * len(z_window_grid)
        * len(z_entry_grid)
        * len(max_hold_grid)
    )
    done = 0
    t0 = time.time()

    live_path = out_dir / f"{campaign_name}_runs_live.csv"
    if live_path.exists():
        live_path.unlink()

    for universe in universes:
        scans = scans_by_universe[universe]
        for family in families:
            for segment in segments:
                for z_window in z_window_grid:
                    for z_entry in z_entry_grid:
                        for max_hold in max_hold_grid:
                            done += 1
                            row = run_one(
                                universe=universe,
                                scans=scans,
                                family=family,
                                z_entry=z_entry,
                                z_window=z_window,
                                max_hold=max_hold,
                                segment=segment,
                            )
                            rows.append(row)

                            if done % 50 == 0 or done == total:
                                elapsed = max(1e-6, time.time() - t0)
                                rate = done / elapsed
                                eta = (total - done) / rate if rate > 0 else np.nan
                                print(
                                    f"[{campaign_name}] {done}/{total} ({done/total:.1%}) | "
                                    f"{rate:.2f} runs/s | ETA {eta/60:.1f} min"
                                )
                                pd.DataFrame(rows).to_csv(live_path, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"{campaign_name}_runs.csv", index=False)
    return out

def build_stability_table(results: pd.DataFrame) -> pd.DataFrame:
    d = results[(results["ok"] == True)].copy()
    if d.empty:
        return pd.DataFrame()

    key = ["universe", "family", "z_entry", "z_window", "max_hold"]
    rows: list[dict[str, Any]] = []

    for k, g in d.groupby(key, as_index=False):
        row = {
            "universe": k[0],
            "family": k[1],
            "z_entry": float(k[2]),
            "z_window": int(k[3]),
            "max_hold": int(k[4]),
        }

        full = g[g["segment"] == "FULL"]
        row["full_sharpe"] = float(full["sharpe"].median()) if not full.empty else np.nan
        row["full_cagr"] = float(full["cagr"].median()) if not full.empty else np.nan
        row["full_maxdd"] = float(full["max_drawdown"].median()) if not full.empty else np.nan
        row["full_trades"] = float(full["nb_trades"].median()) if not full.empty else np.nan
        row["full_anomaly_rate"] = float(full["anomaly_flag"].mean()) if not full.empty else np.nan
        row["full_final_equity"] = float(full["final_equity"].median()) if not full.empty else np.nan

        split_pass_flags: list[bool] = []
        split_oos_vals: list[float] = []
        split_gap_vals: list[float] = []

        for s in SPLITS:
            sid = s["split_id"]
            seg_is = f"{sid.upper()}_IS"
            seg_oos = f"{sid.upper()}_OOS"
            is_sh = float(g.loc[g["segment"] == seg_is, "sharpe"].median()) if (g["segment"] == seg_is).any() else np.nan
            oos_sh = (
                float(g.loc[g["segment"] == seg_oos, "sharpe"].median()) if (g["segment"] == seg_oos).any() else np.nan
            )
            gap = abs(is_sh - oos_sh) if np.isfinite(is_sh) and np.isfinite(oos_sh) else np.nan

            row[f"{sid}_is_sharpe"] = is_sh
            row[f"{sid}_oos_sharpe"] = oos_sh
            row[f"{sid}_gap_abs"] = gap

            pass_split = (
                np.isfinite(oos_sh)
                and np.isfinite(gap)
                and (oos_sh >= ROBUST_OOS_SHARPE_MIN)
                and (gap <= ROBUST_GAP_MAX)
            )
            row[f"{sid}_pass"] = bool(pass_split)
            split_pass_flags.append(bool(pass_split))
            split_oos_vals.append(oos_sh)
            split_gap_vals.append(gap)

        split_oos_arr = np.asarray(split_oos_vals, dtype=float)
        split_gap_arr = np.asarray(split_gap_vals, dtype=float)
        row["oos_min_sharpe"] = float(np.nanmin(split_oos_arr)) if np.isfinite(np.nanmin(split_oos_arr)) else np.nan
        row["max_gap_abs"] = float(np.nanmax(split_gap_arr)) if np.isfinite(np.nanmax(split_gap_arr)) else np.nan
        row["split_pass_count"] = int(sum(1 for x in split_pass_flags if x))
        row["full_pass"] = bool(np.isfinite(row["full_sharpe"]) and row["full_sharpe"] >= ROBUST_FULL_SHARPE_MIN)
        row["near_full_pass"] = bool(np.isfinite(row["full_sharpe"]) and row["full_sharpe"] >= 0.95)
        row["robust_multi_split"] = bool(row["full_pass"] and all(split_pass_flags))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["universe", "family", "full_sharpe"], ascending=[True, True, False])


def annotate_local_clusters(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability

    z_entry_levels = sorted(set(float(x) for x in stability["z_entry"].dropna().unique().tolist()))
    z_window_levels = sorted(set(int(x) for x in stability["z_window"].dropna().unique().tolist()))
    max_hold_levels = sorted(set(int(x) for x in stability["max_hold"].dropna().unique().tolist()))

    ze_idx = {v: i for i, v in enumerate(z_entry_levels)}
    zw_idx = {v: i for i, v in enumerate(z_window_levels)}
    mh_idx = {v: i for i, v in enumerate(max_hold_levels)}

    out = stability.copy()
    out["cluster_size"] = 0
    out["cluster_median_full_sharpe"] = np.nan

    for (u, f), g in out.groupby(["universe", "family"]):
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
                (out["universe"] == u)
                & (out["family"] == f)
                & (out["z_entry"] == float(r.z_entry))
                & (out["z_window"] == int(r.z_window))
                & (out["max_hold"] == int(r.max_hold))
            )
            out.loc[m, "cluster_size"] = int(len(neigh))
            out.loc[m, "cluster_median_full_sharpe"] = float(neigh["full_sharpe"].median()) if not neigh.empty else np.nan

    out["stable_cluster"] = (
        (out["robust_multi_split"] == True)
        & (out["cluster_size"] >= ROBUST_MIN_CLUSTER_SIZE)
        & (out["full_trades"] >= ROBUST_MIN_TRADES)
        & (out["full_anomaly_rate"] <= 0.10)
    )
    return out


def build_country_summary(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability

    cols = [
        "universe",
        "family",
        "z_entry",
        "z_window",
        "max_hold",
        "full_sharpe",
        "split1_oos_sharpe",
        "split2_oos_sharpe",
        "split3_oos_sharpe",
        "oos_min_sharpe",
        "max_gap_abs",
        "full_cagr",
        "full_maxdd",
        "full_trades",
        "cluster_size",
        "stable_cluster",
    ]

    d = stability.copy()
    d["score"] = (
        pd.to_numeric(d["full_sharpe"], errors="coerce").fillna(-9.0)
        + 0.35 * pd.to_numeric(d["oos_min_sharpe"], errors="coerce").fillna(-9.0)
        - 0.20 * pd.to_numeric(d["max_gap_abs"], errors="coerce").fillna(9.0)
        + 0.02 * np.log1p(pd.to_numeric(d["full_trades"], errors="coerce").fillna(0.0))
    )

    best = (
        d.sort_values(["universe", "stable_cluster", "score", "full_sharpe"], ascending=[True, False, False, False])
        .groupby("universe", as_index=False)
        .head(1)
    )
    return best[cols + ["score"]].sort_values("score", ascending=False)


def select_final_configs(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame()

    d = stability.copy()
    d = d[
        (d["stable_cluster"] == True)
        & (d["full_sharpe"] >= ROBUST_FULL_SHARPE_MIN)
        & (d["oos_min_sharpe"] >= ROBUST_OOS_SHARPE_MIN)
        & (d["max_gap_abs"] <= ROBUST_GAP_MAX)
    ].copy()

    if d.empty:
        return d

    d["score"] = (
        pd.to_numeric(d["full_sharpe"], errors="coerce").fillna(-9.0)
        + 0.40 * pd.to_numeric(d["oos_min_sharpe"], errors="coerce").fillna(-9.0)
        - 0.20 * pd.to_numeric(d["max_gap_abs"], errors="coerce").fillna(9.0)
        + 0.02 * np.log1p(pd.to_numeric(d["full_trades"], errors="coerce").fillna(0.0))
    )

    selected_rows: list[pd.Series] = []
    used = set()

    for u in TARGET_UNIVERSES:
        cu = d[d["universe"] == u].sort_values(["score", "full_sharpe"], ascending=False)
        if cu.empty:
            continue
        row = cu.iloc[0]
        selected_rows.append(row)
        used.add(u)

    if len(selected_rows) < 4:
        extras = d[~d["universe"].isin(list(used))].sort_values(["score", "full_sharpe"], ascending=False)
        for _, row in extras.iterrows():
            selected_rows.append(row)
            used.add(str(row["universe"]))
            if len(selected_rows) >= 4:
                break

    if not selected_rows:
        return pd.DataFrame()

    out = pd.DataFrame(selected_rows).drop_duplicates(subset=["universe"], keep="first")
    return out.sort_values(["score", "full_sharpe"], ascending=False).head(4).reset_index(drop=True)

def build_heatmaps(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        return

    d = results[results["ok"] == True].copy()
    if d.empty:
        return

    for universe in sorted(d["universe"].unique()):
        for family in sorted(d["family"].unique()):
            for segment in sorted(d["segment"].unique()):
                x = d[
                    (d["universe"] == universe)
                    & (d["family"] == family)
                    & (d["segment"] == segment)
                ].copy()
                if x.empty:
                    continue
                piv = (
                    x.groupby(["z_window", "z_entry"], as_index=False)["sharpe"]
                    .median()
                    .pivot(index="z_window", columns="z_entry", values="sharpe")
                    .sort_index()
                )
                piv.to_csv(out_dir / f"heatmap_{universe}_{family}_{segment}.csv")

                if HAS_PLOT:
                    plt.figure(figsize=(6.2, 4.0))
                    sns.heatmap(
                        piv,
                        annot=True,
                        fmt=".2f",
                        cmap="RdYlGn",
                        center=0.8,
                        vmin=-1.0,
                        vmax=2.0,
                    )
                    plt.title(f"{universe} | {family} | {segment}")
                    plt.tight_layout()
                    plt.savefig(out_dir / f"heatmap_{universe}_{family}_{segment}.png", dpi=130)
                    plt.close()


def build_baseline_families() -> list[FamilyConfig]:
    return [
        FamilyConfig(name="raw_legacy", signal_space="raw", selection_mode="legacy"),
        FamilyConfig(name="idio_legacy", signal_space="idio_pca", selection_mode="legacy"),
        FamilyConfig(name="raw_composite", signal_space="raw", selection_mode="composite_quality"),
        FamilyConfig(name="idio_composite", signal_space="idio_pca", selection_mode="composite_quality"),
        FamilyConfig(
            name="composite_mild_filter",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            min_corr_12m=0.55,
            max_half_life_6m=80.0,
            max_beta_std=0.35,
            min_n_valid_windows=2,
            max_pairs_per_asset=2,
            notes="mild_filter",
        ),
    ]


def build_ablation_families() -> list[FamilyConfig]:
    return [
        FamilyConfig(
            name="ablation_baseline",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            notes="baseline",
        ),
        FamilyConfig(
            name="ablation_rank_percentile",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            selection_score_variant="rank_percentile",
            notes="selection_rank_percentile",
        ),
        FamilyConfig(
            name="ablation_robust_zscore",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            selection_score_variant="robust_zscore",
            selection_winsor_quantile=0.05,
            notes="selection_robust_zscore",
        ),
        FamilyConfig(
            name="ablation_rank_stability",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            selection_score_variant="rank_stability_penalty",
            selection_stability_penalty=0.35,
            notes="selection_rank_stability_penalty",
        ),
        FamilyConfig(
            name="ablation_pair_cap",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            pair_return_cap=0.05,
            trade_return_isolated_cap=0.20,
            notes="risk_pair_cap",
        ),
        FamilyConfig(
            name="ablation_pair_cap_voltarget",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            pair_return_cap=0.05,
            trade_return_isolated_cap=0.20,
            portfolio_vol_target=0.012,
            portfolio_vol_lookback=20,
            portfolio_vol_max_scale=1.0,
            notes="risk_pair_cap_plus_vol_target",
        ),
    ]


def summarize_ablation(ablation_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if ablation_results.empty:
        return pd.DataFrame()

    stab = annotate_local_clusters(build_stability_table(ablation_results))
    stab.to_csv(out_dir / "ablation_stability.csv", index=False)

    rows: list[dict[str, Any]] = []
    for fam, g in stab.groupby("family"):
        row = {
            "family": fam,
            "n_configs": int(len(g)),
            "median_full_sharpe": float(g["full_sharpe"].median()) if not g.empty else np.nan,
            "median_oos_min_sharpe": float(g["oos_min_sharpe"].median()) if not g.empty else np.nan,
            "median_max_gap": float(g["max_gap_abs"].median()) if not g.empty else np.nan,
            "robust_pass_rate": float((g["robust_multi_split"] == True).mean()) if not g.empty else np.nan,
            "stable_cluster_rate": float((g["stable_cluster"] == True).mean()) if not g.empty else np.nan,
            "anomaly_rate": float((pd.to_numeric(g["full_anomaly_rate"], errors="coerce") > 0).mean()) if not g.empty else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        ["robust_pass_rate", "stable_cluster_rate", "median_oos_min_sharpe", "median_full_sharpe"],
        ascending=False,
    )
    out.to_csv(out_dir / "ablation_summary.csv", index=False)
    return out


def select_retained_variants(ablation_summary: pd.DataFrame) -> list[str]:
    if ablation_summary.empty:
        return []

    base = ablation_summary[ablation_summary["family"] == "ablation_baseline"]
    if base.empty:
        return []

    b = base.iloc[0]
    b_pass = float(b.get("robust_pass_rate", 0.0))
    b_oos = float(b.get("median_oos_min_sharpe", -9.0))
    b_anom = float(b.get("anomaly_rate", 1.0))

    cands = ablation_summary[ablation_summary["family"] != "ablation_baseline"].copy()
    if cands.empty:
        return []

    cands = cands[
        (cands["robust_pass_rate"] >= b_pass - 0.02)
        & (cands["median_oos_min_sharpe"] >= b_oos - 0.05)
        & (cands["anomaly_rate"] <= b_anom + 0.05)
    ].copy()

    if cands.empty:
        return []

    cands = cands.sort_values(
        ["robust_pass_rate", "stable_cluster_rate", "median_oos_min_sharpe", "median_full_sharpe"],
        ascending=False,
    )
    return cands["family"].head(2).tolist()


def variant_family_from_ablation_name(name: str) -> FamilyConfig | None:
    mapping: dict[str, FamilyConfig] = {
        "ablation_rank_percentile": FamilyConfig(
            name="idio_composite_rank_percentile",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            selection_score_variant="rank_percentile",
            notes="retained_ablation_variant",
        ),
        "ablation_robust_zscore": FamilyConfig(
            name="idio_composite_robust_zscore",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            selection_score_variant="robust_zscore",
            selection_winsor_quantile=0.05,
            notes="retained_ablation_variant",
        ),
        "ablation_rank_stability": FamilyConfig(
            name="idio_composite_rank_stability",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            selection_score_variant="rank_stability_penalty",
            selection_stability_penalty=0.35,
            notes="retained_ablation_variant",
        ),
        "ablation_pair_cap": FamilyConfig(
            name="idio_composite_pair_cap",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            pair_return_cap=0.05,
            trade_return_isolated_cap=0.20,
            notes="retained_ablation_variant",
        ),
        "ablation_pair_cap_voltarget": FamilyConfig(
            name="idio_composite_pair_cap_voltarget",
            signal_space="idio_pca",
            selection_mode="composite_quality",
            pair_return_cap=0.05,
            trade_return_isolated_cap=0.20,
            portfolio_vol_target=0.012,
            portfolio_vol_lookback=20,
            portfolio_vol_max_scale=1.0,
            notes="retained_ablation_variant",
        ),
    }
    return mapping.get(name)

def run_and_export_main_campaign(
    campaign_name: str,
    universes: list[str],
    scans_by_universe: dict[str, pd.DataFrame],
    families: list[FamilyConfig],
    z_entry_grid: list[float],
    z_window_grid: list[int],
    max_hold_grid: list[int],
    segments: list[SegmentConfig],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = run_campaign(
        campaign_name=campaign_name,
        universes=universes,
        scans_by_universe=scans_by_universe,
        families=families,
        z_entry_grid=z_entry_grid,
        z_window_grid=z_window_grid,
        max_hold_grid=max_hold_grid,
        segments=segments,
        out_dir=out_dir,
    )

    stability = annotate_local_clusters(build_stability_table(runs))

    runs.to_csv(out_dir / f"{campaign_name}_runs.csv", index=False)
    stability.to_csv(out_dir / f"{campaign_name}_stable_clusters_multi_splits.csv", index=False)

    anomal = runs[(runs["ok"] == True) & (runs["anomaly_flag"] == True)].copy()
    anomal.to_csv(out_dir / f"{campaign_name}_anomalies.csv", index=False)

    country_summary = build_country_summary(stability)
    country_summary.to_csv(out_dir / f"{campaign_name}_country_summary.csv", index=False)

    build_heatmaps(runs, HEATMAP_DIR / campaign_name)

    return runs, stability


def ensure_min_four_countries(
    stability: pd.DataFrame,
    universes_current: list[str],
    scans_by_universe: dict[str, pd.DataFrame],
    segments: list[SegmentConfig],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if stability.empty:
        return pd.DataFrame(), pd.DataFrame(), universes_current

    candidate = stability[
        (stability["stable_cluster"] == True)
        & (stability["full_sharpe"] >= ROBUST_FULL_SHARPE_MIN)
    ].copy()
    n_countries = candidate["universe"].nunique()

    if n_countries >= 4:
        return pd.DataFrame(), pd.DataFrame(), universes_current

    missing_target = [u for u in TARGET_UNIVERSES if u in universes_current and u not in candidate["universe"].unique()]
    add_universes: list[str] = []
    for u in missing_target + BACKUP_UNIVERSES:
        if u not in universes_current and u not in add_universes:
            add_universes.append(u)
        if len(universes_current) + len(add_universes) >= 6:
            break

    if not add_universes:
        return pd.DataFrame(), pd.DataFrame(), universes_current

    inline_cfg = InlineScannerConfig(
        raw_data_path=BASE_DATA_PATH,
        asset_registry_path=ASSET_REGISTRY_PATH,
        lookback_days=504,
        min_obs=100,
        liquidity_lookback=20,
        liquidity_min_moves=0.0,
    )
    for u in add_universes:
        if u in scans_by_universe:
            continue
        scans_u = load_or_build_scan(u, inline_cfg=inline_cfg, rebuild=False)
        if not scans_u.empty:
            scans_by_universe[u] = scans_u

    fallback_universes = [u for u in add_universes if u in scans_by_universe]
    if not fallback_universes:
        return pd.DataFrame(), pd.DataFrame(), universes_current

    fallback_family = FamilyConfig(
        name="idio_composite_guardrails",
        signal_space="idio_pca",
        selection_mode="composite_quality",
        selection_score_variant="rank_stability_penalty",
        selection_stability_penalty=0.35,
        max_pairs_per_asset=2,
        pair_return_cap=0.05,
        trade_return_isolated_cap=0.20,
        notes="fallback_guardrails_pass",
    )

    runs, stability_fb = run_and_export_main_campaign(
        campaign_name="fallback_guardrails_pass",
        universes=fallback_universes,
        scans_by_universe=scans_by_universe,
        families=[fallback_family],
        z_entry_grid=Z_ENTRY_GRID,
        z_window_grid=Z_WINDOW_GRID,
        max_hold_grid=MAX_HOLD_GRID,
        segments=segments,
        out_dir=out_dir,
    )

    all_universes = list(dict.fromkeys(universes_current + fallback_universes))
    return runs, stability_fb, all_universes


def build_config_payload(final_cfgs: pd.DataFrame) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if final_cfgs.empty:
        return payload

    for r in final_cfgs.itertuples(index=False):
        cfg = {
            "universe": str(r.universe),
            "family": str(r.family),
            "z_entry": float(r.z_entry),
            "z_window": int(r.z_window),
            "max_hold": int(r.max_hold),
            "z_exit": round(float(r.z_entry) / 3.0, 4),
            "z_stop": round(2.0 * float(r.z_entry), 4),
            "beta_mode": DEFAULT_BETA_MODE,
            "fees": DEFAULT_FEES,
            "top_n_candidates": DEFAULT_TOP_N,
            "max_positions": DEFAULT_MAX_POSITIONS,
        }
        payload.append(cfg)

    return payload


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    journal: list[dict[str, Any]] = []

    universes = select_universes(min_assets=12, min_required=4)
    journal.append({"stage": "inventory", "selected_universes": ",".join(universes), "n_universes": len(universes)})

    scans_by_universe = build_scans_for_universes(universes, rebuild=False)
    universes = [u for u in universes if u in scans_by_universe]
    journal.append(
        {
            "stage": "scan_build",
            "usable_universes": ",".join(universes),
            "n_usable_universes": len(universes),
            "scan_rows_total": int(sum(len(scans_by_universe[u]) for u in universes)),
        }
    )

    if len(universes) < 4:
        raise RuntimeError(f"Not enough usable universes after scans build: {universes}")

    segs = build_segment_list()

    ablation_universes = [u for u in ["france", "sweden", "italy", "germany"] if u in universes][:2]
    if len(ablation_universes) < 2:
        ablation_universes = universes[:2]

    ablation_families = build_ablation_families()
    ablation_runs = run_campaign(
        campaign_name="ablation",
        universes=ablation_universes,
        scans_by_universe=scans_by_universe,
        families=ablation_families,
        z_entry_grid=ABLATION_Z_ENTRY_GRID,
        z_window_grid=ABLATION_Z_WINDOW_GRID,
        max_hold_grid=ABLATION_MAX_HOLD_GRID,
        segments=segs,
        out_dir=OUT_DIR,
    )
    ablation_runs.to_csv(OUT_DIR / "ablation_runs.csv", index=False)
    ablation_summary = summarize_ablation(ablation_runs, OUT_DIR)
    retained = select_retained_variants(ablation_summary)
    journal.append({"stage": "ablation", "ablation_universes": ",".join(ablation_universes), "retained": ",".join(retained)})

    families = build_baseline_families()
    for name in retained:
        fam = variant_family_from_ablation_name(name)
        if fam is not None:
            families.append(fam)

    baseline_names = {f.name for f in build_baseline_families()}
    families_out: list[FamilyConfig] = []
    extra_count = 0
    for fam in families:
        if fam.name in baseline_names:
            families_out.append(fam)
        else:
            if extra_count < 2:
                families_out.append(fam)
                extra_count += 1
    families = families_out

    fam_df = pd.DataFrame([f.__dict__ for f in families])
    fam_df.to_csv(OUT_DIR / "families_used.csv", index=False)

    runs_main, stability_main = run_and_export_main_campaign(
        campaign_name="main",
        universes=universes,
        scans_by_universe=scans_by_universe,
        families=families,
        z_entry_grid=Z_ENTRY_GRID,
        z_window_grid=Z_WINDOW_GRID,
        max_hold_grid=MAX_HOLD_GRID,
        segments=segs,
        out_dir=OUT_DIR,
    )

    runs_fb, stability_fb, universes_after_fb = ensure_min_four_countries(
        stability=stability_main,
        universes_current=universes,
        scans_by_universe=scans_by_universe,
        segments=segs,
        out_dir=OUT_DIR,
    )

    if not runs_fb.empty:
        runs_all = pd.concat([runs_main, runs_fb], ignore_index=True)
        stability_all = annotate_local_clusters(build_stability_table(runs_all))
        runs_all.to_csv(OUT_DIR / "all_runs_combined.csv", index=False)
        stability_all.to_csv(OUT_DIR / "all_stable_clusters_multi_splits.csv", index=False)
        journal.append(
            {
                "stage": "fallback",
                "fallback_universes": ",".join([u for u in universes_after_fb if u not in universes]),
                "enabled": True,
            }
        )
    else:
        runs_all = runs_main.copy()
        stability_all = stability_main.copy()
        journal.append({"stage": "fallback", "enabled": False})

    country_summary = build_country_summary(stability_all)
    country_summary.to_csv(OUT_DIR / "country_summary_all.csv", index=False)

    final_cfgs = select_final_configs(stability_all)
    final_cfgs.to_csv(OUT_DIR / "final_4_configs.csv", index=False)

    payload = build_config_payload(final_cfgs)
    with open(OUT_DIR / "final_4_configs.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    robust_cov = stability_all[
        (stability_all["stable_cluster"] == True)
        & (stability_all["full_sharpe"] >= ROBUST_FULL_SHARPE_MIN)
    ].copy()
    cov_tbl = (
        robust_cov.groupby("universe", as_index=False)
        .agg(
            n_robust_configs=("full_sharpe", "size"),
            best_full_sharpe=("full_sharpe", "max"),
            median_oos_min=("oos_min_sharpe", "median"),
            median_max_gap=("max_gap_abs", "median"),
        )
        .sort_values(["best_full_sharpe", "n_robust_configs"], ascending=False)
    )
    cov_tbl.to_csv(OUT_DIR / "coverage_by_country.csv", index=False)

    journal_df = pd.DataFrame(journal)
    journal_df.to_csv(OUT_DIR / "campaign_journal.csv", index=False)

    summary_lines = [
        "# Robust Cross-Sectional Campaign 2015-2025",
        "",
        f"- Date range: {START} -> {END}",
        f"- Universes used: {', '.join(sorted(set(runs_all['universe'].astype(str).str.lower().unique().tolist())))}",
        f"- Families used: {', '.join(sorted(set(runs_all['family'].astype(str).unique().tolist())))}",
        f"- Total runs: {len(runs_all)}",
        f"- Robust countries (stable_cluster & full_sharpe>=1.0): {robust_cov['universe'].nunique()}",
        "",
        "## Final 4 configs",
    ]
    if final_cfgs.empty:
        summary_lines.append("- No config met strict robust criteria.")
    else:
        for r in final_cfgs.itertuples(index=False):
            summary_lines.append(
                f"- {r.universe} | {r.family} | z_entry={r.z_entry} z_window={r.z_window} max_hold={r.max_hold} "
                f"| full_sharpe={r.full_sharpe:.2f} | oos_min={r.oos_min_sharpe:.2f} | gap={r.max_gap_abs:.2f} | cluster={int(r.cluster_size)}"
            )

    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Saved campaign outputs in:", OUT_DIR)
    print("Main runs:", OUT_DIR / "main_runs.csv")
    print("Stable clusters:", OUT_DIR / "main_stable_clusters_multi_splits.csv")
    print("All stable clusters:", OUT_DIR / "all_stable_clusters_multi_splits.csv")
    print("Coverage:", OUT_DIR / "coverage_by_country.csv")
    print("Final configs:", OUT_DIR / "final_4_configs.csv")
    print("\nCountry summary:")
    if country_summary.empty:
        print("<empty>")
    else:
        print(country_summary.to_string(index=False))


if __name__ == "__main__":
    main()
