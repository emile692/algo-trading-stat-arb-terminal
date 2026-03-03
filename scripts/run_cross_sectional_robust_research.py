from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.global_loop import run_global_ranking_daily_portfolio
from object.class_file import BatchConfig, StrategyParams


# Validation windows imposed by user request.
IS_START = "2019-12-31"
IS_END = "2024-12-31"
OOS_START = "2024-01-01"
OOS_END = "2025-12-31"

UNIVERSES = ["france", "sweden", "italy"]

BASE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
SCANNER_DIR = PROJECT_ROOT / "data" / "scanner"
OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "robust_cross_sectional"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    signal_space: str
    selection_mode: str
    max_pairs_per_asset: int = 0
    min_corr_12m: float | None = None
    max_half_life_6m: float | None = None
    max_beta_std: float | None = None
    min_n_valid_windows: int | None = None


def linked_thresholds(z_entry: float) -> tuple[float, float]:
    """
    Parsimonious linkage to reduce degrees of freedom:
    - z_exit = z_entry / 3
    - z_stop = 2 * z_entry
    """
    return (round(float(z_entry) / 3.0, 4), round(2.0 * float(z_entry), 4))


def load_scan_universe(universe: str) -> pd.DataFrame:
    path = SCANNER_DIR / f"{universe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing scanner file: {path}")
    df = pd.read_parquet(path).copy()
    df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.normalize()
    df["asset_1"] = df["asset_1"].astype(str).str.upper()
    df["asset_2"] = df["asset_2"].astype(str).str.upper()
    df["eligibility"] = df["eligibility"].astype(str).str.upper()
    return df


def run_one(
    universe: str,
    segment: str,
    scans: pd.DataFrame,
    family: FamilyConfig,
    z_entry: float,
    z_window: int,
    max_hold: int,
) -> dict[str, Any]:
    z_exit, z_stop = linked_thresholds(z_entry)
    cfg = BatchConfig(
        data_path=BASE_DATA_PATH,
        start_date=IS_START if segment == "IS" else OOS_START,
        end_date=IS_END if segment == "IS" else OOS_END,
    )
    params = StrategyParams(
        z_entry=float(z_entry),
        z_exit=float(z_exit),
        z_stop=float(z_stop),
        z_window=int(z_window),
        beta_mode="static",
        fees=0.0002,
        top_n_candidates=20,
        max_positions=5,
        max_holding_days=int(max_hold),
        signal_space=family.signal_space,
        selection_mode=family.selection_mode,
        max_pairs_per_asset=int(family.max_pairs_per_asset),
        min_corr_12m=family.min_corr_12m,
        max_half_life_6m=family.max_half_life_6m,
        max_beta_std=family.max_beta_std,
        min_n_valid_windows=family.min_n_valid_windows,
        pca_signal_window=252,
        pca_signal_components=3,
        pca_signal_min_assets=10,
    )

    res = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=[universe],
        scans=scans,
    )
    if not res:
        return {
            "universe": universe,
            "segment": segment,
            "family": family.name,
            "z_entry": z_entry,
            "z_window": z_window,
            "max_hold": max_hold,
            "ok": False,
        }

    st = res["stats"]
    tr = res["trades"]
    tr = tr[tr["exit_datetime"].notna()].copy() if not tr.empty else tr
    return {
        "universe": universe,
        "segment": segment,
        "family": family.name,
        "signal_space": family.signal_space,
        "selection_mode": family.selection_mode,
        "z_entry": float(z_entry),
        "z_exit": float(z_exit),
        "z_stop": float(z_stop),
        "z_window": int(z_window),
        "max_hold": int(max_hold),
        "ok": True,
        "final_equity": st.get("Final Equity"),
        "sharpe": st.get("Sharpe"),
        "cagr": st.get("CAGR"),
        "max_drawdown": st.get("Max Drawdown"),
        "nb_trades": st.get("Nb Trades"),
        "closed_trades": int(len(tr)) if isinstance(tr, pd.DataFrame) else 0,
        "hit_ratio": float((tr["trade_return"] > 0).mean()) if len(tr) > 0 else np.nan,
        "avg_trade_return": float(tr["trade_return"].mean()) if len(tr) > 0 else np.nan,
    }


def build_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    for universe in sorted(df["universe"].unique()):
        for family in sorted(df["family"].unique()):
            for segment in ("IS", "OOS"):
                d = df[
                    (df["universe"] == universe)
                    & (df["family"] == family)
                    & (df["segment"] == segment)
                    & (df["ok"] == True)
                ].copy()
                if d.empty:
                    continue

                # Median Sharpe across max_hold to visualize robust zones.
                piv = (
                    d.groupby(["z_window", "z_entry"], as_index=False)["sharpe"]
                    .median()
                    .pivot(index="z_window", columns="z_entry", values="sharpe")
                    .sort_index()
                )
                piv.to_csv(out_dir / f"heatmap_{universe}_{family}_{segment}.csv")

                plt.figure(figsize=(6, 4))
                sns.heatmap(
                    piv,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn",
                    center=0.8,
                    vmin=-0.5,
                    vmax=1.5,
                )
                plt.title(f"{universe} | {family} | {segment} | median Sharpe over max_hold")
                plt.tight_layout()
                plt.savefig(out_dir / f"heatmap_{universe}_{family}_{segment}.png", dpi=140)
                plt.close()


def summarize_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["ok"] == True)].copy()
    if d.empty:
        return pd.DataFrame()

    grp = (
        d.groupby(["universe", "family", "segment"], as_index=False)
        .agg(
            sharpe=("sharpe", "median"),
            cagr=("cagr", "median"),
            max_drawdown=("max_drawdown", "median"),
            nb_trades=("nb_trades", "median"),
            hit_ratio=("hit_ratio", "median"),
            n_runs=("sharpe", "size"),
        )
    )

    # Stability proxy: percentage of runs meeting Sharpe >= 0.8.
    stable = (
        d.assign(pass_sharpe=d["sharpe"] >= 0.8)
        .groupby(["universe", "family", "segment"], as_index=False)["pass_sharpe"]
        .mean()
        .rename(columns={"pass_sharpe": "pct_runs_sharpe_ge_0_8"})
    )
    out = grp.merge(stable, on=["universe", "family", "segment"], how="left")
    return out.sort_values(["universe", "family", "segment"])


def summarize_stable_clusters(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["ok"] == True].copy()
    if d.empty:
        return pd.DataFrame()

    key = ["universe", "family", "z_entry", "z_window", "max_hold"]
    piv = d.pivot_table(index=key, columns="segment", values="sharpe", aggfunc="median").reset_index()
    if "IS" not in piv.columns:
        piv["IS"] = np.nan
    if "OOS" not in piv.columns:
        piv["OOS"] = np.nan
    piv["gap_abs"] = (piv["IS"] - piv["OOS"]).abs()
    piv["stable_zone"] = (piv["IS"] >= 0.8) & (piv["OOS"] >= 0.8) & (piv["gap_abs"] <= 0.35)
    return piv.sort_values(["universe", "family", "stable_zone", "IS", "OOS"], ascending=[True, True, False, False, False])


def main() -> None:
    # Coarse, rounded grids only (parsimonious and industrializable).
    z_entry_grid = [1.2, 1.5, 1.8]
    z_window_grid = [40, 60, 80]
    max_hold_grid = [15, 30]

    families = [
        FamilyConfig(name="raw_legacy", signal_space="raw", selection_mode="legacy"),
        FamilyConfig(name="idio_legacy", signal_space="idio_pca", selection_mode="legacy"),
        FamilyConfig(name="raw_composite", signal_space="raw", selection_mode="composite_quality"),
        FamilyConfig(name="idio_composite", signal_space="idio_pca", selection_mode="composite_quality"),
    ]

    rows: list[dict[str, Any]] = []
    exp_log: list[dict[str, Any]] = []
    hypothesis_counter = 0

    runs_path_live = OUT_DIR / "experiment_runs_live.csv"
    if runs_path_live.exists():
        runs_path_live.unlink()

    for universe in UNIVERSES:
        scans_all = load_scan_universe(universe)
        scans_is = scans_all[
            (scans_all["scan_date"] >= pd.Timestamp(IS_START))
            & (scans_all["scan_date"] <= pd.Timestamp(IS_END))
        ].copy()
        scans_oos = scans_all[
            (scans_all["scan_date"] >= pd.Timestamp(OOS_START))
            & (scans_all["scan_date"] <= pd.Timestamp(OOS_END))
        ].copy()

        exp_log.append(
            {
                "type": "dataset",
                "universe": universe,
                "scan_rows_is": int(len(scans_is)),
                "scan_rows_oos": int(len(scans_oos)),
                "scan_dates_is": int(scans_is["scan_date"].nunique()) if not scans_is.empty else 0,
                "scan_dates_oos": int(scans_oos["scan_date"].nunique()) if not scans_oos.empty else 0,
            }
        )

        for family in families:
            hypothesis_counter += 1
            hypothesis_id = f"H{hypothesis_counter:02d}"
            exp_log.append(
                {
                    "type": "hypothesis",
                    "hypothesis_id": hypothesis_id,
                    "universe": universe,
                    "family": family.name,
                    "description": (
                        "Evaluate parsimonious linked-threshold strategy on coarse grid "
                        "and measure IS/OOS stability."
                    ),
                }
            )
            for z_entry in z_entry_grid:
                for z_window in z_window_grid:
                    for max_hold in max_hold_grid:
                        rows.append(run_one(universe, "IS", scans_is, family, z_entry, z_window, max_hold))
                        rows.append(run_one(universe, "OOS", scans_oos, family, z_entry, z_window, max_hold))
            # checkpoint per family
            pd.DataFrame(rows).to_csv(runs_path_live, index=False)

    results = pd.DataFrame(rows)
    results_path = OUT_DIR / "experiment_runs.csv"
    results.to_csv(results_path, index=False)

    exp_log_df = pd.DataFrame(exp_log)
    exp_log_path = OUT_DIR / "experiment_journal.csv"
    exp_log_df.to_csv(exp_log_path, index=False)

    stability = summarize_stable_clusters(results)
    stability_path = OUT_DIR / "stability_clusters.csv"
    stability.to_csv(stability_path, index=False)

    cross = summarize_cross_section(results)
    cross_path = OUT_DIR / "cross_section_summary.csv"
    cross.to_csv(cross_path, index=False)

    # Winner table per (universe, segment, family) for quick review.
    ok = results[results["ok"] == True].copy()
    winners = (
        ok.sort_values("sharpe", ascending=False)
        .groupby(["universe", "segment", "family"], as_index=False)
        .first()
    )
    winners_path = OUT_DIR / "top_params_by_universe_segment_family.csv"
    winners.to_csv(winners_path, index=False)

    build_heatmaps(results, OUT_DIR / "heatmaps")

    # Optional extra cross-sectional check: Germany on the top 2 OOS families from core universes.
    germany_rows: list[dict[str, Any]] = []
    try:
        g_scans = load_scan_universe("germany")
        g_is = g_scans[(g_scans["scan_date"] >= pd.Timestamp(IS_START)) & (g_scans["scan_date"] <= pd.Timestamp(IS_END))].copy()
        g_oos = g_scans[(g_scans["scan_date"] >= pd.Timestamp(OOS_START)) & (g_scans["scan_date"] <= pd.Timestamp(OOS_END))].copy()
        top_fams = (
            winners[(winners["segment"] == "OOS")]
            .groupby("family", as_index=False)["sharpe"].median()
            .sort_values("sharpe", ascending=False)
            .head(2)["family"]
            .tolist()
        )
        fam_map = {f.name: f for f in families}
        for fam_name in top_fams:
            row = winners[(winners["segment"] == "OOS") & (winners["family"] == fam_name)].sort_values("sharpe", ascending=False).head(1)
            if row.empty:
                continue
            z_entry = float(row["z_entry"].iloc[0])
            z_window = int(row["z_window"].iloc[0])
            max_hold = int(row["max_hold"].iloc[0])
            fam = fam_map[fam_name]
            germany_rows.append(run_one("germany", "IS", g_is, fam, z_entry, z_window, max_hold))
            germany_rows.append(run_one("germany", "OOS", g_oos, fam, z_entry, z_window, max_hold))
    except Exception as exc:
        germany_rows.append({"universe": "germany", "ok": False, "error": str(exc)})

    germany_path = OUT_DIR / "germany_targeted_validation.csv"
    pd.DataFrame(germany_rows).to_csv(germany_path, index=False)

    print("Saved:", results_path)
    print("Saved:", exp_log_path)
    print("Saved:", stability_path)
    print("Saved:", cross_path)
    print("Saved:", winners_path)
    print("Saved:", germany_path)
    print("Rows:", len(results))
    print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
