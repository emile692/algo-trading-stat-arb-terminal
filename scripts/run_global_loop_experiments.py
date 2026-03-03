from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.global_loop import run_global_ranking_daily_portfolio
from object.class_file import BatchConfig, StrategyParams


SCANNER_DIR = PROJECT_ROOT / "data" / "scanner"
OUT_DIR = PROJECT_ROOT / "data" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_sweden_scan_df() -> pd.DataFrame:
    is_path = SCANNER_DIR / "scans_is.parquet"
    oos_path = SCANNER_DIR / "scans_oos.parquet"
    if not is_path.exists() or not oos_path.exists():
        raise FileNotFoundError(
            "Missing scans_is.parquet / scans_oos.parquet in data/scanner. "
            "Generate them from notebook pipeline first."
        )
    is_df = pd.read_parquet(is_path)
    oos_df = pd.read_parquet(oos_path)
    return pd.concat([is_df, oos_df], ignore_index=True)


def _eligible_density(scans: pd.DataFrame) -> dict[str, float]:
    d = scans.copy()
    d["scan_date"] = pd.to_datetime(d["scan_date"]).dt.normalize()
    d["eligibility"] = d["eligibility"].astype(str).str.upper()
    by_date = d[d["eligibility"] == "ELIGIBLE"].groupby("scan_date").size()
    return {
        "n_scan_dates": int(d["scan_date"].nunique()),
        "avg_eligible_per_scan_date": float(by_date.mean()) if not by_date.empty else 0.0,
        "median_eligible_per_scan_date": float(by_date.median()) if not by_date.empty else 0.0,
        "max_eligible_per_scan_date": int(by_date.max()) if not by_date.empty else 0,
    }


def _run_case(
    case_name: str,
    cfg: BatchConfig,
    scans: pd.DataFrame,
    params: StrategyParams,
    universes: list[str],
) -> dict[str, Any]:
    res = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=universes,
        scans=scans,
    )
    if not res:
        return {"case": case_name, "ok": False}

    stats = res["stats"]
    eq = res["equity"]
    trades = res["trades"]
    closed = trades[trades["exit_datetime"].notna()].copy() if not trades.empty else trades
    return {
        "case": case_name,
        "ok": True,
        "signal_space": stats.get("Signal space"),
        "selection_mode": stats.get("Selection mode"),
        "selection_labels": stats.get("Selection labels"),
        "final_equity": stats.get("Final Equity"),
        "sharpe": stats.get("Sharpe"),
        "cagr": stats.get("CAGR"),
        "max_drawdown": stats.get("Max Drawdown"),
        "nb_trades": stats.get("Nb Trades"),
        "closed_trades": int(len(closed)) if isinstance(closed, pd.DataFrame) else 0,
        "hit_ratio": float((closed["trade_return"] > 0).mean()) if len(closed) > 0 else np.nan,
        "avg_trade_return": float(closed["trade_return"].mean()) if len(closed) > 0 else np.nan,
        "avg_abs_entry_z": float(closed["entry_z"].abs().mean()) if len(closed) > 0 else np.nan,
        "avg_open_positions": float(eq["n_open_positions"].mean()) if not eq.empty else np.nan,
        "pct_days_with_positions": float((eq["n_open_positions"] > 0).mean()) if not eq.empty else np.nan,
    }


def main() -> None:
    scans = _load_sweden_scan_df()
    universes = ["sweden"]
    density = _eligible_density(scans)

    cfg_full = BatchConfig(
        data_path=PROJECT_ROOT / "data" / "raw" / "d1",
        start_date="2022-12-31",
        end_date="2025-12-31",
    )
    cfg_is = BatchConfig(
        data_path=PROJECT_ROOT / "data" / "raw" / "d1",
        start_date="2022-12-31",
        end_date="2024-12-31",
    )
    cfg_oos = BatchConfig(
        data_path=PROJECT_ROOT / "data" / "raw" / "d1",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    scans_is = scans[pd.to_datetime(scans["scan_date"]) <= pd.Timestamp("2024-12-31")].copy()
    scans_oos = scans[pd.to_datetime(scans["scan_date"]) >= pd.Timestamp("2025-01-01")].copy()

    base = dict(
        beta_mode="static",
        fees=0.0002,
        top_n_candidates=20,
        max_positions=5,
    )

    cases = [
        (
            "raw_baseline_legacy",
            cfg_full,
            scans,
            StrategyParams(
                **base,
                z_entry=1.5,
                z_exit=0.375,
                z_stop=3.0,
                z_window=60,
                max_holding_days=30,
                signal_space="raw",
            ),
        ),
        (
            "idio_baseline_legacy",
            cfg_full,
            scans,
            StrategyParams(
                **base,
                z_entry=1.5,
                z_exit=0.375,
                z_stop=3.0,
                z_window=60,
                max_holding_days=30,
                signal_space="idio_pca",
                pca_signal_window=252,
                pca_signal_components=3,
                pca_signal_min_assets=10,
            ),
        ),
        (
            "raw_best_param_legacy",
            cfg_full,
            scans,
            StrategyParams(
                **base,
                z_entry=1.7,
                z_exit=0.425,
                z_stop=3.74,
                z_window=75,
                max_holding_days=30,
                signal_space="raw",
            ),
        ),
        (
            "idio_best_param_legacy",
            cfg_full,
            scans,
            StrategyParams(
                **base,
                z_entry=1.7,
                z_exit=0.425,
                z_stop=3.06,
                z_window=90,
                max_holding_days=20,
                signal_space="idio_pca",
                pca_signal_window=252,
                pca_signal_components=3,
                pca_signal_min_assets=10,
            ),
        ),
        (
            "idio_best_param_composite",
            cfg_full,
            scans,
            StrategyParams(
                **base,
                z_entry=1.7,
                z_exit=0.425,
                z_stop=3.06,
                z_window=90,
                max_holding_days=20,
                signal_space="idio_pca",
                selection_mode="composite_quality",
                pca_signal_window=252,
                pca_signal_components=3,
                pca_signal_min_assets=10,
            ),
        ),
        (
            "idio_best_param_composite_filters",
            cfg_full,
            scans,
            StrategyParams(
                **base,
                z_entry=1.7,
                z_exit=0.425,
                z_stop=3.06,
                z_window=90,
                max_holding_days=20,
                signal_space="idio_pca",
                selection_mode="composite_quality",
                min_corr_12m=0.75,
                max_half_life_6m=45.0,
                max_beta_std=0.22,
                min_spread_std_6m=0.010,
                min_n_valid_windows=2,
                pca_signal_window=252,
                pca_signal_components=3,
                pca_signal_min_assets=10,
            ),
        ),
        (
            "idio_best_param_IS",
            cfg_is,
            scans_is,
            StrategyParams(
                **base,
                z_entry=1.7,
                z_exit=0.425,
                z_stop=3.06,
                z_window=90,
                max_holding_days=20,
                signal_space="idio_pca",
                pca_signal_window=252,
                pca_signal_components=3,
                pca_signal_min_assets=10,
            ),
        ),
        (
            "idio_best_param_OOS",
            cfg_oos,
            scans_oos,
            StrategyParams(
                **base,
                z_entry=1.7,
                z_exit=0.425,
                z_stop=3.06,
                z_window=90,
                max_holding_days=20,
                signal_space="idio_pca",
                pca_signal_window=252,
                pca_signal_components=3,
                pca_signal_min_assets=10,
            ),
        ),
    ]

    rows = [_run_case(name, cfg, scans_i, params, universes) for (name, cfg, scans_i, params) in cases]
    out = pd.DataFrame(rows)
    out = out.sort_values("case")

    csv_path = OUT_DIR / "global_loop_experiment_log.csv"
    out.to_csv(csv_path, index=False)

    summary = {
        "scan_density": density,
        "output_csv": str(csv_path),
    }
    json_path = OUT_DIR / "global_loop_experiment_summary.json"
    pd.Series(summary).to_json(json_path, indent=2)

    print("Scan density:", density)
    print(out.to_string(index=False))
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
