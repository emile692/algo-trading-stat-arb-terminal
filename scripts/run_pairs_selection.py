from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ============================================================
# PAIRS SELECTION — Rebalance-aware (daily / weekly / monthly)
# ============================================================

import pandas as pd
from dateutil.relativedelta import relativedelta

from object.class_file import StrategyParams


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SCANNER_DIR = PROJECT_ROOT / "data" / "scanner"
OUTPUT_DIR = PROJECT_ROOT / "data" / "universe"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def compute_rebalance_window(scan_date: pd.Timestamp, params: StrategyParams):
    """
    Given a scan_date, return:
    - rebalance_id
    - trade_start
    - trade_end (inclusive, data-driven upper bound)
    """

    if params.rebalance_period == "daily":
        trade_start = scan_date
        trade_end = scan_date
        rebalance_id = scan_date.strftime("%Y-%m-%d")

    elif params.rebalance_period == "weekly":
        # next trading window starts the day AFTER scan_date
        trade_start = scan_date + pd.Timedelta(days=1)
        trade_end = trade_start + pd.Timedelta(days=6)
        rebalance_id = trade_start.strftime("%Y-W%U")

    elif params.rebalance_period == "monthly":
        trade_start = (scan_date + pd.Timedelta(days=1)).replace(day=1)
        trade_end = trade_start + relativedelta(months=1) - pd.Timedelta(days=1)
        rebalance_id = trade_start.strftime("%Y-%m")

    else:
        raise ValueError(f"Unknown rebalance_period: {params.rebalance_period}")

    return rebalance_id, trade_start, trade_end


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def build_rebalance_universes(params: StrategyParams):
    """
    For each scan_date:
    - select top-K eligible pairs
    - build a rebalance window (daily / weekly / monthly)
    - write one parquet per rebalance_id
    """

    scanner_files = sorted(SCANNER_DIR.glob("*.parquet"))

    if not scanner_files:
        raise RuntimeError("No scanner parquet files found.")

    for path in scanner_files:
        df = pd.read_parquet(path)

        if df.empty:
            continue

        # Ensure timestamp
        df["scan_date"] = pd.to_datetime(df["scan_date"])

        df = pd.read_parquet(path)
        df["scan_date"] = pd.to_datetime(df["scan_date"])

        for scan_date, df_day in df.groupby("scan_date"):

            eligible = df_day[df_day["eligibility"] == "ELIGIBLE"].copy()

            if eligible.empty:
                continue

            eligible = eligible.sort_values(
                "eligibility_score", ascending=False
            ).head(params.max_positions)

            rebalance_id, trade_start, trade_end = compute_rebalance_window(
                scan_date, params
            )

            out = eligible.copy()
            out["rebalance_id"] = rebalance_id
            out["scan_date"] = scan_date
            out["trade_start"] = trade_start
            out["trade_end"] = trade_end
            out["rebalance_period"] = params.rebalance_period

            out_path = OUTPUT_DIR / f"universe_{rebalance_id}.parquet"
            out.to_parquet(out_path, index=False)

            print(
                f"[OK] {rebalance_id} | "
                f"{params.rebalance_period} | "
                f"{len(out)} pairs | "
                f"scan={scan_date.date()} → trade=[{trade_start.date()} → {trade_end.date()}]"
            )

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    params = StrategyParams()
    build_rebalance_universes(params)
