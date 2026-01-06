from __future__ import annotations

import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNER_PATH = PROJECT_ROOT / "data" / "scanner" / "scanner_history.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "universe" / "monthly_universe.parquet"

# ============================================================
# PARAMS (100 % configurables)
# ============================================================

UNIVERSE_NAME = "uk"

# selection rules
ELIGIBILITY_ALLOWED = ["ELIGIBLE"]          # or ["ELIGIBLE", "WATCH"]
TOP_K = 10                                  # number of pairs per month

# optional safety
MIN_PAIRS_REQUIRED = 3                      # skip month if too few pairs

# ============================================================
# BUILD MONTHLY UNIVERSE
# ============================================================

def build_monthly_universe() -> pd.DataFrame:

    df = pd.read_parquet(SCANNER_PATH)

    df["scan_date"] = pd.to_datetime(df["scan_date"])

    df = df[df["universe"] == UNIVERSE_NAME]

    out = []

    for scan_date, df_t in df.groupby("scan_date"):

        df_sel = df_t[
            df_t["eligibility"].isin(ELIGIBILITY_ALLOWED)
        ].sort_values(
            "eligibility_score", ascending=False
        )

        if len(df_sel) < MIN_PAIRS_REQUIRED:
            continue

        df_sel = df_sel.head(TOP_K).copy()
        df_sel["rank"] = range(1, len(df_sel) + 1)

        # trading period = full next month
        trade_start = (scan_date + relativedelta(days=1)).replace(day=1)
        trade_end = trade_start + relativedelta(months=1) - relativedelta(days=1)

        df_sel["trade_month"] = trade_start.strftime("%Y-%m")
        df_sel["trade_start"] = trade_start
        df_sel["trade_end"] = trade_end

        out.append(df_sel)

    if not out:
        return pd.DataFrame()

    df_out = pd.concat(out, ignore_index=True)

    cols = [
        "trade_month",
        "trade_start",
        "trade_end",
        "scan_date",
        "universe",
        "asset_1",
        "asset_2",
        "rank",
        "eligibility",
        "eligibility_score",
        "n_valid_windows",
        "beta_std",
    ]

    return df_out[cols]


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    df_monthly = build_monthly_universe()

    if df_monthly.empty:
        print("No monthly universe produced.")
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_monthly.to_parquet(OUTPUT_PATH, index=False)

        print("Monthly universe saved to:", OUTPUT_PATH)
        print("Months:", df_monthly["trade_month"].nunique())
        print("Total rows:", len(df_monthly))
