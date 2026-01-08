from __future__ import annotations

import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta

from config.params import (
    UNIVERSES,
    MONTHLY_UNIVERSE_CONFIG,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCANNER_DIR = PROJECT_ROOT / "data" / "scanner"
OUTPUT_DIR = PROJECT_ROOT / "data" / "universe"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CORE
# ============================================================

def build_monthly_universe_for_universe(universe: str) -> pd.DataFrame:

    scanner_path = SCANNER_DIR / f"{universe}.parquet"

    if not scanner_path.exists():
        print(f"[WARN] Scanner file not found for {universe}")
        return pd.DataFrame()

    df = pd.read_parquet(scanner_path)
    df["scan_date"] = pd.to_datetime(df["scan_date"])

    cfg = MONTHLY_UNIVERSE_CONFIG

    out = []

    for scan_date, df_t in df.groupby("scan_date"):

        df_sel = (
            df_t[df_t["eligibility"].isin(cfg["eligibility_allowed"])]
            .sort_values("eligibility_score", ascending=False)
        )

        if len(df_sel) < cfg["min_pairs_required"]:
            continue

        df_sel = df_sel.head(cfg["top_k"]).copy()
        df_sel["rank"] = range(1, len(df_sel) + 1)

        # trading period = full next month
        trade_start = (scan_date + relativedelta(days=1)).replace(day=1)
        trade_end = trade_start + relativedelta(months=1) - relativedelta(days=1)

        df_sel["trade_month"] = trade_start.strftime("%Y-%m")
        df_sel["trade_start"] = trade_start
        df_sel["trade_end"] = trade_end
        df_sel["universe"] = universe

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


def build_all_monthly_universes():

    for universe in UNIVERSES:
        print(f"\n=== Building monthly universe: {universe} ===")

        df_u = build_monthly_universe_for_universe(universe)

        if df_u.empty:
            print("No monthly universe produced.")
            continue

        out_path = OUTPUT_DIR / f"{universe}.parquet"
        df_u.to_parquet(out_path, index=False)

        print("Saved to:", out_path)
        print("Months:", df_u["trade_month"].nunique())
        print("Total rows:", len(df_u))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_all_monthly_universes()
