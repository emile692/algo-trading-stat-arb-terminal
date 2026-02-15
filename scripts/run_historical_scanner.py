from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is in PYTHONPATH
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.loader import load_price_csv
from utils.scanner import scan_universe

from config.params import (
    UNIVERSES,
    SCANNER_START_DATE,
    SCANNER_END_DATE,
    SCANNER_FREQ,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = PROJECT_ROOT
SCANNER_DATA_PATH = PROJECT_PATH / "data" / "raw" / "d1"

ASSET_REGISTRY_PATH = PROJECT_PATH / "data" / "asset_registry.csv"
OUTPUT_DIR = PROJECT_PATH / "data" / "scanner"


# ============================================================
# UTILS
# ============================================================

def load_universe_assets(universe: str) -> list[str]:
    reg = pd.read_csv(ASSET_REGISTRY_PATH)
    return (
        reg.loc[reg["category_id"] == universe, "asset"]
        .str.upper()
        .tolist()
    )


def load_price_asof(asset: str, end_date: pd.Timestamp) -> pd.Series | None:
    try:
        df = load_price_csv(asset, SCANNER_DATA_PATH).copy()
    except Exception:
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] <= end_date]

    if len(df) < 100:
        return None

    df["log"] = np.log(df["close"])

    # liquidity / activity filter
    price_diff = df["close"].diff().abs()
    if price_diff.rolling(20).sum().iloc[-1] == 0:
        return None

    # normalized log-price (as-of scan_date)
    df["norm"] = df["log"] - df["log"].iloc[-1]

    return df.set_index("datetime")["norm"]


# ============================================================
# MAIN HISTORICAL SCAN (DAILY)
# ============================================================

def run_historical_scan():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scan_dates = pd.date_range(
        start=SCANNER_START_DATE,
        end=SCANNER_END_DATE,
        freq=SCANNER_FREQ,
    )

    for universe in UNIVERSES:

        assets = sorted(set(load_universe_assets(universe)))

        print(f"\n==============================")
        print(f"Universe: {universe}")
        print(f"Assets in universe: {len(assets)}")

        all_scans: list[pd.DataFrame] = []

        for scan_date in scan_dates:
            print(f"\n=== Scan @ {scan_date.date()} ===")

            series: dict[str, pd.Series] = {}

            for asset in assets:
                s = load_price_asof(asset, scan_date)
                if s is not None:
                    series[asset] = s

            if len(series) < 2:
                continue

            prices = pd.DataFrame(series).dropna(how="all")

            if prices.shape[1] < 2:
                continue

            df_scan = scan_universe(
                price_df=prices,
                universe_name=universe,
            )

            if df_scan.empty:
                continue

            df_scan["scan_date"] = scan_date.normalize()
            all_scans.append(df_scan)

            print(f"Pairs scanned: {len(df_scan)}")

        if not all_scans:
            print(f"No scans produced for {universe}.")
            continue

        df_all = pd.concat(all_scans, ignore_index=True)

        out_path = OUTPUT_DIR / f"{universe}.parquet"
        df_all.to_parquet(out_path, index=False)

        print("\n=== DONE ===")
        print(f"Saved to: {out_path}")
        print(f"Total rows: {len(df_all)}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_historical_scan()
