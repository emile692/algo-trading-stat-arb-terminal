from __future__ import annotations

import datetime
from pathlib import Path
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


# ============================================================
# PATHS (alignés avec ton projet)
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parents[1]   # racine projet
BASE_DATA_PATH = PROJECT_PATH / "data" / "raw"
SCANNER_DATA_PATH = BASE_DATA_PATH / "d1"

ASSET_REGISTRY_PATH = PROJECT_PATH / "data" / "asset_registry.csv"
OUTPUT_PATH = PROJECT_PATH / "data" / "scanner" / "scanner_history.parquet"


# ============================================================
# CONFIG
# ============================================================

UNIVERSE_NAME = "sweden"
START_DATE = datetime.datetime(year = 2025, month=1, day=1)
END_DATE = datetime.datetime(year = 2026, month=1, day=1)
FREQ = "ME"


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
    """
    Charge les données Dukascopy d'un asset
    et les coupe en as-of end_date (pas de fuite).
    """
    try:
        df = load_price_csv(asset, SCANNER_DATA_PATH).copy()
    except Exception:
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] <= end_date]

    if len(df) < 100:
        return None

    df["log"] = np.log(df["close"])
    df["norm"] = df["log"] - df["log"].iloc[0]

    return df.set_index("datetime")["norm"]


# ============================================================
# MAIN HISTORICAL SCAN
# ============================================================

def run_historical_scan():

    assets = sorted(set(load_universe_assets(UNIVERSE_NAME)))

    print(f"Universe: {UNIVERSE_NAME}")
    print(f"Assets in universe: {len(assets)}")

    scan_dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq=FREQ,
    )

    all_scans: list[pd.DataFrame] = []

    for scan_date in scan_dates:
        print(f"\n=== Scan @ {scan_date.date()} ===")

        series: dict[str, pd.Series] = {}

        for asset in assets:
            s = load_price_asof(asset, scan_date)
            if s is not None:
                series[asset] = s

        if len(series) < 2:
            print("Not enough assets with data, skipping.")
            continue

        prices = pd.DataFrame(series).dropna(how="all")

        if prices.shape[1] < 2:
            print("No aligned data, skipping.")
            continue

        df_scan = scan_universe(
            price_df=prices,
            universe_name=UNIVERSE_NAME,
        )

        if df_scan.empty:
            print("No pairs found.")
            continue

        df_scan["scan_date"] = scan_date.normalize()
        all_scans.append(df_scan)

        print(f"Pairs scanned: {len(df_scan)}")

    if not all_scans:
        print("No scans produced.")
        return

    df_all = pd.concat(all_scans, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(OUTPUT_PATH, index=False)

    print("\n=== DONE ===")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df_all)}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_historical_scan()
