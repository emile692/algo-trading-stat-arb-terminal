from pathlib import Path
from typing import List

import pandas as pd

# ============================================================
# Candidate pool (monthly, cross-universe)
# ============================================================


def build_global_month_candidates(
    monthly_universe_dir: Path,
    trade_month: str,
    universes: List[str],
    top_n: int,
) -> pd.DataFrame:

    frames = []
    for u in universes:
        fp = monthly_universe_dir / f"{u}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df = df[df["trade_month"] == trade_month]
        if df.empty:
            continue
        df["universe"] = u
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["a_min"] = df[["asset_1", "asset_2"]].min(axis=1)
    df["a_max"] = df[["asset_1", "asset_2"]].max(axis=1)

    df = (
        df.sort_values("eligibility_score", ascending=False)
        .drop_duplicates(["a_min", "a_max", "trade_month"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return df
