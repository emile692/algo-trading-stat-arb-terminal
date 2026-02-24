from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 3,
    anchored: bool = False,
) -> list[WalkForwardSplit]:
    """Create chronological IS/OOS splits without lookahead.

    Anchored mode keeps train_start fixed and expands train_end.
    Rolling mode moves both train and test windows.
    """
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train_months/test_months/step_months must be > 0")

    splits: list[WalkForwardSplit] = []
    cursor = start

    while True:
        train_start = start if anchored else cursor
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_end > end:
            break

        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        cursor = cursor + pd.DateOffset(months=step_months)

    return splits


def pair_selection_stability(scans: pd.DataFrame, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pair-level and date-level stability diagnostics from scan output."""
    if scans.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = scans.copy()
    df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.normalize()
    df["pair_id"] = (
        df["asset_1"].astype(str).str.upper() + "_" + df["asset_2"].astype(str).str.upper()
    )

    ranked = (
        df[df["eligibility"] == "ELIGIBLE"]
        .sort_values(["scan_date", "eligibility_score"], ascending=[True, False])
        .groupby("scan_date", as_index=False)
        .head(top_n)
    )

    if ranked.empty:
        return pd.DataFrame(), pd.DataFrame()

    pair_stats = (
        ranked.groupby("pair_id", as_index=False)
        .agg(
            selected_count=("scan_date", "size"),
            first_seen=("scan_date", "min"),
            last_seen=("scan_date", "max"),
            avg_rank_score=("eligibility_score", "mean"),
        )
        .sort_values("selected_count", ascending=False)
    )

    by_date = (
        ranked.groupby("scan_date")["pair_id"]
        .agg(lambda s: list(s))
        .to_frame("pairs")
        .sort_index()
    )
    by_date["n_pairs"] = by_date["pairs"].str.len()

    prev = None
    jaccards = []
    for dt, row in by_date.iterrows():
        cur = set(row["pairs"])
        if prev is None:
            jaccards.append(np.nan)
        else:
            union = prev | cur
            jaccards.append((len(prev & cur) / len(union)) if union else np.nan)
        prev = cur

    by_date["topn_jaccard_vs_prev"] = jaccards
    return pair_stats, by_date.reset_index()


def edge_decomposition(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate trade-level edge by pair/side/time buckets."""
    if trades.empty:
        return {
            "by_pair": pd.DataFrame(),
            "by_side": pd.DataFrame(),
            "by_year_month": pd.DataFrame(),
        }

    t = trades.copy()
    t["entry_datetime"] = pd.to_datetime(t["entry_datetime"])
    t["trade_return"] = pd.to_numeric(t.get("trade_return"), errors="coerce")
    t["trade_return_isolated"] = pd.to_numeric(t.get("trade_return_isolated"), errors="coerce")

    metric = "trade_return_isolated" if t["trade_return_isolated"].notna().any() else "trade_return"

    by_pair = (
        t.groupby("pair_id", as_index=False)[metric]
        .agg(["count", "mean", "median", "std", "sum"])
        .reset_index()
        .rename(columns={"index": "pair_id"})
        .sort_values("sum", ascending=False)
    )

    by_side = t.groupby("side", as_index=False)[metric].agg(["count", "mean", "median", "std", "sum"]).reset_index()

    t["year_month"] = t["entry_datetime"].dt.to_period("M").astype(str)
    by_year_month = t.groupby("year_month", as_index=False)[metric].agg(["count", "mean", "sum", "std"]).reset_index()

    return {
        "by_pair": by_pair,
        "by_side": by_side,
        "by_year_month": by_year_month,
    }
