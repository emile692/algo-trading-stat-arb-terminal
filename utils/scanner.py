# utils/scanner.py
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict

from config.params import SCAN_LOOKBACKS, SCANNER_THRESHOLDS, SCANNER_WEIGHTS
from utils.metrics import (
    compute_hedge_ratio,
    compute_spread,
    compute_adf,
    compute_coint,
    compute_corr,
    compute_half_life,
)


# ============================================================
# Scan sur UNE fenêtre
# ============================================================

def scan_pair_window(
    y: pd.Series,
    x: pd.Series,
    lookback: int,
) -> Dict[str, float]:

    y = y.dropna()
    x = x.dropna()

    df = pd.concat([y, x], axis=1).dropna()
    df = df.iloc[-lookback:]

    if len(df) < max(30, int(0.8 * lookback)):
        raise ValueError("Not enough data")

    yv = df.iloc[:, 0]
    xv = df.iloc[:, 1]

    beta = compute_hedge_ratio(yv, xv)
    spread = compute_spread(yv, xv, beta)

    adf_t, adf_p, _ = compute_adf(spread)
    eg_t, eg_p, _ = compute_coint(yv, xv)
    corr = compute_corr(yv, xv)
    hl = compute_half_life(spread)

    return {
        "beta": beta,
        "corr": corr,
        "adf_p": adf_p,
        "eg_p": eg_p,
        "half_life": hl,
        "spread_std": float(np.nanstd(spread)),
    }


# ============================================================
# Validation d'une fenêtre
# ============================================================

def window_is_valid(res: Dict[str, float]) -> bool:
    if res is None:
        return False
    if res["half_life"] is None:
        return False

    return (
        res["eg_p"] < SCANNER_THRESHOLDS["eg_p_max"]
        and res["adf_p"] < SCANNER_THRESHOLDS["adf_p_max"]
        and res["half_life"] < SCANNER_THRESHOLDS["half_life_max"]
        and abs(res["corr"]) > SCANNER_THRESHOLDS["corr_min"]
    )


# ============================================================
# Scan multi-fenêtres (CORE)
# ============================================================

def scan_pair_multi_window(
    y: pd.Series,
    x: pd.Series,
) -> Dict[str, float]:

    window_results = {}

    for label, lb in SCAN_LOOKBACKS.items():
        try:
            window_results[label] = scan_pair_window(y, x, lb)
        except Exception:
            window_results[label] = None

    valid_windows = [k for k, v in window_results.items() if window_is_valid(v)]
    n_valid = len(valid_windows)

    if n_valid >= 2:
        eligibility = "ELIGIBLE"
    elif n_valid == 1:
        eligibility = "WATCH"
    else:
        eligibility = "OUT"

    betas = [
        v["beta"]
        for v in window_results.values()
        if v is not None and np.isfinite(v["beta"])
    ]
    beta_std = float(np.std(betas)) if len(betas) >= 2 else np.nan

    score = (
        SCANNER_WEIGHTS["n_valid"] * n_valid
        + SCANNER_WEIGHTS["corr_12m"] * (window_results.get("12m", {}) or {}).get("corr", 0.0)
        + SCANNER_WEIGHTS["half_life_6m"] * (window_results.get("6m", {}) or {}).get("half_life", 0.0)
        + SCANNER_WEIGHTS["beta_stability"] * (beta_std if np.isfinite(beta_std) else 0.0)
    )

    out = {
        "eligibility": eligibility,
        "eligibility_score": float(score),
        "n_valid_windows": n_valid,
        "beta_std": beta_std,
    }

    for w, res in window_results.items():
        if res is None:
            continue
        for k, v in res.items():
            out[f"{w}_{k}"] = v

    return out


# ============================================================
# Scan d'un univers
# ============================================================

def scan_universe(
    price_df: pd.DataFrame,
    universe_name: str,
) -> pd.DataFrame:

    results = []

    for a1, a2 in combinations(price_df.columns, 2):
        try:
            res = scan_pair_multi_window(price_df[a1], price_df[a2])
            res.update({
                "asset_1": a1,
                "asset_2": a2,
                "universe": universe_name,
            })
            results.append(res)
        except Exception:
            continue

    df = pd.DataFrame(results)
    df["scan_date"] = pd.Timestamp.utcnow().normalize()
    return df


def scan_all_universes(
    universe_to_prices: dict[str, pd.DataFrame],
    max_workers: int | None = None,
) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    dfs = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(scan_universe, prices, univ): univ
            for univ, prices in universe_to_prices.items()
            if prices.shape[1] >= 2
        }

        for fut in as_completed(futures):
            res = fut.result()
            if res is not None and not res.empty:
                dfs.append(res)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
