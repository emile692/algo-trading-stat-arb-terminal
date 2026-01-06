from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
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
# Helpers – numerical sanity
# ============================================================

def _is_valid_series(
    y: pd.Series,
    x: pd.Series,
    min_len: int = 50,
    min_std: float = 1e-8,
) -> bool:
    """
    Reject degenerate series before any stat.
    """
    if y is None or x is None:
        return False
    if len(y) < min_len or len(x) < min_len:
        return False
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        return False
    if y.std() < min_std or x.std() < min_std:
        return False
    return True


# ============================================================
# Scan sur UNE fenêtre
# ============================================================

def scan_pair_window(
    y: pd.Series,
    x: pd.Series,
    lookback: int,
) -> Dict[str, float]:

    # Align + slice
    df = pd.concat([y, x], axis=1).dropna()
    df = df.iloc[-lookback:]

    if len(df) < max(30, int(0.8 * lookback)):
        raise ValueError("Not enough data")

    yv = df.iloc[:, 0]
    xv = df.iloc[:, 1]

    if not _is_valid_series(yv, xv):
        raise ValueError("Degenerate series")

    # All heavy stats guarded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        beta = compute_hedge_ratio(yv, xv)
        if not np.isfinite(beta):
            raise ValueError("Invalid beta")

        spread = compute_spread(yv, xv, beta)

        corr = compute_corr(yv, xv)
        if not np.isfinite(corr):
            raise ValueError("Invalid corr")

        adf_t, adf_p, _ = compute_adf(spread)
        eg_t, eg_p, _ = compute_coint(yv, xv)
        hl = compute_half_life(spread)

    if hl is None or not np.isfinite(hl):
        raise ValueError("Invalid half-life")

    # reject flat price windows
    if yv.diff().abs().sum() < 1e-6 or xv.diff().abs().sum() < 1e-6:
        raise ValueError("Flat price window")

    return {
        "beta": float(beta),
        "corr": float(corr),
        "adf_p": float(adf_p),
        "eg_p": float(eg_p),
        "half_life": float(hl),
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

    window_results: Dict[str, Dict | None] = {}

    for label, lb in SCAN_LOOKBACKS.items():
        try:
            window_results[label] = scan_pair_window(y, x, lb)
        except Exception:
            window_results[label] = None

    valid_windows = [k for k, v in window_results.items() if window_is_valid(v)]
    n_valid = len(valid_windows)

    # if no window usable, drop the pair entirely
    if n_valid == 0:
        raise ValueError("No valid window")

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
        + SCANNER_WEIGHTS["corr_12m"]
        * (window_results.get("12m") or {}).get("corr", 0.0)
        + SCANNER_WEIGHTS["half_life_6m"]
        * (window_results.get("6m") or {}).get("half_life", 0.0)
        + SCANNER_WEIGHTS["beta_stability"]
        * (beta_std if np.isfinite(beta_std) else 0.0)
    )

    out = {
        "eligibility": eligibility,
        "eligibility_score": float(score),
        "n_valid_windows": n_valid,
        "beta_std": beta_std,
    }

    # Flatten window stats
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
            res.update(
                {
                    "asset_1": a1,
                    "asset_2": a2,
                    "universe": universe_name,
                }
            )
            results.append(res)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


# ============================================================
# Scan multi-univers (parallel)
# ============================================================

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
            try:
                res = fut.result()
                if res is not None and not res.empty:
                    dfs.append(res)
            except Exception:
                continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
