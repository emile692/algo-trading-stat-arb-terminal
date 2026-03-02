from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from typing import Dict

from config.params import SCAN_LOOKBACKS, SCANNER_THRESHOLDS, SCANNER_WEIGHTS
from utils.metrics import compute_adf, compute_coint


# ============================================================
# Helpers - numerical sanity
# ============================================================

_MIN_STD = 1e-8
_MIN_VAR = 1e-12
_MIN_LEN = 50
_MIN_WINDOW_LEN = 30
_MIN_WINDOW_RATIO = 0.8
_FLAT_MOVE_EPS = 1e-6


def _align_pair(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return aligned finite arrays.
    - Series/Series: keeps index alignment semantics.
    - ndarray inputs (same length): fast finite mask path.
    """
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        df = pd.concat([y, x], axis=1).dropna()
        if df.empty:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        return (
            df.iloc[:, 0].to_numpy(dtype=float, copy=False),
            df.iloc[:, 1].to_numpy(dtype=float, copy=False),
        )

    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)

    if yv.shape != xv.shape:
        df = pd.concat([pd.Series(yv), pd.Series(xv)], axis=1).dropna()
        if df.empty:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        return (
            df.iloc[:, 0].to_numpy(dtype=float, copy=False),
            df.iloc[:, 1].to_numpy(dtype=float, copy=False),
        )

    mask = np.isfinite(yv) & np.isfinite(xv)
    if not np.any(mask):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return yv[mask], xv[mask]


def _is_valid_array(
    y: np.ndarray,
    x: np.ndarray,
    min_len: int = 50,
    min_std: float = 1e-8,
) -> bool:
    """
    Reject degenerate series before any stat.
    """
    if y is None or x is None:
        return False
    if y.size < min_len or x.size < min_len:
        return False
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        return False
    if float(np.std(y)) < min_std or float(np.std(x)) < min_std:
        return False
    return True


def _fast_ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    """
    OLS slope for y = a + beta*x computed in closed form.
    Equivalent to statsmodels OLS slope with intercept.
    """
    dx = x - float(np.mean(x))
    dy = y - float(np.mean(y))
    denom = float(np.dot(dx, dx))
    if not np.isfinite(denom) or denom <= _MIN_VAR:
        return np.nan
    return float(np.dot(dx, dy) / denom)


def _fast_corr(y: np.ndarray, x: np.ndarray) -> float:
    dy = y - float(np.mean(y))
    dx = x - float(np.mean(x))
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if not np.isfinite(denom) or denom <= _MIN_VAR:
        return np.nan
    return float(np.dot(dx, dy) / denom)


def _fast_half_life(spread: np.ndarray) -> float | None:
    """
    Half-life via AR(1): ds = a + phi*s_lag + e.
    """
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        return None

    s_lag = s[:-1]
    ds = np.diff(s)
    if ds.size < 20:
        return None

    dx = s_lag - float(np.mean(s_lag))
    dy = ds - float(np.mean(ds))
    denom = float(np.dot(dx, dx))
    if not np.isfinite(denom) or denom <= _MIN_VAR:
        return None

    phi = float(np.dot(dx, dy) / denom)
    if not np.isfinite(phi) or phi >= 0:
        return None

    hl = -np.log(2.0) / phi
    if not np.isfinite(hl) or hl <= 0 or hl > 100000:
        return None
    return float(hl)


def _scan_pair_window_aligned(
    y_aligned: np.ndarray,
    x_aligned: np.ndarray,
    lookback: int,
) -> Dict[str, float]:
    min_required = max(_MIN_WINDOW_LEN, int(_MIN_WINDOW_RATIO * lookback))
    if y_aligned.size < min_required or x_aligned.size < min_required:
        raise ValueError("Not enough data")

    if y_aligned.size > lookback:
        yv = y_aligned[-lookback:]
        xv = x_aligned[-lookback:]
    else:
        yv = y_aligned
        xv = x_aligned

    if not _is_valid_array(yv, xv, min_len=_MIN_LEN, min_std=_MIN_STD):
        raise ValueError("Degenerate series")

    # Reject flat price windows before expensive tests.
    if float(np.abs(np.diff(yv)).sum()) < _FLAT_MOVE_EPS or float(np.abs(np.diff(xv)).sum()) < _FLAT_MOVE_EPS:
        raise ValueError("Flat price window")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        beta = _fast_ols_beta(yv, xv)
        if not np.isfinite(beta):
            raise ValueError("Invalid beta")

        spread = yv - beta * xv
        corr = _fast_corr(yv, xv)
        if not np.isfinite(corr):
            raise ValueError("Invalid corr")

        hl = _fast_half_life(spread)
        if hl is None or not np.isfinite(hl):
            raise ValueError("Invalid half-life")

        # Early exits that are logically equivalent for validity:
        # if corr/half-life already fail hard constraints, ADF/EG cannot make this window valid.
        run_stationarity = (
            abs(corr) > SCANNER_THRESHOLDS["corr_min"]
            and hl < SCANNER_THRESHOLDS["half_life_max"]
        )

        adf_p = np.nan
        eg_p = np.nan
        if run_stationarity:
            _, adf_p, _ = compute_adf(spread)

            # EG p-value only matters if ADF already passes the required threshold.
            if np.isfinite(adf_p) and adf_p < SCANNER_THRESHOLDS["adf_p_max"]:
                _, eg_p, _ = compute_coint(yv, xv)

    return {
        "beta": float(beta),
        "corr": float(corr),
        "adf_p": float(adf_p),
        "eg_p": float(eg_p),
        "half_life": float(hl),
        "spread_std": float(np.nanstd(spread)),
    }


# ============================================================
# Scan on one window
# ============================================================

def scan_pair_window(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    lookback: int,
) -> Dict[str, float]:
    y_aligned, x_aligned = _align_pair(y, x)
    return _scan_pair_window_aligned(y_aligned, x_aligned, lookback)


# ============================================================
# Window validation
# ============================================================

def window_is_valid(res: Dict[str, float]) -> bool:
    if res is None:
        return False

    half_life = res.get("half_life")
    adf_p = res.get("adf_p")
    eg_p = res.get("eg_p")
    corr = res.get("corr")

    if half_life is None:
        return False
    if not (np.isfinite(half_life) and np.isfinite(corr) and np.isfinite(adf_p) and np.isfinite(eg_p)):
        return False

    return (
        eg_p < SCANNER_THRESHOLDS["eg_p_max"]
        and adf_p < SCANNER_THRESHOLDS["adf_p_max"]
        and half_life < SCANNER_THRESHOLDS["half_life_max"]
        and abs(corr) > SCANNER_THRESHOLDS["corr_min"]
    )


# ============================================================
# Multi-window scan (core)
# ============================================================

def _safe_metric(window_results: Dict[str, Dict | None], window: str, key: str, default: float = 0.0) -> float:
    res = window_results.get(window)
    if res is None:
        return default
    v = res.get(key, default)
    try:
        vf = float(v)
    except (TypeError, ValueError):
        return default
    return vf if np.isfinite(vf) else default


def scan_pair_multi_window(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
) -> Dict[str, float]:
    y_aligned, x_aligned = _align_pair(y, x)

    window_results: Dict[str, Dict | None] = {}

    for label, lb in SCAN_LOOKBACKS.items():
        try:
            window_results[label] = _scan_pair_window_aligned(y_aligned, x_aligned, lb)
        except Exception:
            window_results[label] = None

    valid_windows = [k for k, v in window_results.items() if window_is_valid(v)]
    n_valid = len(valid_windows)

    # If no window usable, drop the pair entirely.
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
        + SCANNER_WEIGHTS["corr_12m"] * _safe_metric(window_results, "12m", "corr", default=0.0)
        + SCANNER_WEIGHTS["half_life_6m"] * _safe_metric(window_results, "6m", "half_life", default=0.0)
        + SCANNER_WEIGHTS["beta_stability"] * (beta_std if np.isfinite(beta_std) else 0.0)
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
# Scan one universe
# ============================================================

def scan_universe(
    price_df: pd.DataFrame,
    universe_name: str,
) -> pd.DataFrame:
    if price_df is None or price_df.shape[1] < 2:
        return pd.DataFrame()

    asset_names = [str(c) for c in price_df.columns]
    values = price_df.to_numpy(dtype=float, copy=False)
    n_assets = len(asset_names)

    results = []

    for i in range(n_assets - 1):
        a1 = asset_names[i]
        y = values[:, i]
        for j in range(i + 1, n_assets):
            a2 = asset_names[j]
            try:
                res = scan_pair_multi_window(y, values[:, j])
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

    return pd.DataFrame(results)


# ============================================================
# Multi-universe scan (parallel)
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
