# utils/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def _to_series(x, name: str = "x") -> pd.Series:
    """Convertit list/np.ndarray/pd.Series en pd.Series float propre."""
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        # sécurité: si on te passe une df par erreur
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be a Series/1-col DataFrame, got shape={x.shape}")
        s = x.iloc[:, 0].copy()
    else:
        s = pd.Series(x)

    s = pd.to_numeric(s, errors="coerce").astype(float)
    s.name = name
    return s


def compute_hedge_ratio(y, x) -> float:
    """
    OLS: y = a + beta * x
    Retourne beta.
    Robuste aux index/labels (utilise .iloc).
    """
    y = _to_series(y, "y")
    x = _to_series(x, "x")

    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 5:
        return np.nan

    yv = df["y"]
    xv = df["x"]
    X = sm.add_constant(xv, has_constant="add")
    model = sm.OLS(yv, X).fit()

    # params = [const, beta] (en général) -> on prend le 2e coef de manière robuste
    if hasattr(model.params, "iloc"):
        beta = float(model.params.iloc[1])
    else:
        beta = float(model.params[1])
    return beta


def compute_spread(y, x, beta: float | None = None) -> pd.Series:
    """
    Spread = y - beta*x (beta calculé si None).
    Retourne une pd.Series (pas un ndarray), ce qui évite tes erreurs .dropna/.diff/.iloc.
    """
    y = _to_series(y, "y")
    x = _to_series(x, "x")

    if beta is None or (isinstance(beta, float) and not np.isfinite(beta)):
        beta = compute_hedge_ratio(y, x)

    spread = y - float(beta) * x
    spread.name = "spread"
    return spread


def compute_zscore(series, window: int) -> pd.Series:
    """
    Z-score rolling standard.
    """
    s = _to_series(series, "series")

    window = int(window)
    if window <= 1:
        out = pd.Series(np.nan, index=s.index, name="zscore")
        return out

    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    z = (s - m) / sd.replace(0.0, np.nan)
    z.name = "zscore"
    return z


def compute_adf(series):
    """
    ADF test. Retourne (stat, pvalue, crit_values_dict).
    """
    s = _to_series(series, "series").dropna()
    if len(s) < 20:
        return np.nan, np.nan, {}

    stat, pval, _, _, crit, _ = adfuller(s.values, autolag="AIC")
    return float(stat), float(pval), crit


def compute_half_life(spread) -> float | None:
    """
    Half-life via AR(1) sur delta_spread = a + phi * spread_lag + e
    half_life = -ln(2)/phi, si phi < 0
    """
    s = _to_series(spread, "spread").dropna()
    if len(s) < 20:
        return None

    s_lag = s.shift(1)
    ds = s - s_lag
    df = pd.concat([ds.rename("ds"), s_lag.rename("s_lag")], axis=1).dropna()
    if len(df) < 20:
        return None

    y = df["ds"]
    x = df["s_lag"]
    X = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, X).fit()
    phi = float(model.params.iloc[1]) if hasattr(model.params, "iloc") else float(model.params[1])

    # mean-reverting => phi < 0
    if not np.isfinite(phi) or phi >= 0:
        return None

    hl = -np.log(2) / phi
    if not np.isfinite(hl) or hl <= 0 or hl > 100000:
        return None
    return float(hl)

def compute_coint(series_y, series_x):
    y = pd.Series(series_y).dropna().astype(float)
    x = pd.Series(series_x).dropna().astype(float)
    if len(y) < 30 or len(x) < 30:
        return np.nan, np.nan, {}
    eg_t, eg_p, eg_crit = coint(y, x)
    return eg_t, eg_p, eg_crit


def compute_corr(series_y, series_x):
    y = pd.Series(series_y).astype(float)
    x = pd.Series(series_x).astype(float)
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))
