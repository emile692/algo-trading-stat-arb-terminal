import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm


def compute_hedge_ratio(series_y, series_x):
    """OLS hedge ratio."""
    x = sm.add_constant(series_x)
    model = sm.OLS(series_y, x).fit()
    beta = model.params[1]
    return beta


def compute_spread(series_y, series_x, beta):
    return series_y - beta * series_x


def compute_adf(spread):
    """Return ADF test (t-stat, p-value, critical values)."""
    result = adfuller(spread, maxlag=1, autolag='AIC')
    t_stat = result[0]
    p_value = result[1]
    c_values = result[4]
    return t_stat, p_value, c_values


def compute_coint(series_y, series_x):
    eg_t, eg_p, eg_crit = coint(series_y, series_x)
    return eg_t, eg_p, eg_crit


def compute_corr(series_y, series_x):
    return series_y.corr(series_x)


def compute_zscore(spread, window=60):
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

def compute_half_life(spread):
    spread = spread.dropna()

    # Need minimum length
    if len(spread) < 20:
        return None

    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag

    spread_lag = spread_lag.dropna()
    spread_ret = spread_ret.dropna()

    x = sm.add_constant(spread_lag)

    # Fit OU regression
    model = sm.OLS(spread_ret, x).fit()

    # If regression failed to produce slope → not mean reverting
    if len(model.params) < 2:
        return None

    const, phi = model.params

    # theta = -phi
    theta = phi

    # OU requires phi < 0 (negative slope)
    if theta >= 0:
        return None

    half_life = -np.log(2) / theta

    # guardrail against absurd values
    if half_life < 0 or half_life > 10000:
        return None

    return half_life

