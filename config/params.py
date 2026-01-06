# config/params.py

# =========================
# Lookback mapping UI / CLI
# =========================
lookback_mapping = {
    "1 Week": 7,
    "2 Weeks": 14,
    "1 Month": 31,
    "3 Months": 92,
    "6 Months": 183,
    "1 Year": 365,
    "2 Years": 365 * 2,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
    "Max": 100_000,
}

# =========================
# Scanner multi-fenêtres
# (en nombre de barres)
# =========================
SCAN_LOOKBACKS = {
    "3m": 63,
    "6m": 126,
    "12m": 252,
}

# =========================
# Seuils de validation
# =========================
SCANNER_THRESHOLDS = {
    "eg_p_max": 0.05,
    "adf_p_max": 0.05,
    "half_life_max": 100,
    "corr_min": 0.3,
}

# =========================
# Scoring
# =========================
SCANNER_WEIGHTS = {
    "n_valid": 1.0,
    "corr_12m": 0.5,
    "half_life_6m": -0.01,
    "beta_stability": -1.0,
}
