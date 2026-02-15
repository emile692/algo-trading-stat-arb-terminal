# config/params.py
import datetime

# ============================================================
# BTT
# ============================================================

UNIVERSES = [
    "sweden",
    "denmark",
    "norway",
    "finland",
    "france",
    "germany",
    "italy",
    "spain",
]

START_DATE = datetime.datetime(year=2024, month=12, day=1)
END_DATE = datetime.datetime(year=2026, month=1, day=1)

# ============================================================
# LOOKBACK mapping UI / CLI
# ============================================================

LOOKBACK_MAPPING = {
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

# ============================================================
# SCANNER (pair eligibility)
# ============================================================

SCANNER_FREQ = "B"

SCANNER_START_DATE = datetime.datetime(year=2015, month=12, day=1)
SCANNER_END_DATE = datetime.datetime(year=2026, month=1, day=1)

SCAN_LOOKBACKS = {
    "3m": 63,
    "6m": 126,
    "12m": 252,
}

SCANNER_THRESHOLDS = {
    "eg_p_max": 0.05,
    "adf_p_max": 0.05,
    "half_life_max": 100,
    "corr_min": 0.3,
}

SCANNER_WEIGHTS = {
    "n_valid": 1.0,
    "corr_12m": 0.5,
    "half_life_6m": -0.01,
    "beta_stability": -1.0,
}

# ============================================================
# MONTHLY UNIVERSE CONSTRUCTION
# ============================================================

MONTHLY_UNIVERSE_CONFIG = {
    # which scanner labels are allowed
    "eligibility_allowed": ["ELIGIBLE"],   # or ["ELIGIBLE", "WATCH"]

    # number of pairs kept per month
    "top_k": 20,

    # safety: skip month if too few candidates
    "min_pairs_required": 3,
}
