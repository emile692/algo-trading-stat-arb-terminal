from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Tuple

import pandas as pd


# ============================================================
# POSITION (runtime only)
# ============================================================

@dataclass
class Position:
    pair_id: str
    asset_1: str
    asset_2: str
    side: str  # LONG_SPREAD / SHORT_SPREAD
    beta: float
    entry_datetime: pd.Timestamp
    entry_spread: float
    entry_z: float
    entry_y: float
    entry_x: float


# ============================================================
# STRATEGY PARAMETERS (MODEL / ENGINE)
# ============================================================

@dataclass(frozen=True)
class StrategyParams:
    # Entry / exit
    z_entry: float = 2.0
    z_exit: float = 0.4
    z_stop: float = 4.0
    z_window: int = 60

    # Beta estimation
    wf_train: int = 120
    beta_mode: str = "static"   # "static" | "wf"

    # Costs
    fees: float = 0.0002

    # Portfolio construction
    top_n_candidates: int = 20
    max_positions: int = 5

    # Time stop (business days)
    max_holding_days: int = 30

    # Execution lag (pour éviter look-ahead : scan J, trade J+1)
    exec_lag_days: int = 1

    # Scan scheduling controls (used by inline scanner / global_loop)
    # "daily": keep every available scan date.
    # "weekly": keep the last available scan date in each week ending on scan_weekday.
    scan_frequency: str = "daily"   # "daily" | "weekly"
    scan_weekday: str = "FRI"

    # Signal construction space
    # "raw" keeps current behavior (signals on raw log prices).
    # "idio_pca" builds signals on PCA de-factorized idiosyncratic series.
    signal_space: str = "raw"   # "raw" | "idio_pca"

    # PCA signal-space params (used only when signal_space="idio_pca")
    pca_signal_window: int = 252
    pca_signal_components: int = 3
    pca_signal_min_assets: int = 10

    # Entry controls (engine only)
    # "baseline_entry": current |z| threshold entry with no extra filter.
    # "entry_with_spread_speed_filter": benchmark cap on |delta_spread| / spread_std.
    # "entry_zspeed_hard_cap": cap on |delta_z|.
    # "entry_zspeed_ewma_cap": cap on EWMA(|delta_z|).
    # "entry_zspeed_vol_normalized": cap on |delta_z| / rolling_std(delta_z).
    # "entry_slowdown_confirmation": require |delta_z_t| < |delta_z_{t-1}|.
    entry_mode: str = "baseline_entry"
    spread_speed_cap: Optional[float] = None
    zspeed_cap: Optional[float] = None
    zspeed_ewma_span: int = 3
    zspeed_ewma_cap: Optional[float] = None
    zspeed_vol_window: int = 10
    zspeed_vol_cap: Optional[float] = None

    # Scan-time pair selection controls (applied in global_loop)
    # "legacy": keep existing behavior (sort by eligibility_score only).
    # "composite_quality": rank with a blend of scanner features.
    selection_mode: str = "legacy"  # "legacy" | "composite_quality"

    # Allowed scan labels for candidate pool.
    eligibility_labels: Tuple[str, ...] = ("ELIGIBLE",)

    # Optional hard filters (disabled when None).
    min_corr_12m: Optional[float] = None
    max_half_life_6m: Optional[float] = None
    max_beta_std: Optional[float] = None
    min_spread_std_6m: Optional[float] = None
    min_n_valid_windows: Optional[int] = None

    # Optional diversification: cap selected pairs sharing one asset.
    # 0 or less => disabled.
    max_pairs_per_asset: int = 0

    # Composite selection score variants (used when selection_mode="composite_quality")
    # "baseline": current weighted percentile blend.
    # "rank_percentile": equal-weight percentile blend.
    # "robust_zscore": winsorized robust-z aggregation.
    # "rank_stability_penalty": baseline score minus stability penalty.
    # Legacy ranking research variants (used when selection_mode="legacy")
    # also reuse selection_score_variant with light scan-time adjustments.
    selection_score_variant: str = "baseline"
    # 0.0 disables winsorization; e.g. 0.05 clips to [5%, 95%].
    selection_winsor_quantile: float = 0.0
    # Stability penalty strength for "rank_stability_penalty" (0.0 disables).
    selection_stability_penalty: float = 0.0
    # Legacy ranking adjustment strengths (0.0 disables / falls back to default light value).
    selection_half_life_penalty: float = 0.0
    selection_speed_penalty: float = 0.0
    selection_distance_weight: float = 0.0
    selection_corr_penalty: float = 0.0

    # Optional risk guardrails (disabled by default to preserve legacy behavior)
    # Cap each pair daily MTM contribution |ret| before portfolio aggregation.
    pair_return_cap: Optional[float] = None
    # Cap isolated trade return reported in logs (does not affect engine PnL path).
    trade_return_isolated_cap: Optional[float] = None
    # Target daily volatility on portfolio MTM returns.
    portfolio_vol_target: Optional[float] = None
    portfolio_vol_lookback: int = 20
    # Maximum scaling applied by vol targeting (>= 1 allows limited leverage-up).
    portfolio_vol_max_scale: float = 1.0


# ============================================================
# BATCH CONFIG (DATA / IO)
# ============================================================

@dataclass(frozen=True)
class BatchConfig:
    data_path: Path
    scanner_path: Optional[Path] = None
    out_dir: Optional[Path] = None

    universe_name: Optional[str] = None
    timeframe: str = "Daily"

    warmup_extra: int = 50
    equal_weight: bool = True

    start_date: pd.Timestamp = pd.Timestamp(year=2020, month=12, day=1)
    end_date: pd.Timestamp = pd.Timestamp(year=2026, month=1, day=1)


# ============================================================
# PAIR REGISTRY (research / monitoring)
# ============================================================

@dataclass
class PairConfig:
    pair_id: str
    asset1: str
    asset2: str
    timeframe: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    created_at: str
    notes: str = ""
