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

    # Signal construction space
    # "raw" keeps current behavior (signals on raw log prices).
    # "idio_pca" builds signals on PCA de-factorized idiosyncratic series.
    signal_space: str = "raw"   # "raw" | "idio_pca"

    # PCA signal-space params (used only when signal_space="idio_pca")
    pca_signal_window: int = 252
    pca_signal_components: int = 3
    pca_signal_min_assets: int = 10

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
