from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

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
