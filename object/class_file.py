from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import pandas as pd

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


@dataclass(frozen=True)
class StrategyParams:
    z_entry: float = 2.0
    z_exit: float = 0.4
    z_stop: float = 4.0
    z_window: int = 60

    wf_train: int = 120
    wf_test: int = 30  # utilisé uniquement en beta_mode="wf"

    fees: float = 0.0002
    beta_mode: str = "monthly"  # "monthly" | "wf"

    top_n_candidates : int = 20
    max_positions : int = 5
    rebalance_period: str = "daily"  # "daily" | "weekly" | "monthly"

    rebalance_period:str ="monthly"


@dataclass(frozen=True)
class BatchConfig:
    data_path: Path
    monthly_universe_path: Path
    out_dir: Path
    universe_name: Optional[str] = None
    timeframe: str = "Daily"
    warmup_extra: int = 50
    equal_weight: bool = True
    start_date: pd.Timestamp = pd.Timestamp(year=2020, month=12, day=1)
    end_date: pd.Timestamp = pd.Timestamp(year=2026, month=1, day=1)


@dataclass
class PairConfig:
    pair_id: str                 # ex: "AAPL-MSFT"
    asset1: str
    asset2: str
    timeframe: str               # ex: "1H" (ou ce que tu utilises)
    params: dict[str, Any]       # z_entry, z_exit, z_window, wf_train, wf_test, fees, z_stop_mult, etc.
    metrics: dict[str, Any]      # Sharpe_real, Robustness, Sharpe_min, etc.
    created_at: str              # ISO timestamp
    notes: str = ""              # optionnel
