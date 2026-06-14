from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.core4_audit_common import DAILY_BOOK_EXPOSURES_REQUIRED_COLUMNS, DAILY_PORTFOLIO_EXPOSURES_REQUIRED_COLUMNS, DAILY_POSITIONS_REQUIRED_COLUMNS
from utils.core4_position_reconstruction import Core4PositionReconstructionOptions, reconstruct_core4_positions


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core4_position_reconstruction_smoke_outputs_expected_schema() -> None:
    bundle = reconstruct_core4_positions(
        Core4PositionReconstructionOptions(
            output_root=PROJECT_ROOT / "data" / "experiments" / "core4_audit_pack",
            config_path=PROJECT_ROOT / "config" / "core_portfolio_reference.json",
            daily_cache_dir=PROJECT_ROOT / "data" / "experiments" / "core_portfolio_reference_daily_cache",
            smoke=True,
        ),
        project_root=PROJECT_ROOT,
    )

    daily_positions = bundle["daily_positions"]
    daily_book = bundle["daily_book_exposures"]
    daily_portfolio = bundle["daily_portfolio_exposures"]

    assert all(column in daily_positions.columns for column in DAILY_POSITIONS_REQUIRED_COLUMNS)
    assert all(column in daily_book.columns for column in DAILY_BOOK_EXPOSURES_REQUIRED_COLUMNS)
    assert all(column in daily_portfolio.columns for column in DAILY_PORTFOLIO_EXPOSURES_REQUIRED_COLUMNS)

    if not daily_positions.empty:
        dates = pd.to_datetime(daily_positions["date"], errors="coerce")
        assert dates.is_monotonic_increasing
        assert not daily_positions.duplicated(subset=["date", "book", "trade_id"]).any()

    if not daily_book.empty:
        dates = pd.to_datetime(daily_book["date"], errors="coerce")
        assert dates.is_monotonic_increasing
