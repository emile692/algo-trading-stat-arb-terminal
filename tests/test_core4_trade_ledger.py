from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.core4_audit_common import TRADE_LEDGER_REQUIRED_COLUMNS
from utils.core4_trade_ledger import Core4TradeLedgerOptions, reconstruct_core4_trade_ledger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core4_trade_ledger_smoke_has_expected_columns_and_sorted_dates() -> None:
    bundle = reconstruct_core4_trade_ledger(
        Core4TradeLedgerOptions(
            output_root=PROJECT_ROOT / "data" / "experiments" / "core4_audit_pack",
            config_path=PROJECT_ROOT / "config" / "core_portfolio_reference.json",
            daily_cache_dir=PROJECT_ROOT / "data" / "experiments" / "core_portfolio_reference_daily_cache",
            smoke=True,
        ),
        project_root=PROJECT_ROOT,
    )

    ledger = bundle["trade_ledger"]
    assert all(column in ledger.columns for column in TRADE_LEDGER_REQUIRED_COLUMNS)

    if not ledger.empty:
        open_dates = pd.to_datetime(ledger["open_date"], errors="coerce")
        assert open_dates.is_monotonic_increasing
        assert not ledger.duplicated(subset=["trade_id"]).any()
        assert not ledger.duplicated(subset=["book", "pair_id", "open_date", "close_date"]).any()
