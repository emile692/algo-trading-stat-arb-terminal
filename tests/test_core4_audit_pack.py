from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.run_core4_audit_pack import run_core4_audit_pack
from utils.core4_audit_common import BORROW_REQUIRED_SCENARIOS, EXECUTION_DELAY_REQUIRED_SCENARIOS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core4_audit_pack_smoke_runs_and_exports_all_core_artifacts(tmp_path: Path) -> None:
    args = Namespace(
        output_root=tmp_path,
        output_dir=tmp_path / "smoke_pack",
        config_path=PROJECT_ROOT / "config" / "core_portfolio_reference.json",
        daily_cache_dir=PROJECT_ROOT / "data" / "experiments" / "core_portfolio_reference_daily_cache",
        freeze_path=PROJECT_ROOT / "config" / "frozen" / "core_4_country_v1.freeze.json",
        start=None,
        end=None,
        rebuild_daily_cache=False,
        smoke=True,
        no_near_match=False,
        log_level="INFO",
    )
    result = run_core4_audit_pack(args, project_root=PROJECT_ROOT)
    output_dir = result["output_dir"]

    expected_files = [
        "audit_pack_summary.md",
        "daily_positions.csv",
        "daily_book_exposures.csv",
        "daily_portfolio_exposures.csv",
        "trade_ledger.csv",
        "execution_delay_stress.csv",
        "borrow_cost_stress.csv",
    ]
    for name in expected_files:
        assert (output_dir / name).exists(), name

    execution = pd.read_csv(output_dir / "execution_delay_stress.csv")
    borrow = pd.read_csv(output_dir / "borrow_cost_stress.csv")
    assert set(EXECUTION_DELAY_REQUIRED_SCENARIOS).issubset(set(execution["scenario"].astype(str)))
    assert set(BORROW_REQUIRED_SCENARIOS).issubset(set(borrow["scenario"].astype(str)))
