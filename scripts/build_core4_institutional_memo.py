from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.core4_institutional_memo import (
    DEFAULT_AUDIT_PACK_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VALIDATION_PACK_DIR,
    Core4InstitutionalMemoOptions,
    build_core4_institutional_memo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Core 4 institutional HTML memo from frozen audit artifacts.")
    parser.add_argument("--audit-pack-dir", type=Path, default=PROJECT_ROOT / DEFAULT_AUDIT_PACK_DIR)
    parser.add_argument("--validation-pack-dir", type=Path, default=PROJECT_ROOT / DEFAULT_VALIDATION_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze-path", type=Path, default=PROJECT_ROOT / "config" / "frozen" / "core_4_country_v1.freeze.json")
    parser.add_argument("--reporting-dir", type=Path, default=PROJECT_ROOT / "data" / "reports" / "core4_daily_reporting")
    parser.add_argument("--allocation-research-dir", type=Path, default=PROJECT_ROOT / "data" / "experiments" / "portfolio_allocation_research_france_germany_netherlands_sweden_20260423_000056_codex_main_final")
    parser.add_argument("--multibook-dir", type=Path, default=PROJECT_ROOT / "data" / "experiments" / "multibook_portfolio_sweden_germany_france_netherlands_20260421_190655")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    options = Core4InstitutionalMemoOptions(
        audit_pack_dir=Path(args.audit_pack_dir),
        validation_pack_dir=Path(args.validation_pack_dir),
        output_dir=Path(args.output_dir),
        freeze_path=Path(args.freeze_path),
        reporting_dir=Path(args.reporting_dir),
        allocation_research_dir=Path(args.allocation_research_dir),
        multibook_dir=Path(args.multibook_dir),
        smoke=bool(args.smoke),
    )
    result = build_core4_institutional_memo(options, project_root=PROJECT_ROOT)
    logging.info("Core 4 memo HTML written to %s", result["html_path"])
    logging.info("Memo manifest written to %s", result["manifest_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
