from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.core4_audit_common import DEFAULT_AUDIT_OUTPUT_ROOT, build_timestamped_output_dir
from utils.core4_position_reconstruction import Core4PositionReconstructionOptions, run_core4_position_reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct Core 4 daily positions and exposure diagnostics.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / DEFAULT_AUDIT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "config" / "core_portfolio_reference.json")
    parser.add_argument("--daily-cache-dir", type=Path, default=PROJECT_ROOT / "data" / "experiments" / "core_portfolio_reference_daily_cache")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebuild-daily-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-near-match", action="store_true", help="Disable near-match ledger reconstruction for Germany-like gaps.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    output_dir = args.output_dir or build_timestamped_output_dir(Path(args.output_root), smoke=bool(args.smoke))
    options = Core4PositionReconstructionOptions(
        output_root=Path(args.output_root),
        config_path=Path(args.config_path),
        daily_cache_dir=Path(args.daily_cache_dir),
        start=args.start,
        end=args.end,
        rebuild_daily_cache=bool(args.rebuild_daily_cache),
        smoke=bool(args.smoke),
        allow_near_match=not bool(args.no_near_match),
    )
    bundle = run_core4_position_reconstruction(options, project_root=PROJECT_ROOT, output_dir=Path(output_dir))
    logging.info("Core 4 position reconstruction written to %s", output_dir)
    logging.info("Reconstructed %s daily position rows", len(bundle["daily_positions"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
