from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.core4_audit_common import (
    DEFAULT_AUDIT_OUTPUT_ROOT,
    DEFAULT_FREEZE_PATH,
    build_timestamped_output_dir,
    ensure_freeze_manifest,
    freeze_manifest_from_context,
)
from utils.core4_borrow_cost_stress import Core4BorrowCostStressOptions, run_core4_borrow_cost_stress
from utils.core4_execution_delay_stress import Core4ExecutionDelayStressOptions, run_core4_execution_delay_stress
from utils.core4_position_reconstruction import Core4PositionReconstructionOptions, run_core4_position_reconstruction
from utils.core4_trade_ledger import Core4TradeLedgerOptions, run_core4_trade_ledger
from utils.core4_validation_pack import Core4ValidationOptions, load_frozen_reference_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the consolidated Core 4 audit pack for paper-readiness review.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / DEFAULT_AUDIT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "config" / "core_portfolio_reference.json")
    parser.add_argument("--daily-cache-dir", type=Path, default=PROJECT_ROOT / "data" / "experiments" / "core_portfolio_reference_daily_cache")
    parser.add_argument("--freeze-path", type=Path, default=PROJECT_ROOT / DEFAULT_FREEZE_PATH)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebuild-daily-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-near-match", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def run_core4_audit_pack(args: argparse.Namespace, *, project_root: Path) -> dict[str, Any]:
    context = load_frozen_reference_context(
        Core4ValidationOptions(
            config_path=Path(args.config_path),
            daily_cache_dir=Path(args.daily_cache_dir),
            start=args.start,
            end=args.end,
            rebuild_daily_cache=bool(args.rebuild_daily_cache),
            smoke=bool(args.smoke),
        ),
        project_root=project_root,
    )
    freeze_path = ensure_freeze_manifest(context, project_root=project_root, freeze_path=Path(args.freeze_path))
    output_dir = Path(args.output_dir) if args.output_dir else build_timestamped_output_dir(Path(args.output_root), stamp=datetime.now(), smoke=bool(args.smoke))
    output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "output_root": Path(args.output_root),
        "config_path": Path(args.config_path),
        "daily_cache_dir": Path(args.daily_cache_dir),
        "start": args.start,
        "end": args.end,
        "rebuild_daily_cache": bool(args.rebuild_daily_cache),
        "smoke": bool(args.smoke),
        "allow_near_match": not bool(args.no_near_match),
    }

    position_bundle = run_core4_position_reconstruction(
        Core4PositionReconstructionOptions(**common_kwargs),
        project_root=project_root,
        output_dir=output_dir,
    )
    trade_bundle = run_core4_trade_ledger(
        Core4TradeLedgerOptions(**common_kwargs),
        project_root=project_root,
        output_dir=output_dir,
    )
    execution_bundle = run_core4_execution_delay_stress(
        Core4ExecutionDelayStressOptions(**common_kwargs),
        project_root=project_root,
        output_dir=output_dir,
    )
    borrow_bundle = run_core4_borrow_cost_stress(
        Core4BorrowCostStressOptions(**common_kwargs),
        project_root=project_root,
        output_dir=output_dir,
    )

    manifest = freeze_manifest_from_context(context, project_root=project_root)
    (output_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = _build_audit_pack_summary(
        manifest=manifest,
        freeze_path=freeze_path,
        position_bundle=position_bundle,
        trade_bundle=trade_bundle,
        execution_bundle=execution_bundle,
        borrow_bundle=borrow_bundle,
    )
    (output_dir / "audit_pack_summary.md").write_text(summary, encoding="utf-8")
    result = {
        "output_dir": output_dir,
        "freeze_path": freeze_path,
        "position_bundle": position_bundle,
        "trade_bundle": trade_bundle,
        "execution_bundle": execution_bundle,
        "borrow_bundle": borrow_bundle,
        "summary": summary,
    }
    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    result = run_core4_audit_pack(args, project_root=PROJECT_ROOT)
    logging.info("Core 4 audit pack written to %s", result["output_dir"])
    logging.info("Freeze manifest available at %s", result["freeze_path"])
    return 0


def _build_audit_pack_summary(
    *,
    manifest: dict[str, Any],
    freeze_path: Path,
    position_bundle: dict[str, Any],
    trade_bundle: dict[str, Any],
    execution_bundle: dict[str, Any],
    borrow_bundle: dict[str, Any],
) -> str:
    context = position_bundle["context"]
    daily_positions = position_bundle["daily_positions"]
    daily_portfolio_exposures = position_bundle["daily_portfolio_exposures"]
    trades = trade_bundle["trade_ledger"]
    execution = execution_bundle["execution_delay_stress"]
    borrow = borrow_bundle["borrow_cost_stress"]

    verdict = _paper_verdict(
        daily_positions=daily_positions,
        trades=trades,
        execution=execution,
        borrow=borrow,
    )
    lines = [
        "# Core 4 Audit Pack Summary",
        "",
        f"- Status: {manifest['status']}",
        f"- Verdict: {verdict}",
        f"- Strategy id: {manifest['strategy_id']}",
        f"- Freeze date: {manifest['freeze_date']}",
        f"- Freeze manifest: {freeze_path}",
        f"- Analysis window: {context['analysis_start'].date()} to {context['analysis_end'].date()}",
        "",
        "## Frozen Configuration",
        f"- Reference allocator: {manifest['reference_allocation']['allocator_id']}",
        f"- Benchmark allocator: {manifest['benchmark_allocation']['allocator_id']}",
        f"- Books: {', '.join(book['book'] for book in manifest['countries_books'])}",
        "",
        "## Validation Read-Through",
        "- Existing frozen validation source remains the primary performance reference.",
        f"- Frozen full-sample return proxy: {_fmt_pct(context['portfolio_returns'].mean() * 252.0)} annualized mean-of-daily approximation.",
        "",
        "## Reconstructed Exposures",
        f"- Daily positions rows: {len(daily_positions)}",
        f"- Daily trade ledger rows: {len(trades)}",
    ]
    if not daily_portfolio_exposures.empty:
        latest = daily_portfolio_exposures.iloc[-1]
        lines.append(
            f"- Latest portfolio exposure snapshot ({latest['date']}): gross={_fmt_number(latest['gross_exposure'])} | "
            f"net={_fmt_number(latest['net_exposure'])} | active_pairs={_fmt_number(latest['num_active_pairs'])}"
        )

    lines.extend(["", "## Trade-Level Summary"])
    if trades.empty:
        lines.append("- No trade ledger rows were reconstructed.")
    else:
        pnl = pd.to_numeric(trades["pnl_net_before_borrow"], errors="coerce")
        lines.append(f"- Total trades: {len(trades)}")
        lines.append(f"- Hit ratio: {_fmt_pct((pnl.fillna(0.0) > 0.0).mean())}")
        lines.append(f"- Average trade PnL: {_fmt_number(pnl.mean())}")

    lines.extend(["", "## Execution Delay Stress"])
    for row in execution.itertuples(index=False):
        lines.append(
            f"- {row.scenario}: Sharpe={_fmt_number(row.sharpe)} | maxDD={_fmt_pct(row.max_drawdown)} | "
            f"deltaSharpe={_fmt_number(row.delta_sharpe_vs_reference)}"
        )

    lines.extend(["", "## Borrow Cost Stress"])
    for row in borrow.itertuples(index=False):
        lines.append(
            f"- {row.scenario}: Sharpe={_fmt_number(row.sharpe_after_borrow)} | maxDD={_fmt_pct(row.max_drawdown_after_borrow)} | "
            f"cost={_fmt_number(row.total_estimated_borrow_cost)}"
        )

    lines.extend(["", "## Remaining Limitations"])
    for item in manifest["known_limitations"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Verdict Rationale",
            "- This pack adds a first institutional audit layer without changing alpha or the frozen Core 4 composition.",
            "- The package is useful for paper review, but exact position-state persistence and richer execution plumbing are still missing.",
        ]
    )
    return "\n".join(lines) + "\n"


def _paper_verdict(
    *,
    daily_positions: pd.DataFrame,
    trades: pd.DataFrame,
    execution: pd.DataFrame,
    borrow: pd.DataFrame,
) -> str:
    if daily_positions.empty or trades.empty or execution.empty or borrow.empty:
        return "not_paper_ready"
    if "near_match" in set(trades.get("source_match_status", pd.Series(dtype=str)).astype(str)):
        return "paper_ready_with_limitations"
    return "paper_ready_with_limitations"


def _fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{100.0 * float(value):.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
