from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils.core4_validation_pack import Core4ValidationOptions, load_frozen_reference_context


DEFAULT_AUDIT_OUTPUT_ROOT = Path("data/experiments/core4_audit_pack")
DEFAULT_FREEZE_PATH = Path("config/frozen/core_4_country_v1.freeze.json")
DEFAULT_CONFIG_PATH = Path("config/core_portfolio_reference.json")
DEFAULT_DAILY_CACHE_DIR = Path("data/experiments/core_portfolio_reference_daily_cache")
REFERENCE_ALLOCATOR_ID = "inverse_vol__lb126__weekly__floor_cap"
BENCHMARK_ALLOCATOR_ID = "equal_weight__lb126__monthly__unconstrained"
OPTIONAL_MONITOR_ALLOCATOR_ID = "risk_parity__lb126__weekly__floor_cap"
DEFAULT_VALIDATION_SOURCE = Path("data/experiments/core4_validation_pack/20260514_105610/validation_summary.md")
DEFAULT_REPORTING_SOURCE = Path("data/reports/core4_daily_reporting/conclusion.txt")
DEFAULT_MULTIBOOK_SOURCE = Path("data/experiments/multibook_portfolio_sweden_germany_france_netherlands_20260421_190655")
DEFAULT_GERMANY_SHADOW_SOURCE = Path("data/experiments/germany_shadow_validation_20260421_222202")

DAILY_POSITIONS_REQUIRED_COLUMNS = [
    "date",
    "book",
    "pair_id",
    "leg_1",
    "leg_2",
    "side",
    "signal_state",
    "entry_z",
    "current_z",
    "notional_leg_1",
    "notional_leg_2",
    "gross_notional",
    "net_notional",
    "weight",
    "allocation_method",
    "source_artifact",
]

DAILY_BOOK_EXPOSURES_REQUIRED_COLUMNS = [
    "date",
    "book",
    "gross_exposure",
    "net_exposure",
    "num_active_pairs",
    "largest_pair_weight",
    "long_notional",
    "short_notional",
]

DAILY_PORTFOLIO_EXPOSURES_REQUIRED_COLUMNS = [
    "date",
    "gross_exposure",
    "net_exposure",
    "num_active_books",
    "num_active_pairs",
    "largest_book_weight",
    "largest_pair_weight",
]

TRADE_LEDGER_REQUIRED_COLUMNS = [
    "trade_id",
    "book",
    "pair_id",
    "leg_1",
    "leg_2",
    "open_date",
    "close_date",
    "holding_days",
    "entry_z",
    "exit_z",
    "exit_reason",
    "pnl_gross",
    "pnl_net_before_borrow",
    "estimated_borrow_cost",
    "pnl_net_after_borrow",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "source_artifact",
]

EXECUTION_DELAY_REQUIRED_SCENARIOS = [
    "delay_0d",
    "delay_1d",
    "delay_2d",
    "delay_3d",
    "entry_delay_only",
    "exit_delay_only",
    "entry_and_exit_delay",
]

BORROW_REQUIRED_SCENARIOS = [
    "borrow_0bps",
    "borrow_50bps",
    "borrow_100bps",
    "borrow_200bps",
    "borrow_500bps",
]


@dataclass(frozen=True)
class Core4AuditBaseOptions:
    output_root: Path = DEFAULT_AUDIT_OUTPUT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    daily_cache_dir: Path = DEFAULT_DAILY_CACHE_DIR
    start: str | None = None
    end: str | None = None
    rebuild_daily_cache: bool = False
    smoke: bool = False
    allow_near_match: bool = True


def resolve_path(project_root: Path, path_like: Path | str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def build_timestamped_output_dir(
    output_root: Path,
    *,
    stamp: datetime | None = None,
    smoke: bool = False,
) -> Path:
    now = (stamp or datetime.now()).replace(microsecond=0)
    name = now.strftime("%Y%m%d_%H%M%S")
    if smoke:
        name = f"{name}_smoke"
    out_dir = Path(output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_core4_audit_context(
    options: Core4AuditBaseOptions,
    *,
    project_root: Path,
) -> dict[str, Any]:
    validation_options = Core4ValidationOptions(
        config_path=resolve_path(project_root, options.config_path),
        daily_cache_dir=resolve_path(project_root, options.daily_cache_dir),
        start=options.start,
        end=options.end,
        rebuild_daily_cache=bool(options.rebuild_daily_cache),
        smoke=bool(options.smoke),
    )
    return load_frozen_reference_context(validation_options, project_root=project_root)


def freeze_manifest_from_context(context: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    config = context["config"]
    books = []
    for book_cfg in config.get("books", []):
        source_dir = resolve_path(project_root, book_cfg["source_dir"])
        source_file = resolve_path(project_root, Path(book_cfg["source_dir"]) / book_cfg["source_file"])
        books.append(
            {
                "book": str(book_cfg.get("book", "")),
                "country": str(book_cfg.get("country", "")),
                "role": str(book_cfg.get("role", "")),
                "config_name": str(book_cfg.get("config_name", "")),
                "logic": str(book_cfg.get("logic", "")),
                "maturity_status": str(book_cfg.get("maturity_status", "")),
                "source_dir": str(source_dir),
                "source_file": str(source_file),
                "source_config_name": str(book_cfg.get("source_config_name", "")),
                "source_book": str(book_cfg.get("source_book", "")),
                "notes": str(book_cfg.get("notes", "")),
            }
        )

    manifest = {
        "strategy_id": str(config.get("portfolio_id", "core_4_country_v1")),
        "strategy_name": str(config.get("name", "Core 4-country stat-arb portfolio")),
        "freeze_date": str(config.get("frozen_at", "")),
        "status": "paper_candidate_not_live",
        "validation_period": {
            "start": str(context["full_start"].date()),
            "end": str(context["full_end"].date()),
        },
        "analysis_window": {
            "start": str(context["full_start"].date()),
            "end": str(context["full_end"].date()),
        },
        "countries_books": books,
        "reference_allocation": {
            "allocator_id": REFERENCE_ALLOCATOR_ID,
            "description": "Frozen Core 4 reference allocator kept as the paper-trading candidate.",
        },
        "benchmark_allocation": {
            "allocator_id": BENCHMARK_ALLOCATOR_ID,
            "description": "Equal-weight benchmark retained as the control allocator.",
        },
        "optional_monitor_allocation": {
            "allocator_id": OPTIONAL_MONITOR_ALLOCATOR_ID,
            "description": "Optional comparison allocator preserved for monitoring only.",
        },
        "source_of_truth": {
            "config_path": str(resolve_path(project_root, context["config_path"])),
            "daily_cache_dir": str(resolve_path(project_root, context["daily_cache_dir"])),
            "monthly_source_paths": [str(resolve_path(project_root, path)) for path in context.get("monthly_source_paths", [])],
            "validation_summary": str(resolve_path(project_root, DEFAULT_VALIDATION_SOURCE)),
            "daily_reporting_summary": str(resolve_path(project_root, DEFAULT_REPORTING_SOURCE)),
            "multibook_decision_dir": str(resolve_path(project_root, DEFAULT_MULTIBOOK_SOURCE)),
            "germany_shadow_validation_dir": str(resolve_path(project_root, DEFAULT_GERMANY_SHADOW_SOURCE)),
        },
        "known_limitations": [
            "Daily positions still require reconstruction from frozen caches and available trade ledgers.",
            "Trade ledger coverage is not exact for every book; Germany relies on a nearby source unless a frozen exact ledger is later exported.",
            "Borrow costs are not modeled in the historical frozen reference and require explicit stress assumptions.",
            "Execution-delay stress remains approximation-based because true daily signal state is not persisted in the frozen artifacts.",
        ],
        "methodology_note": (
            "This manifest freezes the already-approved Core 4 reference only. "
            "It does not retune alpha, modify trading rules, or alter the frozen country-book composition."
        ),
    }
    return manifest


def ensure_freeze_manifest(
    context: dict[str, Any],
    *,
    project_root: Path,
    freeze_path: Path = DEFAULT_FREEZE_PATH,
) -> Path:
    manifest_path = resolve_path(project_root, freeze_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = freeze_manifest_from_context(context, project_root=project_root)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def read_freeze_manifest(project_root: Path, freeze_path: Path = DEFAULT_FREEZE_PATH) -> dict[str, Any]:
    path = resolve_path(project_root, freeze_path)
    return json.loads(path.read_text(encoding="utf-8"))


def non_reconstructible_fields_text(fields: list[str]) -> str:
    cleaned = [str(field).strip() for field in fields if str(field).strip()]
    if not cleaned:
        return ""
    return "; ".join(sorted(dict.fromkeys(cleaned)))


def parse_date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()
