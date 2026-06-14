from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.core4_audit_common import (
    DAILY_BOOK_EXPOSURES_REQUIRED_COLUMNS,
    DAILY_PORTFOLIO_EXPOSURES_REQUIRED_COLUMNS,
    DAILY_POSITIONS_REQUIRED_COLUMNS,
    DEFAULT_AUDIT_OUTPUT_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DAILY_CACHE_DIR,
    REFERENCE_ALLOCATOR_ID,
    Core4AuditBaseOptions,
    load_core4_audit_context,
    non_reconstructible_fields_text,
    parse_date_series,
    resolve_path,
)
from utils.core4_trade_ledger import Core4TradeLedgerOptions, reconstruct_core4_trade_ledger
from utils.core4_validation_pack import compute_daily_exposure


@dataclass(frozen=True)
class Core4PositionReconstructionOptions(Core4AuditBaseOptions):
    output_root: Path = DEFAULT_AUDIT_OUTPUT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    daily_cache_dir: Path = DEFAULT_DAILY_CACHE_DIR


def reconstruct_core4_positions(
    options: Core4PositionReconstructionOptions,
    *,
    project_root: Path,
) -> dict[str, Any]:
    context = load_core4_audit_context(options, project_root=project_root)
    trade_bundle = reconstruct_core4_trade_ledger(
        Core4TradeLedgerOptions(
            output_root=options.output_root,
            config_path=options.config_path,
            daily_cache_dir=options.daily_cache_dir,
            start=options.start,
            end=options.end,
            rebuild_daily_cache=options.rebuild_daily_cache,
            smoke=options.smoke,
            allow_near_match=options.allow_near_match,
        ),
        project_root=project_root,
    )
    raw_daily_exposure = compute_daily_exposure(context["daily_long"], context["frozen_weights"])
    daily_positions, coverage = _build_daily_positions(
        trade_ledger=trade_bundle["trade_ledger"],
        daily_exposure=raw_daily_exposure,
        frozen_weights=context["frozen_weights"],
    )
    daily_book_exposures = _build_daily_book_exposures(
        daily_positions=daily_positions,
        daily_exposure=raw_daily_exposure,
        frozen_weights=context["frozen_weights"],
    )
    daily_portfolio_exposures = _build_daily_portfolio_exposures(
        daily_positions=daily_positions,
        daily_book_exposures=daily_book_exposures,
    )
    summary = _build_position_summary(
        context=context,
        coverage=coverage,
        daily_positions=daily_positions,
        daily_book_exposures=daily_book_exposures,
        daily_portfolio_exposures=daily_portfolio_exposures,
    )
    return {
        "context": context,
        "trade_bundle": trade_bundle,
        "raw_daily_exposure": raw_daily_exposure,
        "daily_positions": daily_positions,
        "daily_book_exposures": daily_book_exposures,
        "daily_portfolio_exposures": daily_portfolio_exposures,
        "position_reconstruction_summary": summary,
    }


def run_core4_position_reconstruction(
    options: Core4PositionReconstructionOptions,
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bundle = reconstruct_core4_positions(options, project_root=project_root)
    output_dir = resolve_path(project_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle["daily_positions"].to_csv(output_dir / "daily_positions.csv", index=False)
    bundle["daily_book_exposures"].to_csv(output_dir / "daily_book_exposures.csv", index=False)
    bundle["daily_portfolio_exposures"].to_csv(output_dir / "daily_portfolio_exposures.csv", index=False)
    (output_dir / "position_reconstruction_summary.md").write_text(bundle["position_reconstruction_summary"], encoding="utf-8")
    return bundle


def _build_daily_positions(
    *,
    trade_ledger: pd.DataFrame,
    daily_exposure: pd.DataFrame,
    frozen_weights: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = DAILY_POSITIONS_REQUIRED_COLUMNS + [
        "trade_id",
        "source_match_status",
        "requested_config_name",
        "source_config_name",
        "book_gross_exposure",
        "book_num_pairs_proxy",
        "book_weight_reference",
        "approximation_notes",
        "non_reconstructible_fields",
    ]
    if trade_ledger.empty:
        return pd.DataFrame(columns=columns), {"residual_exposure_book_days": np.nan, "covered_book_days": 0}

    trade_frame = trade_ledger.copy()
    trade_frame["open_date"] = parse_date_series(trade_frame["open_date"])
    trade_frame["close_date"] = parse_date_series(trade_frame["close_date"])
    trade_frame = trade_frame.dropna(subset=["open_date", "close_date"]).copy()
    if trade_frame.empty:
        return pd.DataFrame(columns=columns), {"residual_exposure_book_days": np.nan, "covered_book_days": 0}

    exposure = daily_exposure.copy()
    exposure = exposure[exposure["country"].astype(str).ne("portfolio")].copy()
    exposure["book"] = exposure["country"].astype(str)
    exposure["date"] = parse_date_series(exposure["date"])
    exposure = exposure.dropna(subset=["date"]).sort_values(["date", "book"]).reset_index(drop=True)
    exposure_map = exposure.set_index(["date", "book"])

    rows: list[dict[str, Any]] = []
    for row in trade_frame.itertuples(index=False):
        if row.close_date < row.open_date:
            continue
        active_dates = pd.date_range(row.open_date, row.close_date, freq="D")
        for active_date in active_dates:
            rows.append(
                {
                    "date": active_date.normalize(),
                    "book": str(row.book),
                    "pair_id": str(row.pair_id),
                    "leg_1": str(row.leg_1),
                    "leg_2": str(row.leg_2),
                    "side": str(row.side),
                    "signal_state": "active_between_open_close_proxy",
                    "entry_z": row.entry_z,
                    "current_z": np.nan,
                    "notional_leg_1": np.nan,
                    "notional_leg_2": np.nan,
                    "gross_notional": np.nan,
                    "net_notional": 0.0,
                    "weight": np.nan,
                    "allocation_method": REFERENCE_ALLOCATOR_ID,
                    "source_artifact": str(row.source_artifact),
                    "trade_id": str(row.trade_id),
                    "source_match_status": str(row.source_match_status),
                    "requested_config_name": str(row.requested_config_name),
                    "source_config_name": str(row.source_config_name),
                    "book_gross_exposure": np.nan,
                    "book_num_pairs_proxy": np.nan,
                    "book_weight_reference": float(frozen_weights.get(str(row.book), np.nan)),
                    "approximation_notes": non_reconstructible_fields_text(
                        [
                            str(row.approximation_notes),
                            "Daily position state is expanded from open_date..close_date inclusive.",
                            "Pair notionals are allocated from book-level exposure proxies when available.",
                        ]
                    ),
                    "non_reconstructible_fields": non_reconstructible_fields_text(["current_z"]),
                }
            )

    positions = pd.DataFrame(rows, columns=columns)
    if positions.empty:
        return positions, {"residual_exposure_book_days": np.nan, "covered_book_days": 0}

    positions["date"] = parse_date_series(positions["date"])
    active_counts = (
        positions.groupby(["date", "book"], dropna=False)["trade_id"]
        .nunique()
        .rename("active_trade_count")
        .reset_index()
    )
    positions = positions.merge(active_counts, on=["date", "book"], how="left")

    merged_rows = []
    for row in positions.itertuples(index=False):
        key = (row.date, row.book)
        if key in exposure_map.index:
            exp = exposure_map.loc[key]
            gross = float(pd.to_numeric(exp["gross_exposure"], errors="coerce"))
            num_pairs_proxy = float(pd.to_numeric(exp["number_of_pairs"], errors="coerce"))
        else:
            gross = np.nan
            num_pairs_proxy = np.nan
        active_trade_count = float(getattr(row, "active_trade_count", np.nan))
        if np.isfinite(gross) and np.isfinite(active_trade_count) and active_trade_count > 0.0:
            gross_notional = gross / active_trade_count
            leg_notional = gross_notional / 2.0
            weight = gross_notional
        else:
            gross_notional = np.nan
            leg_notional = np.nan
            weight = np.nan
        merged_rows.append(
            {
                **row._asdict(),
                "book_gross_exposure": gross,
                "book_num_pairs_proxy": num_pairs_proxy,
                "gross_notional": gross_notional,
                "notional_leg_1": leg_notional,
                "notional_leg_2": leg_notional,
                "weight": weight,
                "net_notional": 0.0 if np.isfinite(gross_notional) else np.nan,
            }
        )

    positions = pd.DataFrame(merged_rows, columns=positions.columns.tolist())
    positions["date"] = pd.to_datetime(positions["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    positions = positions.sort_values(["date", "book", "trade_id"]).reset_index(drop=True)

    exposure_book_days = exposure[["date", "book"]].copy()
    exposure_book_days["date"] = pd.to_datetime(exposure_book_days["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    exposure_book_days = exposure_book_days.drop_duplicates()
    position_book_days = positions[["date", "book"]].drop_duplicates()
    residual = exposure_book_days.merge(position_book_days, on=["date", "book"], how="left", indicator=True)
    residual_count = int(residual["_merge"].eq("left_only").sum())
    coverage = {
        "residual_exposure_book_days": residual_count,
        "covered_book_days": int(len(position_book_days)),
    }
    ordered = columns
    return positions.loc[:, ordered], coverage


def _build_daily_book_exposures(
    *,
    daily_positions: pd.DataFrame,
    daily_exposure: pd.DataFrame,
    frozen_weights: pd.Series,
) -> pd.DataFrame:
    columns = DAILY_BOOK_EXPOSURES_REQUIRED_COLUMNS + [
        "book_weight_reference",
        "num_pairs_proxy",
        "coverage_status",
        "exposure_quality",
        "source_artifact",
    ]
    exposure = daily_exposure.copy()
    exposure = exposure[exposure["country"].astype(str).ne("portfolio")].copy()
    exposure["date"] = parse_date_series(exposure["date"]).dt.strftime("%Y-%m-%d")
    exposure["book"] = exposure["country"].astype(str)
    exposure["book_weight_reference"] = exposure["book"].map(lambda x: float(frozen_weights.get(str(x), np.nan)))
    exposure["gross_exposure"] = pd.to_numeric(exposure["gross_exposure"], errors="coerce")
    exposure["net_exposure"] = pd.to_numeric(exposure["net_exposure"], errors="coerce")
    exposure["long_notional"] = pd.to_numeric(exposure["long_exposure"], errors="coerce")
    exposure["short_notional"] = pd.to_numeric(exposure["short_exposure"], errors="coerce")
    exposure["num_pairs_proxy"] = pd.to_numeric(exposure["number_of_pairs"], errors="coerce")

    if daily_positions.empty:
        out = exposure.assign(
            num_active_pairs=np.nan,
            largest_pair_weight=np.nan,
            coverage_status="no_reconstructed_positions",
            source_artifact="exposure_proxy_only",
        )
        ordered = [col for col in columns if col in out.columns]
        return out.loc[:, ordered].sort_values(["date", "book"]).reset_index(drop=True)

    positions = daily_positions.copy()
    grouped = (
        positions.groupby(["date", "book"], dropna=False)
        .agg(
            num_active_pairs=("trade_id", "nunique"),
            largest_pair_weight=("weight", "max"),
            source_artifact=("source_artifact", lambda x: "; ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )

    out = exposure.merge(grouped, on=["date", "book"], how="left")
    out["coverage_status"] = np.where(
        out["num_active_pairs"].fillna(0.0) > 0.0,
        "paired_with_reconstructed_positions",
        "exposure_without_reconstructed_pairs",
    )
    ordered = columns
    return out.loc[:, ordered].sort_values(["date", "book"]).reset_index(drop=True)


def _build_daily_portfolio_exposures(
    *,
    daily_positions: pd.DataFrame,
    daily_book_exposures: pd.DataFrame,
) -> pd.DataFrame:
    columns = DAILY_PORTFOLIO_EXPOSURES_REQUIRED_COLUMNS + [
        "long_notional",
        "short_notional",
        "coverage_status",
    ]
    if daily_book_exposures.empty:
        return pd.DataFrame(columns=columns)

    positions = daily_positions.copy()
    pair_max = (
        positions.groupby("date", dropna=False)["weight"].max().rename("largest_pair_weight").reset_index()
        if not positions.empty
        else pd.DataFrame(columns=["date", "largest_pair_weight"])
    )
    book = daily_book_exposures.copy()
    out = (
        book.groupby("date", dropna=False)
        .agg(
            gross_exposure=("gross_exposure", "sum"),
            net_exposure=("net_exposure", "sum"),
            num_active_books=("book", lambda x: int(pd.Series(x).nunique())),
            num_active_pairs=("num_active_pairs", "sum"),
            largest_book_weight=("gross_exposure", "max"),
            long_notional=("long_notional", "sum"),
            short_notional=("short_notional", "sum"),
        )
        .reset_index()
    )
    out = out.merge(pair_max, on="date", how="left")
    out["coverage_status"] = np.where(
        pd.to_numeric(out["num_active_pairs"], errors="coerce").fillna(0.0) > 0.0,
        "portfolio_proxy_with_pair_overlay",
        "portfolio_proxy_only",
    )
    return out.loc[:, columns].sort_values("date").reset_index(drop=True)


def _build_position_summary(
    *,
    context: dict[str, Any],
    coverage: dict[str, Any],
    daily_positions: pd.DataFrame,
    daily_book_exposures: pd.DataFrame,
    daily_portfolio_exposures: pd.DataFrame,
) -> str:
    lines = [
        "# Core 4 Position Reconstruction Summary",
        "",
        f"- Analysis window: {context['analysis_start'].date()} to {context['analysis_end'].date()}",
        f"- Reconstructed daily position rows: {int(len(daily_positions))}",
        f"- Covered book-days with reconstructed pairs: {coverage.get('covered_book_days', 'n/a')}",
        f"- Exposure book-days without reconstructed pair rows: {coverage.get('residual_exposure_book_days', 'n/a')}",
        "",
        "## Coverage Notes",
        "- Book-level gross/net/long/short exposures come from the frozen `n_open_positions` proxy already used in the validation pack.",
        "- Pair-level notionals are allocated evenly across the reconstructed active trades for each book-day.",
        "- `current_z` is left as NaN because the frozen artifacts do not persist a trustworthy daily signal-state path.",
        "",
        "## Exposure Snapshot",
    ]
    if not daily_portfolio_exposures.empty:
        latest = daily_portfolio_exposures.iloc[-1]
        lines.append(
            f"- Latest portfolio date: {latest['date']} | gross={_fmt_number(latest['gross_exposure'])} | "
            f"net={_fmt_number(latest['net_exposure'])} | active_pairs={_fmt_number(latest['num_active_pairs'])}"
        )
    else:
        lines.append("- Portfolio exposure summary unavailable.")

    if not daily_book_exposures.empty:
        by_book = (
            daily_book_exposures.groupby("book", dropna=False)
            .agg(
                avg_gross=("gross_exposure", "mean"),
                avg_pairs=("num_active_pairs", "mean"),
                max_pair_weight=("largest_pair_weight", "max"),
            )
            .reset_index()
            .sort_values("book")
        )
        lines.extend(["", "## Average Book Exposure"])
        for row in by_book.itertuples(index=False):
            lines.append(
                f"- {row.book}: avg_gross={_fmt_number(row.avg_gross)} | avg_pairs={_fmt_number(row.avg_pairs)} | "
                f"max_pair_weight={_fmt_number(row.max_pair_weight)}"
            )

    missing = sorted(set(filter(None, daily_positions.get("non_reconstructible_fields", pd.Series(dtype=str)).astype(str).tolist())))
    lines.extend(["", "## Non-Reconstructible Fields"])
    for item in missing:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"
