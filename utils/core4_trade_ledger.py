from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.core4_audit_common import (
    DEFAULT_AUDIT_OUTPUT_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DAILY_CACHE_DIR,
    TRADE_LEDGER_REQUIRED_COLUMNS,
    Core4AuditBaseOptions,
    load_core4_audit_context,
    non_reconstructible_fields_text,
    parse_date_series,
    resolve_path,
)
from utils.core4_validation_pack import discover_trade_ledgers


@dataclass(frozen=True)
class Core4TradeLedgerOptions(Core4AuditBaseOptions):
    output_root: Path = DEFAULT_AUDIT_OUTPUT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    daily_cache_dir: Path = DEFAULT_DAILY_CACHE_DIR


def reconstruct_core4_trade_ledger(
    options: Core4TradeLedgerOptions,
    *,
    project_root: Path,
) -> dict[str, Any]:
    context = load_core4_audit_context(options, project_root=project_root)
    trade_sources = discover_trade_ledgers(context["config"], project_root=project_root)
    ledger = _build_trade_ledger(
        trade_sources=trade_sources,
        allow_near_match=bool(options.allow_near_match),
    )
    summary_text = _build_trade_summary(ledger, trade_sources=trade_sources, allow_near_match=bool(options.allow_near_match))
    return {
        "context": context,
        "trade_sources": trade_sources,
        "trade_ledger": ledger,
        "trade_ledger_summary": summary_text,
    }


def run_core4_trade_ledger(
    options: Core4TradeLedgerOptions,
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bundle = reconstruct_core4_trade_ledger(options, project_root=project_root)
    output_dir = resolve_path(project_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle["trade_ledger"].to_csv(output_dir / "trade_ledger.csv", index=False)
    bundle["trade_sources"].to_csv(output_dir / "trade_ledger_sources.csv", index=False)
    (output_dir / "trade_ledger_summary.md").write_text(bundle["trade_ledger_summary"], encoding="utf-8")
    return bundle


def _build_trade_ledger(*, trade_sources: pd.DataFrame, allow_near_match: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in trade_sources.itertuples(index=False):
        status = str(row.status)
        if status == "exact":
            frame = _load_source_frame(
                source_path=Path(str(row.path)),
                requested_config=str(row.config_name),
                source_match_status="exact",
                source_config_name=str(row.config_name),
            )
            frames.append(_normalize_trade_frame(frame, book=str(row.book), requested_config=str(row.config_name)))
            continue

        if status == "near_match" and allow_near_match:
            closest = _extract_closest_config(str(row.notes))
            if closest:
                frame = _load_source_frame(
                    source_path=Path(str(row.path)),
                    requested_config=closest,
                    source_match_status="near_match",
                    source_config_name=closest,
                )
                frames.append(_normalize_trade_frame(frame, book=str(row.book), requested_config=str(row.config_name)))

    if not frames:
        return pd.DataFrame(columns=TRADE_LEDGER_REQUIRED_COLUMNS + _extra_trade_columns())

    ledger = pd.concat(frames, ignore_index=True, sort=False)
    ledger["open_date"] = parse_date_series(ledger["open_date"])
    ledger["close_date"] = parse_date_series(ledger["close_date"])
    ledger = ledger.drop_duplicates(
        subset=["book", "pair_id", "open_date", "close_date", "source_artifact"],
        keep="first",
    ).copy()
    ledger = ledger.sort_values(["open_date", "close_date", "book", "pair_id"]).reset_index(drop=True)
    ledger["trade_id"] = [f"core4_trade_{idx + 1:05d}" for idx in range(len(ledger))]
    ledger["open_date"] = ledger["open_date"].dt.strftime("%Y-%m-%d")
    ledger["close_date"] = ledger["close_date"].dt.strftime("%Y-%m-%d")
    ordered = TRADE_LEDGER_REQUIRED_COLUMNS + [col for col in _extra_trade_columns() if col in ledger.columns]
    return ledger.loc[:, ordered].reset_index(drop=True)


def _load_source_frame(
    *,
    source_path: Path,
    requested_config: str,
    source_match_status: str,
    source_config_name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    if "config_name" in frame.columns:
        frame = frame[frame["config_name"].astype(str).eq(str(requested_config))].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["__source_artifact"] = str(source_path)
    frame["__source_match_status"] = str(source_match_status)
    frame["__source_config_name"] = str(source_config_name)
    return frame


def _normalize_trade_frame(frame: pd.DataFrame, *, book: str, requested_config: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=TRADE_LEDGER_REQUIRED_COLUMNS + _extra_trade_columns())

    out = pd.DataFrame()
    out["book"] = pd.Series([str(book)] * len(frame), index=frame.index, dtype="object")
    out["pair_id"] = _series_or_blank(frame, "pair_id")
    out["leg_1"] = _series_or_blank(frame, "asset_left", fallback="asset_1")
    out["leg_2"] = _series_or_blank(frame, "asset_right", fallback="asset_2")
    out["open_date"] = _series_or_blank(frame, "entry_datetime")
    out["close_date"] = _series_or_blank(frame, "exit_datetime")
    out["holding_days"] = _series_or_nan(frame, "holding_days", fallback="duration_days")
    out["entry_z"] = _series_or_nan(frame, "entry_z", fallback="z_entry_observed")
    out["exit_z"] = _series_or_nan(frame, "exit_z")
    out["exit_reason"] = _series_or_blank(frame, "exit_reason_norm", fallback="exit_reason")
    out["pnl_gross"] = _series_or_nan(frame, "pnl", fallback="trade_pnl")
    out["pnl_net_before_borrow"] = out["pnl_gross"]
    out["estimated_borrow_cost"] = 0.0
    out["pnl_net_after_borrow"] = out["pnl_net_before_borrow"] - out["estimated_borrow_cost"]
    out["max_adverse_excursion"] = _series_or_nan(frame, "mae")
    out["max_favorable_excursion"] = _series_or_nan(frame, "mfe")
    out["source_artifact"] = _series_or_blank(frame, "__source_artifact")

    out["side"] = _series_or_blank(frame, "side")
    out["source_match_status"] = _series_or_blank(frame, "__source_match_status")
    out["requested_config_name"] = pd.Series([str(requested_config)] * len(frame), index=frame.index, dtype="object")
    out["source_config_name"] = _series_or_blank(frame, "__source_config_name")
    out["capital_at_entry"] = _series_or_nan(frame, "capital_at_entry")
    out["capital_at_exit"] = _series_or_nan(frame, "capital_at_exit")
    out["return_pct"] = _series_or_nan(frame, "return_pct", fallback="trade_return_isolated")
    out["entry_spread"] = _series_or_nan(frame, "entry_spread")
    out["exit_spread"] = _series_or_nan(frame, "exit_spread")
    out["beta_entry"] = _series_or_nan(frame, "beta_entry", fallback="beta")
    out["hedge_ratio"] = _series_or_nan(frame, "hedge_ratio")

    base_notes = []
    if out["source_match_status"].astype(str).eq("near_match").any():
        base_notes.append("Source ledger is a near-match proxy rather than the exact frozen config.")
    base_notes.append("Borrow cost is not modeled in the base trade ledger and is set to 0bps by construction.")
    out["approximation_notes"] = non_reconstructible_fields_text(base_notes)

    missing_fields = []
    if "mae" not in frame.columns:
        missing_fields.append("max_adverse_excursion")
    if "mfe" not in frame.columns:
        missing_fields.append("max_favorable_excursion")
    out["missing_or_approximated_fields"] = non_reconstructible_fields_text(missing_fields + ["estimated_borrow_cost_model"])

    out["open_date"] = parse_date_series(out["open_date"]).dt.strftime("%Y-%m-%d")
    out["close_date"] = parse_date_series(out["close_date"]).dt.strftime("%Y-%m-%d")
    out["holding_days"] = pd.to_numeric(out["holding_days"], errors="coerce")
    mask = out["holding_days"].isna()
    if mask.any():
        opened = pd.to_datetime(out.loc[mask, "open_date"], errors="coerce")
        closed = pd.to_datetime(out.loc[mask, "close_date"], errors="coerce")
        out.loc[mask, "holding_days"] = (closed - opened).dt.days

    ordered = [col for col in TRADE_LEDGER_REQUIRED_COLUMNS if col != "trade_id"] + [col for col in _extra_trade_columns() if col in out.columns]
    return out.loc[:, ordered]


def _build_trade_summary(ledger: pd.DataFrame, *, trade_sources: pd.DataFrame, allow_near_match: bool) -> str:
    lines = [
        "# Core 4 Trade Ledger Summary",
        "",
        f"- Trade-source rows discovered: {len(trade_sources)}",
        f"- Near-match sources allowed: {'yes' if allow_near_match else 'no'}",
    ]

    if ledger.empty:
        lines.extend(
            [
                "- No trade rows were reconstructed.",
                "",
                "## Remaining Gaps",
                "- No exact or approved near-match source file produced a usable frozen trade ledger.",
            ]
        )
        return "\n".join(lines) + "\n"

    pnl = pd.to_numeric(ledger["pnl_net_before_borrow"], errors="coerce")
    lines.extend(
        [
            f"- Total reconstructed trades: {int(len(ledger))}",
            f"- Books covered: {', '.join(sorted(ledger['book'].dropna().astype(str).unique().tolist()))}",
            f"- Total base PnL proxy: {_fmt_number(pnl.sum())}",
            "",
            "## Trades By Book",
        ]
    )

    by_book = (
        ledger.assign(pnl_value=pnl)
        .groupby("book", dropna=False)
        .agg(
            trades=("trade_id", "count"),
            hit_ratio=("pnl_value", lambda x: float((pd.to_numeric(x, errors='coerce').fillna(0.0) > 0.0).mean())),
            avg_trade_pnl=("pnl_value", "mean"),
            median_trade_pnl=("pnl_value", "median"),
        )
        .reset_index()
        .sort_values("book")
    )
    for row in by_book.itertuples(index=False):
        lines.append(
            f"- {row.book}: trades={int(row.trades)} | hit_ratio={_fmt_pct(row.hit_ratio)} | "
            f"avg_pnl={_fmt_number(row.avg_trade_pnl)} | median_pnl={_fmt_number(row.median_trade_pnl)}"
        )

    lines.extend(["", "## Exit Reasons"])
    exit_reason = (
        ledger.assign(pnl_value=pnl)
        .groupby("exit_reason", dropna=False)
        .agg(trades=("trade_id", "count"), pnl=("pnl_value", "sum"))
        .reset_index()
        .sort_values(["pnl", "trades"], ascending=[False, False])
    )
    for row in exit_reason.itertuples(index=False):
        label = row.exit_reason if str(row.exit_reason).strip() else "blank"
        lines.append(f"- {label}: trades={int(row.trades)} | pnl={_fmt_number(row.pnl)}")

    holding = pd.to_numeric(ledger["holding_days"], errors="coerce").dropna()
    lines.extend(
        [
            "",
            "## Holding Distribution",
            f"- mean={_fmt_number(holding.mean())} | median={_fmt_number(holding.median())} | "
            f"p90={_fmt_number(holding.quantile(0.9))} | max={_fmt_number(holding.max())}"
            if not holding.empty
            else "- Holding-day distribution unavailable.",
        ]
    )

    worst = ledger.assign(pnl_value=pnl).sort_values("pnl_value", ascending=True).head(10)
    best = ledger.assign(pnl_value=pnl).sort_values("pnl_value", ascending=False).head(10)
    lines.extend(["", "## Worst 10 Trades"])
    for row in worst.itertuples(index=False):
        lines.append(
            f"- {row.trade_id} | {row.book} | {row.pair_id} | open={row.open_date} | close={row.close_date} | pnl={_fmt_number(row.pnl_value)}"
        )
    lines.extend(["", "## Best 10 Trades"])
    for row in best.itertuples(index=False):
        lines.append(
            f"- {row.trade_id} | {row.book} | {row.pair_id} | open={row.open_date} | close={row.close_date} | pnl={_fmt_number(row.pnl_value)}"
        )

    approximate = sorted(set(filter(None, ledger["missing_or_approximated_fields"].astype(str).tolist())))
    notes = sorted(set(filter(None, ledger["approximation_notes"].astype(str).tolist())))
    lines.extend(["", "## Fields Missing Or Approximated"])
    for item in approximate:
        lines.append(f"- {item}")
    lines.extend(["", "## Approximation Notes"])
    for item in notes:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _extract_closest_config(notes: str) -> str | None:
    match = re.search(r"closest configs=(\[[^\]]*\])", notes)
    if not match:
        return None
    try:
        values = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    if not values:
        return None
    return str(values[0]).strip()


def _series_or_blank(frame: pd.DataFrame, column: str, fallback: str | None = None) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("").astype(str)
    if fallback and fallback in frame.columns:
        return frame[fallback].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def _series_or_nan(frame: pd.DataFrame, column: str, fallback: str | None = None) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    if fallback and fallback in frame.columns:
        return pd.to_numeric(frame[fallback], errors="coerce")
    return pd.Series([np.nan] * len(frame), index=frame.index, dtype="float64")


def _extra_trade_columns() -> list[str]:
    return [
        "side",
        "source_match_status",
        "requested_config_name",
        "source_config_name",
        "capital_at_entry",
        "capital_at_exit",
        "return_pct",
        "entry_spread",
        "exit_spread",
        "beta_entry",
        "hedge_ratio",
        "approximation_notes",
        "missing_or_approximated_fields",
    ]


def _fmt_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{100.0 * float(value):.2f}%"
