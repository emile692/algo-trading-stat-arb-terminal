from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.country_research_pipeline as crp
from utils.country_research_pipeline import PipelineOptions, run_country_research_pipeline


BASE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
ASSET_REGISTRY_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"
SUBSET_INPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "us_subset_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized country research pipeline on a documented US "
            "liquid-history subset. The main US universe is too large for the "
            "current exhaustive scanner, so this script creates a temporary "
            "asset registry for category_id=us_liquidN and reuses the existing pipeline."
        )
    )
    parser.add_argument("--subset-size", type=int, default=100, help="Number of US assets to retain.")
    parser.add_argument("--start", default=crp.DEFAULT_START)
    parser.add_argument("--end", default=crp.DEFAULT_END)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "experiments")
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--max-ablation-variants", type=int, default=5)
    parser.add_argument("--rebuild-scans", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def _read_price_coverage(asset: str) -> dict[str, object] | None:
    fp = BASE_DATA_PATH / f"{asset}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(
            fp,
            usecols=lambda c: str(c).lower() in {"datetime", "date", "close"},
        )
    except Exception:
        return None
    date_col = next((c for c in df.columns if str(c).lower() in {"datetime", "date"}), None)
    close_col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if date_col is None or close_col is None:
        return None
    dt = pd.to_datetime(df[date_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    ret = close.pct_change()
    return {
        "asset": asset,
        "raw_file": str(fp),
        "n_rows": int(len(df)),
        "close_obs": int(close.notna().sum()),
        "first_date": dt.min(),
        "last_date": dt.max(),
        "nonzero_return_days": int((ret.fillna(0).abs() > 0).sum()),
    }


def build_us_subset_registry(subset_size: int, start: str, end: str) -> tuple[str, Path, pd.DataFrame, dict[str, object]]:
    if subset_size < 10:
        raise ValueError("--subset-size must be >= 10")

    registry = pd.read_csv(ASSET_REGISTRY_PATH)
    us = registry[registry["category_id"].astype(str).str.lower() == "us"].copy()
    if us.empty:
        raise RuntimeError("No category_id=us rows found in data/asset_registry.csv")

    coverage_rows = []
    for asset in us["asset"].astype(str):
        info = _read_price_coverage(asset)
        if info is not None:
            coverage_rows.append(info)
    coverage = pd.DataFrame(coverage_rows)
    if coverage.empty:
        raise RuntimeError("No usable US raw price files found.")

    end_ts = pd.Timestamp(end)
    coverage["last_date"] = pd.to_datetime(coverage["last_date"], errors="coerce")
    coverage["first_date"] = pd.to_datetime(coverage["first_date"], errors="coerce")
    coverage["ends_near_requested_end"] = coverage["last_date"] >= (end_ts - pd.Timedelta(days=45))
    coverage["starts_before_requested_start"] = coverage["first_date"] <= pd.Timestamp(start)

    ranked = coverage.sort_values(
        [
            "ends_near_requested_end",
            "close_obs",
            "nonzero_return_days",
            "first_date",
            "asset",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).head(int(subset_size))

    selected_assets = set(ranked["asset"].astype(str))
    subset_registry = us[us["asset"].astype(str).isin(selected_assets)].copy()
    subset_registry = subset_registry.merge(
        ranked[["asset", "close_obs", "first_date", "last_date", "nonzero_return_days"]],
        on="asset",
        how="left",
    ).sort_values(["close_obs", "nonzero_return_days", "asset"], ascending=[False, False, True])

    country_id = f"us_liquid{int(subset_size)}"
    output_registry = subset_registry[["asset", "symbol", "category_name", "category_name_clean", "category_id"]].copy()
    output_registry["category_name"] = f"United States liquid-history top {int(subset_size)}"
    output_registry["category_name_clean"] = "United States liquid-history subset"
    output_registry["category_id"] = country_id

    SUBSET_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = SUBSET_INPUT_DIR / f"{country_id}_asset_registry.csv"
    output_registry.to_csv(registry_path, index=False)

    selection = subset_registry[
        ["asset", "symbol", "close_obs", "first_date", "last_date", "nonzero_return_days"]
    ].copy()
    selection_path = SUBSET_INPUT_DIR / f"{country_id}_selection.csv"
    selection.to_csv(selection_path, index=False)

    metadata = {
        "source_country": "us",
        "pipeline_country_id": country_id,
        "subset_size_requested": int(subset_size),
        "subset_size_selected": int(len(output_registry)),
        "selection_rule": (
            "Rank US assets with raw price files by latest coverage near requested end, "
            "then close observations, non-zero return days, earliest first date, asset name."
        ),
        "start": start,
        "end": end,
        "registry_path": str(registry_path),
        "selection_path": str(selection_path),
        "methodological_limitation": (
            "This is not the full 546-name US universe. It is a compute-controlled "
            "liquid-history proxy needed because the exhaustive US scanner did not "
            "finish a 2025-Q1 smoke run within 30 minutes."
        ),
    }
    (SUBSET_INPUT_DIR / f"{country_id}_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return country_id, registry_path, selection, metadata


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    start = crp.SMOKE_START if args.smoke else args.start
    end = crp.SMOKE_END if args.smoke else args.end
    country_id, registry_path, selection, subset_metadata = build_us_subset_registry(
        subset_size=args.subset_size,
        start=start,
        end=end,
    )

    crp.ASSET_REGISTRY_PATH = registry_path
    options = PipelineOptions(
        country=country_id,
        start=args.start,
        end=args.end,
        output_root=args.output_root,
        output_suffix=args.output_suffix,
        skip_robustness=args.skip_robustness,
        smoke=args.smoke,
        rebuild_scans=args.rebuild_scans,
        max_ablation_variants=args.max_ablation_variants,
    )
    out_dir = run_country_research_pipeline(options)

    selection.to_csv(out_dir / "us_subset_selection.csv", index=False)
    (out_dir / "us_subset_metadata.json").write_text(
        json.dumps(subset_metadata, indent=2, default=str),
        encoding="utf-8",
    )
    logging.getLogger("us_subset_research").info("Output directory: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
