from __future__ import annotations

import argparse
import json
import subprocess
import time
import types
from pathlib import Path

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.scanner as scanner_new
from utils.inline_scanner import InlineScannerConfig, load_price_asof_norm, load_universe_assets


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=str, default="us")
    p.add_argument("--scan-date", type=str, default="2024-03-29")
    p.add_argument("--n-list", type=str, default="40,80,120")
    p.add_argument("--lookback-days", type=int, default=504)
    p.add_argument("--min-obs", type=int, default=100)
    p.add_argument("--liquidity-lookback", type=int, default=20)
    p.add_argument("--liquidity-min-moves", type=float, default=0.0)
    p.add_argument("--out-json", type=str, default="data/scanner/benchmarks/scalability_old_vs_new.json")
    return p.parse_args()


def _pair_keys(df: pd.DataFrame, eligible_only: bool = True) -> set[str]:
    if df is None or df.empty:
        return set()
    d = df[df["eligibility"] == "ELIGIBLE"] if eligible_only else df
    if d.empty:
        return set()
    a1 = d["asset_1"].astype(str).str.upper()
    a2 = d["asset_2"].astype(str).str.upper()
    lo = a1.where(a1 <= a2, a2)
    hi = a2.where(a1 <= a2, a1)
    return set((lo + "|" + hi).tolist())


def _load_old_scanner_module() -> types.ModuleType:
    src = subprocess.check_output(["git", "show", "HEAD:utils/scanner.py"], text=True)
    mod = types.ModuleType("scanner_old")
    exec(src, mod.__dict__)
    return mod


def main() -> None:
    args = _parse_args()
    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    scan_date = pd.Timestamp(args.scan_date)

    cfg = InlineScannerConfig(
        raw_data_path=Path("data/raw/d1"),
        asset_registry_path=Path("data/asset_registry.csv"),
        lookback_days=args.lookback_days,
        min_obs=args.min_obs,
        liquidity_lookback=args.liquidity_lookback,
        liquidity_min_moves=args.liquidity_min_moves,
    )

    scanner_old = _load_old_scanner_module()

    assets = sorted(set(load_universe_assets(cfg.asset_registry_path, args.universe)))
    series = {}
    for asset in assets:
        s = load_price_asof_norm(asset, scan_date, cfg)
        if s is not None:
            series[asset] = s

    prices_all = pd.DataFrame(series).dropna(how="all")
    available_assets = int(prices_all.shape[1])

    rows: list[dict] = []
    for n in n_list:
        if n > available_assets:
            continue
        prices = prices_all.iloc[:, :n]
        pairs = n * (n - 1) // 2

        t0 = time.perf_counter()
        old_df = scanner_old.scan_universe(prices, args.universe)
        old_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        new_df = scanner_new.scan_universe(prices, args.universe)
        new_sec = time.perf_counter() - t1

        old_elig = _pair_keys(old_df, eligible_only=True)
        new_elig = _pair_keys(new_df, eligible_only=True)
        inter = old_elig & new_elig

        rows.append(
            {
                "n_assets": n,
                "n_pairs": pairs,
                "old_time_sec": old_sec,
                "new_time_sec": new_sec,
                "speedup_x": (old_sec / new_sec) if new_sec > 0 else None,
                "eligible_old": len(old_elig),
                "eligible_new": len(new_elig),
                "eligible_overlap_count": len(inter),
                "eligible_overlap_ratio_vs_old": (len(inter) / len(old_elig)) if old_elig else None,
                "eligible_overlap_ratio_vs_new": (len(inter) / len(new_elig)) if new_elig else None,
            }
        )

    payload = {
        "universe": args.universe,
        "scan_date": str(scan_date.date()),
        "available_assets": available_assets,
        "rows": rows,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
