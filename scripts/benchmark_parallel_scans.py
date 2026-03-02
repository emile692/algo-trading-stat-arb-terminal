from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import pandas as pd
from joblib import Parallel, cpu_count, delayed
import sys

# Ensure local project imports work when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.inline_scanner as inline_scanner
import utils.scanner as scanner
from utils.inline_scanner import InlineScannerConfig, build_scans_inline


def _date_chunks(start: str, end: str, freq: str, chunk_days: int) -> list[tuple[str, str]]:
    dates = pd.date_range(start, end, freq=freq)
    if len(dates) == 0:
        return []
    chunks: list[tuple[str, str]] = []
    for i in range(0, len(dates), chunk_days):
        s = dates[i].strftime("%Y-%m-%d")
        e = dates[min(i + chunk_days - 1, len(dates) - 1)].strftime("%Y-%m-%d")
        chunks.append((s, e))
    return chunks


def parallel_build_scans_inline(
    universes: list[str],
    start_date: str,
    end_date: str,
    freq: str,
    cfg: InlineScannerConfig,
    n_jobs: int | None = None,
    chunk_days: int = 30,
) -> pd.DataFrame:
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)

    chunks = _date_chunks(start_date, end_date, freq, chunk_days)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(build_scans_inline)(
            universes=universes,
            start_date=s,
            end_date=e,
            freq=freq,
            cfg=cfg,
        )
        for s, e in chunks
    )
    results = [r for r in results if (r is not None and len(r) > 0)]
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


class PerfCollector:
    def __init__(self) -> None:
        self.time_sec: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def track(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.time_sec[name] += time.perf_counter() - t0
            self.calls[name] += 1

    def as_dict(self) -> dict:
        return {
            "time_sec": dict(sorted(self.time_sec.items(), key=lambda kv: kv[1], reverse=True)),
            "calls": dict(self.calls),
            "counts": dict(self.counts),
        }


def _count_pairs_from_scan_universe_call(args, kwargs, _out, collector: PerfCollector) -> None:
    price_df = None
    if args:
        price_df = args[0]
    elif "price_df" in kwargs:
        price_df = kwargs["price_df"]
    if price_df is None:
        return
    n = int(price_df.shape[1])
    collector.counts["pairs_tested_total"] = collector.counts.get("pairs_tested_total", 0) + (n * (n - 1) // 2)


def _patch(
    obj: object,
    attr: str,
    collector: PerfCollector,
    track_name: str,
    extra: Callable | None = None,
) -> tuple[object, str, Callable]:
    original = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        with collector.track(track_name):
            out = original(*args, **kwargs)
        if extra is not None:
            extra(args, kwargs, out, collector)
        return out

    setattr(obj, attr, wrapped)
    return obj, attr, original


def _patch_if_exists(
    obj: object,
    attr: str,
    collector: PerfCollector,
    track_name: str,
    extra: Callable | None = None,
) -> tuple[object, str, Callable] | None:
    if not hasattr(obj, attr):
        return None
    return _patch(obj, attr, collector, track_name, extra=extra)


def _unpatch_all(patches: list[tuple[object, str, Callable]]) -> None:
    for obj, attr, original in patches:
        setattr(obj, attr, original)


def _pair_key(df: pd.DataFrame) -> pd.Series:
    a1 = df["asset_1"].astype(str).str.upper()
    a2 = df["asset_2"].astype(str).str.upper()
    lo = a1.where(a1 <= a2, a2)
    hi = a2.where(a1 <= a2, a1)
    d = pd.to_datetime(df["scan_date"]).dt.normalize().dt.strftime("%Y-%m-%d")
    u = df["universe"].astype(str)
    return d + "|" + u + "|" + lo + "|" + hi


def _overlap_stats(before_keys: set[str], after_keys: set[str], prefix: str) -> dict:
    inter = before_keys & after_keys
    only_old = before_keys - after_keys
    only_new = after_keys - before_keys
    return {
        f"{prefix}_before": len(before_keys),
        f"{prefix}_after": len(after_keys),
        f"{prefix}_overlap_count": len(inter),
        f"{prefix}_overlap_ratio_vs_before": (len(inter) / len(before_keys)) if before_keys else None,
        f"{prefix}_overlap_ratio_vs_after": (len(inter) / len(after_keys)) if after_keys else None,
        f"{prefix}_missing_after_count": len(only_old),
        f"{prefix}_new_after_count": len(only_new),
        f"{prefix}_missing_after_examples": sorted(list(only_old))[:20],
        f"{prefix}_new_after_examples": sorted(list(only_new))[:20],
    }


def compare_results(current_df: pd.DataFrame, baseline_path: Path) -> dict:
    if not baseline_path.exists():
        return {"error": f"baseline file not found: {baseline_path}"}

    baseline_df = pd.read_parquet(baseline_path)

    cur_all_keys = set(_pair_key(current_df))
    base_all_keys = set(_pair_key(baseline_df))

    cur_elig = current_df[current_df["eligibility"] == "ELIGIBLE"].copy()
    base_elig = baseline_df[baseline_df["eligibility"] == "ELIGIBLE"].copy()

    cur_keys = set(_pair_key(cur_elig))
    base_keys = set(_pair_key(base_elig))
    out = {"baseline_path": str(baseline_path)}
    out.update(_overlap_stats(base_all_keys, cur_all_keys, prefix="retained_pairs"))
    out.update(_overlap_stats(base_keys, cur_keys, prefix="eligible_pairs"))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universes", type=str, required=True, help="Comma-separated category ids, e.g. france")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--freq", type=str, default="B")
    p.add_argument("--chunk-days", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--lookback-days", type=int, default=504)
    p.add_argument("--min-obs", type=int, default=100)
    p.add_argument("--liquidity-lookback", type=int, default=20)
    p.add_argument("--liquidity-min-moves", type=float, default=0.0)
    p.add_argument("--label", type=str, default="run")
    p.add_argument("--out-dir", type=str, default="data/scanner/benchmarks")
    p.add_argument("--profile", action="store_true", help="Collect detailed timings (requires n_jobs=1)")
    p.add_argument("--compare-with", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = [u.strip() for u in args.universes.split(",") if u.strip()]
    cfg = InlineScannerConfig(
        raw_data_path=Path("data/raw/d1"),
        asset_registry_path=Path("data/asset_registry.csv"),
        lookback_days=args.lookback_days,
        min_obs=args.min_obs,
        liquidity_lookback=args.liquidity_lookback,
        liquidity_min_moves=args.liquidity_min_moves,
    )

    collector = PerfCollector()
    patches: list[tuple[object, str, Callable]] = []

    if args.profile:
        if args.n_jobs != 1:
            raise ValueError("--profile currently supports only --n-jobs 1")

        for maybe_patch in (
            _patch_if_exists(inline_scanner, "load_universe_assets", collector, "load_universe_assets"),
            _patch_if_exists(inline_scanner, "load_price_asof_norm", collector, "load_price_asof_norm"),
            _patch_if_exists(inline_scanner, "scan_universe_asof", collector, "scan_universe_asof"),
            _patch_if_exists(
                inline_scanner,
                "scan_universe",
                collector,
                "scan_universe",
                extra=_count_pairs_from_scan_universe_call,
            ),
            _patch_if_exists(scanner, "scan_universe", collector, "scan_universe_core"),
            _patch_if_exists(scanner, "scan_pair_multi_window", collector, "scan_pair_multi_window"),
            _patch_if_exists(scanner, "scan_pair_window", collector, "scan_pair_window"),
            _patch_if_exists(scanner, "_scan_pair_window_aligned", collector, "scan_pair_window_aligned"),
            _patch_if_exists(scanner, "compute_hedge_ratio", collector, "compute_hedge_ratio"),
            _patch_if_exists(scanner, "compute_adf", collector, "compute_adf"),
            _patch_if_exists(scanner, "compute_coint", collector, "compute_coint"),
            _patch_if_exists(scanner, "compute_half_life", collector, "compute_half_life"),
            _patch_if_exists(scanner, "_fast_ols_beta", collector, "fast_ols_beta"),
            _patch_if_exists(scanner, "_fast_half_life", collector, "fast_half_life"),
            _patch_if_exists(scanner, "_fast_corr", collector, "fast_corr"),
        ):
            if maybe_patch is not None:
                patches.append(maybe_patch)

    t0 = time.perf_counter()
    try:
        scans = parallel_build_scans_inline(
            universes=universes,
            start_date=args.start,
            end_date=args.end,
            freq=args.freq,
            cfg=cfg,
            n_jobs=args.n_jobs,
            chunk_days=args.chunk_days,
        )
    finally:
        _unpatch_all(patches)
    total_sec = time.perf_counter() - t0

    if not scans.empty:
        scans = scans.sort_values(["scan_date", "universe", "asset_1", "asset_2"]).reset_index(drop=True)

    parquet_path = out_dir / f"{args.label}.parquet"
    scans.to_parquet(parquet_path, index=False)

    metrics = {
        "label": args.label,
        "universes": universes,
        "start": args.start,
        "end": args.end,
        "freq": args.freq,
        "chunk_days": args.chunk_days,
        "n_jobs": args.n_jobs,
        "profile": bool(args.profile),
        "total_time_sec": total_sec,
        "scan_rows": int(len(scans)),
        "scan_dates": int(scans["scan_date"].nunique()) if not scans.empty else 0,
        "retained_pairs": int(len(scans)),
        "eligible_pairs": int((scans["eligibility"] == "ELIGIBLE").sum()) if not scans.empty else 0,
        "watch_pairs": int((scans["eligibility"] == "WATCH").sum()) if not scans.empty else 0,
        "unique_pairs_any_date": int(_pair_key(scans).nunique()) if not scans.empty else 0,
        "output_parquet": str(parquet_path),
    }

    if args.profile:
        metrics["profiling"] = collector.as_dict()

    if args.compare_with:
        metrics["comparison"] = compare_results(scans, Path(args.compare_with))

    json_path = out_dir / f"{args.label}.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
