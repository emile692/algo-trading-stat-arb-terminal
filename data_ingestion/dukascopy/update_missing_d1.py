from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_PATH = PROJECT_ROOT / "data_ingestion" / "dukascopy" / "instruments_all_from_readme.csv"
RAW_D1_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
REPORT_ROOT = PROJECT_ROOT / "data" / "reports" / "dukascopy_incremental"


def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w\-\.]", "", name)
    return name


def find_node_dir() -> Path:
    env_dir = os.environ.get("DUKASCOPY_NODE_DIR", "").strip()
    if env_dir:
        cand = Path(env_dir)
        if (cand / "node.exe").exists() and (cand / "npx.cmd").exists():
            return cand

    node_on_path = shutil.which("node")
    npx_on_path = shutil.which("npx") or shutil.which("npx.cmd")
    if node_on_path and npx_on_path:
        return Path(node_on_path).resolve().parent

    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
    winget_root = local_appdata / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        candidates = sorted(
            winget_root.glob("OpenJS.NodeJS.LTS_*/*"),
            key=lambda p: p.name,
            reverse=True,
        )
        for cand in candidates:
            if (cand / "node.exe").exists() and (cand / "npx.cmd").exists():
                return cand

    fallback = local_appdata / "Programs" / "nodejs"
    if (fallback / "node.exe").exists() and (fallback / "npx.cmd").exists():
        return fallback

    raise FileNotFoundError(
        "Node.js introuvable. Installe Node LTS ou renseigne DUKASCOPY_NODE_DIR."
    )


def build_subprocess_env(node_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{node_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def format_date(value: date) -> str:
    return value.isoformat()


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_existing_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError(f"{path} missing required column: datetime")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def _safe_read_dukascopy_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    expected = {"timestamp", "open", "high", "low", "close"}
    if not expected.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.drop(columns=["timestamp"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return df.reset_index(drop=True)


def merge_bid_ask(
    bid: pd.DataFrame | None,
    ask: pd.DataFrame | None,
) -> pd.DataFrame:
    if bid is None and ask is None:
        return pd.DataFrame()

    if bid is None:
        out = ask.copy()
        out = out.rename(columns={c: f"ask_{c}" for c in ["open", "high", "low", "close"]})
        for c in ["open", "high", "low", "close"]:
            out[c] = out[f"ask_{c}"]
        out["spread_close"] = 0.0
        return out

    if ask is None:
        out = bid.copy()
        out = out.rename(columns={c: f"bid_{c}" for c in ["open", "high", "low", "close"]})
        for c in ["open", "high", "low", "close"]:
            out[c] = out[f"bid_{c}"]
        out["spread_close"] = 0.0
        return out

    bid_ = bid.rename(columns={c: f"bid_{c}" for c in ["open", "high", "low", "close"]})
    ask_ = ask.rename(columns={c: f"ask_{c}" for c in ["open", "high", "low", "close"]})

    merged = pd.merge(bid_, ask_, on="datetime", how="inner")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    for c in ["open", "high", "low", "close"]:
        merged[c] = 0.5 * (merged[f"bid_{c}"] + merged[f"ask_{c}"])

    merged["spread_close"] = merged["ask_close"] - merged["bid_close"]
    return merged


def align_columns(existing: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_cols = list(existing.columns)
    extra_new_cols = [c for c in new_rows.columns if c not in existing_cols]
    all_cols = existing_cols + extra_new_cols

    existing_aligned = existing.reindex(columns=all_cols)
    new_aligned = new_rows.reindex(columns=all_cols)
    return existing_aligned, new_aligned


@dataclass(frozen=True)
class InstrumentCandidate:
    asset: str
    instrument_name: str
    instrument_id: str
    category_id: str


@dataclass(frozen=True)
class UpdateTask:
    asset: str
    asset_path: Path
    instrument_name: str
    instrument_id: str
    category_id: str
    start_date: date
    end_date: date
    last_date_before: date


@dataclass
class UpdateResult:
    asset: str
    instrument_id: str
    category_id: str
    start_date: str
    end_date: str
    last_date_before: str
    last_date_after: str
    rows_added: int
    status: str
    message: str


class DukascopyUpdater:
    def __init__(
        self,
        end_date: date,
        workers: int,
        limit: int | None,
        retry_count: int,
        retry_pause_ms: int,
        timeout_seconds: int,
        dry_run: bool,
    ) -> None:
        self.end_date = end_date
        self.workers = workers
        self.limit = limit
        self.retry_count = retry_count
        self.retry_pause_ms = retry_pause_ms
        self.timeout_seconds = timeout_seconds
        self.dry_run = dry_run

        self.node_dir = find_node_dir()
        self.env = build_subprocess_env(self.node_dir)
        self.npx_cmd = self.node_dir / "npx.cmd"

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = REPORT_ROOT / run_id
        self.download_dir = self.run_dir / "raw"
        self.resolve_dir = self.run_dir / "resolve"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.resolve_dir.mkdir(parents=True, exist_ok=True)

        self.meta = self._load_metadata()
        self.candidates_by_asset = self._build_candidates_by_asset()

    def _load_metadata(self) -> pd.DataFrame:
        if not META_PATH.exists():
            raise FileNotFoundError(f"Metadata not found: {META_PATH}")

        meta = pd.read_csv(META_PATH)
        required = {"instrument_name", "id", "category_id"}
        missing = required - set(meta.columns)
        if missing:
            raise RuntimeError(f"Missing metadata columns: {sorted(missing)}")

        meta = meta.copy()
        meta["asset"] = meta["instrument_name"].astype(str).map(sanitize_filename)
        return meta

    def _build_candidates_by_asset(self) -> dict[str, list[InstrumentCandidate]]:
        out: dict[str, list[InstrumentCandidate]] = {}
        for row in self.meta.itertuples(index=False):
            cand = InstrumentCandidate(
                asset=str(row.asset),
                instrument_name=str(row.instrument_name),
                instrument_id=str(row.id),
                category_id=str(row.category_id),
            )
            out.setdefault(cand.asset, []).append(cand)
        return out

    def build_tasks(self) -> list[UpdateTask]:
        if not RAW_D1_PATH.exists():
            raise FileNotFoundError(f"Raw D1 folder not found: {RAW_D1_PATH}")

        asset_paths = sorted(RAW_D1_PATH.glob("*.csv"))
        tasks: list[UpdateTask] = []

        for asset_path in asset_paths:
            existing = load_existing_series(asset_path)
            if existing.empty:
                continue

            last_date_before = existing["datetime"].max().date()
            start_date = last_date_before + timedelta(days=1)
            if start_date > self.end_date:
                continue

            candidates = self.candidates_by_asset.get(asset_path.stem, [])
            if not candidates:
                print(f"[WARN] No metadata match for asset={asset_path.stem}")
                continue

            chosen = self.resolve_candidate(asset_path, existing, candidates)
            tasks.append(
                UpdateTask(
                    asset=asset_path.stem,
                    asset_path=asset_path,
                    instrument_name=chosen.instrument_name,
                    instrument_id=chosen.instrument_id,
                    category_id=chosen.category_id,
                    start_date=start_date,
                    end_date=self.end_date,
                    last_date_before=last_date_before,
                )
            )

        if self.limit is not None:
            tasks = tasks[: self.limit]

        return tasks

    def resolve_candidate(
        self,
        asset_path: Path,
        existing: pd.DataFrame,
        candidates: list[InstrumentCandidate],
    ) -> InstrumentCandidate:
        if len(candidates) == 1:
            return candidates[0]

        compare_end = existing["datetime"].max().date()
        compare_start = max(existing["datetime"].min().date(), compare_end - timedelta(days=21))
        reference = existing.loc[
            existing["datetime"].dt.date.between(compare_start, compare_end),
            ["datetime", "close"],
        ].copy()
        reference["datetime"] = reference["datetime"].dt.normalize()

        scored: list[tuple[float, InstrumentCandidate]] = []

        for candidate in candidates:
            merged = self.download_and_merge(
                instrument_id=candidate.instrument_id,
                start_date=compare_start,
                end_date=compare_end,
                output_dir=self.resolve_dir,
            )
            if merged.empty or "close" not in merged.columns:
                continue

            probe = merged[["datetime", "close"]].copy()
            probe["datetime"] = pd.to_datetime(probe["datetime"], errors="coerce").dt.normalize()

            overlap = reference.merge(probe, on="datetime", how="inner", suffixes=("_existing", "_probe"))
            if overlap.empty:
                continue

            score = (
                (overlap["close_existing"].astype(float) - overlap["close_probe"].astype(float))
                .abs()
                .mean()
            )
            scored.append((safe_float(score), candidate))

        if scored:
            scored.sort(key=lambda x: x[0])
            best = scored[0][1]
            print(
                f"[INFO] Ambiguous asset resolved: {asset_path.stem} -> {best.instrument_id} ({best.category_id})"
            )
            return best

        fallback = sorted(candidates, key=lambda c: (c.category_id, c.instrument_id))[0]
        print(
            f"[WARN] Ambiguous asset unresolved, fallback used: {asset_path.stem} -> "
            f"{fallback.instrument_id} ({fallback.category_id})"
        )
        return fallback

    def run_cli_download(
        self,
        instrument_id: str,
        start_date: date,
        end_date: date,
        price_type: str,
        output_dir: Path,
    ) -> Path | None:
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.npx_cmd),
            "dukascopy-node",
            "-i",
            instrument_id,
            "-from",
            format_date(start_date),
            "-to",
            format_date(end_date),
            "-t",
            "d1",
            "-p",
            price_type,
            "-f",
            "csv",
            "-dir",
            str(output_dir),
            "-r",
            str(self.retry_count),
            "-rp",
            str(self.retry_pause_ms),
            "-re",
            "-fr",
            "--silent",
        ]

        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )

        expected = output_dir / (
            f"{instrument_id}-d1-{price_type}-{format_date(start_date)}-{format_date(end_date)}.csv"
        )
        if expected.exists():
            return expected

        if proc.returncode != 0:
            output = (proc.stdout or "").strip()
            raise RuntimeError(
                f"dukascopy-node failed for {instrument_id}/{price_type} "
                f"({proc.returncode}): {output[-500:]}"
            )

        return None

    def download_and_merge(
        self,
        instrument_id: str,
        start_date: date,
        end_date: date,
        output_dir: Path,
    ) -> pd.DataFrame:
        bid_path = self.run_cli_download(
            instrument_id=instrument_id,
            start_date=start_date,
            end_date=end_date,
            price_type="bid",
            output_dir=output_dir,
        )
        ask_path = self.run_cli_download(
            instrument_id=instrument_id,
            start_date=start_date,
            end_date=end_date,
            price_type="ask",
            output_dir=output_dir,
        )

        bid_df = _safe_read_dukascopy_csv(bid_path)
        ask_df = _safe_read_dukascopy_csv(ask_path)

        if bid_df is not None and bid_df.empty:
            bid_df = None
        if ask_df is not None and ask_df.empty:
            ask_df = None

        merged = merge_bid_ask(bid_df, ask_df)
        if merged.empty:
            return merged

        merged = merged.sort_values("datetime").drop_duplicates("datetime", keep="last")
        merged = merged.dropna(subset=["close"])
        merged = merged[merged["close"] > 0]
        return merged.reset_index(drop=True)

    def update_one(self, task: UpdateTask) -> UpdateResult:
        base = UpdateResult(
            asset=task.asset,
            instrument_id=task.instrument_id,
            category_id=task.category_id,
            start_date=format_date(task.start_date),
            end_date=format_date(task.end_date),
            last_date_before=format_date(task.last_date_before),
            last_date_after=format_date(task.last_date_before),
            rows_added=0,
            status="skipped",
            message="",
        )

        if self.dry_run:
            base.status = "dry_run"
            base.message = (
                f"Would download {task.instrument_id} from {base.start_date} to {base.end_date}"
            )
            return base

        try:
            merged_new = self.download_and_merge(
                instrument_id=task.instrument_id,
                start_date=task.start_date,
                end_date=task.end_date,
                output_dir=self.download_dir,
            )
            if merged_new.empty:
                base.status = "no_data"
                base.message = "No new merged rows returned by Dukascopy"
                return base

            existing = load_existing_series(task.asset_path)
            existing, merged_new = align_columns(existing, merged_new)

            before_rows = len(existing)
            combined = pd.concat([existing, merged_new], ignore_index=True)
            combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
            combined = combined.dropna(subset=["datetime"])
            combined = combined.sort_values("datetime").drop_duplicates("datetime", keep="last")

            if "close" in combined.columns:
                combined["close"] = pd.to_numeric(combined["close"], errors="coerce")
                combined = combined.dropna(subset=["close"])
                combined = combined[combined["close"] > 0]

            combined = combined.reset_index(drop=True)
            combined["datetime"] = combined["datetime"].dt.strftime("%Y-%m-%d")
            combined.to_csv(task.asset_path, index=False)

            after_rows = len(combined)
            last_after = parse_date(str(combined["datetime"].iloc[-1]))

            base.rows_added = max(0, after_rows - before_rows)
            base.last_date_after = format_date(last_after)
            base.status = "updated" if base.rows_added > 0 else "dedup_only"
            base.message = f"{base.rows_added} rows added"
            return base
        except Exception as exc:  # noqa: BLE001
            base.status = "failed"
            base.message = str(exc)
            return base

    def run(self) -> list[UpdateResult]:
        tasks = self.build_tasks()
        print(
            f"[INFO] Node dir: {self.node_dir}\n"
            f"[INFO] End date: {format_date(self.end_date)}\n"
            f"[INFO] Assets to refresh: {len(tasks)}\n"
            f"[INFO] Download dir: {self.download_dir}"
        )

        if not tasks:
            return []

        results: list[UpdateResult] = []
        completed = 0

        with cf.ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_map = {executor.submit(self.update_one, task): task for task in tasks}
            for future in cf.as_completed(future_map):
                result = future.result()
                results.append(result)
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {result.asset} -> {result.status} "
                    f"({result.last_date_before} -> {result.last_date_after}, +{result.rows_added})"
                )

        self.write_report(results)
        return results

    def write_report(self, results: Iterable[UpdateResult]) -> None:
        df = pd.DataFrame(asdict(r) for r in results)
        report_path = self.run_dir / "update_report.csv"
        df.to_csv(report_path, index=False)
        print(f"[OK] Report saved: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge uniquement la partie manquante des séries D1 Dukascopy."
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Date cible inclusive au format YYYY-MM-DD. Défaut: aujourd'hui.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Nombre de workers parallèles. Défaut: 6.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite le nombre d'assets à rafraîchir pour un run de test.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Nombre de retries dukascopy-node par requête.",
    )
    parser.add_argument(
        "--retry-pause-ms",
        type=int,
        default=1500,
        help="Pause entre retries pour dukascopy-node.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout d'une invocation CLI bid/ask.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construit le plan de refresh sans télécharger.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = parse_date(args.end_date)

    updater = DukascopyUpdater(
        end_date=end_date,
        workers=max(1, int(args.workers)),
        limit=args.limit,
        retry_count=max(0, int(args.retries)),
        retry_pause_ms=max(0, int(args.retry_pause_ms)),
        timeout_seconds=max(30, int(args.timeout_seconds)),
        dry_run=bool(args.dry_run),
    )
    results = updater.run()

    if not results:
        print("[OK] Nothing to update.")
        return

    status_counts = pd.Series([r.status for r in results]).value_counts().to_dict()
    print(f"[OK] Status counts: {status_counts}")


if __name__ == "__main__":
    main()
