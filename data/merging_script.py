from __future__ import annotations

from tqdm import tqdm

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSTRUMENT_PATH = (
    PROJECT_ROOT
    / "data_ingestion"
    / "dukascopy"
    / "instruments_all_from_readme.csv"
)

RAW_TEMP_FOLDER = (
    PROJECT_ROOT
    / "data_ingestion"
    / "dukascopy"
    / "download"
    / "raw"
)

RAW_OUT_FOLDER = PROJECT_ROOT / "data" / "raw"


# ============================================================
# Filename parser
# Example:
# 0005hkhkd-d1-bid-2018-01-02-2018-12-31.csv
# 0005hkhkd-d1-ask-2025-01-01-2025-12-16T15-12.csv
# ============================================================
FNAME_RE = re.compile(
    r"^(?P<instrument>.+)-"
    r"(?P<tf>[a-zA-Z0-9]+)-"
    r"(?P<side>bid|ask)-"
    r"(?P<start>\d{4}-\d{2}-\d{2})-"
    r"(?P<end>\d{4}-\d{2}-\d{2}(?:T\d{2}-\d{2})?)"
    r"\.csv$"
)


# ============================================================
# Helpers
# ============================================================
def sanitize_filename(name: str) -> str:
    """
    Make a string safe for filesystem usage (Windows/Linux).
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w\-\.]", "", name)
    return name


@dataclass(frozen=True)
class DukascopyChunkKey:
    instrument: str
    timeframe: str
    start: str
    end: str


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[WARN] Empty CSV skipped: {path.name}")
        return pd.DataFrame()

    if df.empty:
        print(f"[WARN] Empty CSV skipped: {path.name}")
        return pd.DataFrame()

    expected = {"timestamp", "open", "high", "low", "close"}
    missing = expected - set(df.columns)
    if missing:
        print(f"[WARN] Missing columns {missing} in {path.name}, skipped")
        return pd.DataFrame()

    df = df.copy()
    df["datetime"] = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .dt.tz_convert(None)
    )
    df = df.drop(columns=["timestamp"])

    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])

    if df.empty:
        return pd.DataFrame()

    return df.reset_index(drop=True)


def _merge_bid_ask(
    bid: Optional[pd.DataFrame],
    ask: Optional[pd.DataFrame],
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


def _load_instrument_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    instr_col = next(
        (cols[c] for c in ["instrument", "instrument_id", "id", "dukascopy_id"] if c in cols),
        None,
    )
    label_col = next(
        (cols[c] for c in ["symbol", "ticker", "name", "instrument_name"] if c in cols),
        None,
    )

    if instr_col is None or label_col is None:
        return {}

    return {
        str(r[instr_col]).strip(): str(r[label_col]).strip()
        for _, r in df.iterrows()
        if pd.notna(r[instr_col]) and pd.notna(r[label_col])
    }


# ============================================================
# Main pipeline
# ============================================================
def merge_dukascopy_folder(
    input_folder: Path = RAW_TEMP_FOLDER,
    output_folder: Path = RAW_OUT_FOLDER,
    mapping_path: Path = INSTRUMENT_PATH,
) -> None:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    mapping = _load_instrument_mapping(mapping_path)

    files = sorted(
        p for p in input_folder.rglob("*.csv") if FNAME_RE.match(p.name)
    )
    if not files:
        raise RuntimeError(f"No dukascopy CSV files found in {input_folder}")

    chunk_groups: dict[DukascopyChunkKey, dict[str, Path]] = {}

    for f in tqdm(files, desc="Indexing Dukascopy chunks", unit="file"):
        gd = FNAME_RE.match(f.name).groupdict()
        key = DukascopyChunkKey(
            instrument=gd["instrument"],
            timeframe=gd["tf"].lower(),
            start=gd["start"],
            end=gd["end"],
        )
        chunk_groups.setdefault(key, {})[gd["side"]] = f

    per_series: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for key, sides in tqdm(
        chunk_groups.items(),
        desc="Merging bid/ask chunks",
        unit="chunk",
    ):
        bid_df = _safe_read_csv(sides.get("bid")) if "bid" in sides else None
        ask_df = _safe_read_csv(sides.get("ask")) if "ask" in sides else None

        if (bid_df is not None and bid_df.empty) or (ask_df is not None and ask_df.empty):
            continue

        merged = _merge_bid_ask(bid_df, ask_df)
        if merged.empty:
            continue

        per_series.setdefault((key.instrument, key.timeframe), []).append(merged)

    for (instrument, tf), chunks in tqdm(
        per_series.items(),
        desc="Writing merged series",
        unit="series",
    ):
        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        raw_symbol = mapping.get(instrument, instrument)
        safe_symbol = sanitize_filename(raw_symbol)

        out_dir = output_folder / tf
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{safe_symbol}.csv"
        df.to_csv(out_path, index=False)

    print(f"[OK] Merged {len(per_series)} instrument/timeframe series into {output_folder}")


if __name__ == "__main__":
    merge_dukascopy_folder()
