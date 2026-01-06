from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import pandas as pd


# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

META_PATH = PROJECT_ROOT / "data_ingestion" / "dukascopy" / "instruments_all_from_readme.csv"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "d1"
OUT_PATH = PROJECT_ROOT / "data" / "asset_registry.csv"


# ============================================================
# HELPERS
# ============================================================
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _strip_accents(s: str) -> str:
    # NFKD separates accents; then we drop diacritics
    s_norm = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s_norm if not unicodedata.combining(ch))


def _clean_text(s: str) -> str:
    """
    Clean aggressive (hedge-fund-grade):
    - remove emojis / non-ascii
    - strip accents
    - keep only [A-Za-z0-9 _-]
    - collapse whitespace
    """
    if s is None:
        return ""

    s = str(s)

    # Remove non-printable / weird whitespace
    s = "".join(ch if ch.isprintable() else " " for ch in s)

    # Normalize accents
    s = _strip_accents(s)

    # Drop non-ascii (kills emojis & most exotic symbols)
    s = s.encode("ascii", errors="ignore").decode("ascii")

    # Replace separators with spaces
    s = s.replace("/", " ").replace("\\", " ").replace("|", " ").replace(":", " ")

    # Keep only alnum, space, underscore, hyphen
    s = re.sub(r"[^A-Za-z0-9 _-]+", " ", s)

    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _to_asset_key(symbol: str) -> str:
    """
    Convert symbol/name into your on-disk asset convention:
    - uppercase
    - spaces -> underscores
    - remove remaining junk
    """
    s = _clean_text(symbol).upper()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Daily data folder not found: {DATA_PATH}")

    meta = pd.read_csv(META_PATH)

    # --- Column detection (robust)
    symbol_col = _pick_col(meta, ["instrument_name", "symbol", "ticker", "name"])
    cat_name_col = _pick_col(meta, ["category_name", "category", "sector", "universe"])
    cat_id_col = _pick_col(meta, ["category_id", "categoryid"])

    print("[INFO] META columns:", list(meta.columns))
    print("[INFO] Picked columns:", {
        "symbol_col": symbol_col,
        "cat_name_col": cat_name_col,
        "cat_id_col": cat_id_col,
    })

    if symbol_col is None or cat_name_col is None:
        raise RuntimeError("Required columns not found in metadata")

    # --- Assets present on disk (truth source)
    available_assets = sorted(f.stem.upper() for f in DATA_PATH.glob("*.csv"))
    if not available_assets:
        raise RuntimeError(f"No CSV files found in {DATA_PATH}")

    # --- Build registry from metadata
    reg = meta.copy()

    # Clean fields
    reg["symbol"] = reg[symbol_col].astype(str).map(_clean_text)
    reg["category_name"] = reg[cat_name_col].astype(str)
    reg["category_name_clean"] = reg[cat_name_col].astype(str).map(_clean_text)

    # Canonical asset key (must match file stems)
    reg["asset"] = reg[symbol_col].astype(str).map(_to_asset_key)

    # Keep only assets truly available on disk
    reg = reg[reg["asset"].isin(available_assets)]

    # Keep columns
    keep_cols = ["asset", "symbol", "category_name", "category_name_clean"]
    if cat_id_col:
        reg["category_id"] = reg[cat_id_col]
        keep_cols.append("category_id")

    reg = reg[keep_cols].drop_duplicates()

    # Sort
    reg = reg.sort_values(["category_name_clean", "asset"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(OUT_PATH, index=False)

    print(f"[OK] Registry saved: {len(reg)} assets → {OUT_PATH}")
    print("[OK] Example categories (clean):", reg["category_name_clean"].dropna().unique()[:10])


if __name__ == "__main__":
    main()
