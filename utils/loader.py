from pathlib import Path
import pandas as pd

try:
    from config.params import UNIVERSES as PARAM_UNIVERSES
except Exception:
    PARAM_UNIVERSES = None


def load_price_csv(asset: str, data_path: Path) -> pd.DataFrame:
    file = data_path / f"{asset}.csv"

    if not file.exists():
        raise FileNotFoundError(
            f"[DATA ERROR] Missing price file: {file}\n"
            f"→ asset='{asset}'\n"
            f"→ data_path='{data_path}'"
        )

    df = pd.read_csv(file)

    if df.empty:
        raise ValueError(f"[DATA ERROR] Empty CSV file: {file}")

    return df

# Tous les tickers disponibles
def list_assets(base_path: Path) -> list[str]:
    p = base_path / "d1"
    if not p.exists():
        return []
    return sorted(f.stem.upper() for f in p.glob("*.csv"))