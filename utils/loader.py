from pathlib import Path
import pandas as pd


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
