import pandas as pd
from pathlib import Path

def load_price_csv(symbol: str, DATA_PATH : Path) -> pd.DataFrame:

    file = DATA_PATH / f"{symbol}.csv"
    df = pd.read_csv(file)

    # handle timestamp column
    time_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
    df[time_col] = pd.to_datetime(df[time_col])

    # keep only useful data
    df = df[[time_col, "close"]].rename(columns={time_col: "datetime"})

    # sort by date ascending
    df = df.sort_values("datetime").reset_index(drop=True)

    return df
