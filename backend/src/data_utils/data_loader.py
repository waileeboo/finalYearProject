import pandas as pd
from typing import List
from pathlib import Path
from src.utils.paths import RAW_DATA_DIR
from src.utils.config import DEFAULT_TICKERS


def load_raw_data(
    raw_data_dir: Path = RAW_DATA_DIR, tickers: List[str] = DEFAULT_TICKERS
) -> dict[str, pd.DataFrame]:
    """
    load raw data from csv files for given tickers and set Date as index
    """
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        file_path = raw_data_dir / f"{ticker}_1d.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing data file: {file_path}")

        df = pd.read_csv(file_path, parse_dates=["Date"])

        df = df.set_index("Date")
        df = df.sort_index()
        data[ticker] = df

    return data

if __name__ == "__main__":
    load_raw_data()