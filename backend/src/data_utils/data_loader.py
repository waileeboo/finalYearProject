import pandas as pd
from typing import List
from pathlib import Path
import numpy as np

from src.utils.paths import RAW_DATA_DIR, SYNTHETIC_DATA_DIR
from src.utils.config import DEFAULT_TICKERS


def load_raw_data(
    raw_data_dir: Path = RAW_DATA_DIR, tickers: List[str] = DEFAULT_TICKERS
) -> dict[str, pd.DataFrame]:
    """
    load raw data from csv files into a dictionary of dataframes, one for each ticker. The csv files should be in the format "{ticker}_1d.csv" and located in the raw_data_dir.

    Args:
        raw_data_dir (Path, optional): directory containing raw data csv files. Defaults to RAW_DATA_DIR.
        tickers (List[str], optional): list of ticker symbols to load data for. Defaults to DEFAULT_TICKERS.
    Returns:
        dict[str, pd.DataFrame]: dictionary mapping ticker symbols to their corresponding dataframes
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

def load_synthetic_series(series_name: str, series_number: int) -> np.ndarray: 
    """
    load a generated synthetic series csv

    Args:
        series_name (str): the name of the series to load (e.g. "linear_abrupt_drift")
        series_number (int): the number of the series to load (since we generate multiple series for each concept type)

    Returns:
        np.ndarray: series data as numpy array
    """
    file_path = SYNTHETIC_DATA_DIR / series_name /f"{series_name}{series_number}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing synthetic data file: {file_path}")
    
    df = pd.read_csv(file_path, header=None)
    return df[0].values
    
    
    

if __name__ == "__main__":
    # load_raw_data()
    a = load_synthetic_series("linear_abrupt_drift", 1)
    print(a)