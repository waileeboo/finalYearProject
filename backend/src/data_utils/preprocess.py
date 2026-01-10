import pandas as pd 
import numpy as np
from typing import Tuple 
from sklearn.preprocessing import MinMaxScaler
from src.utils.config import DEFAULT_TICKERS

from src.data_utils.data_loader import load_raw_data

def check_missing_values(df: pd.DataFrame) -> None:
    """
    Check that DataFrame contains no missing values in any columns
    """
    if df.isna().values.any():
        missing_cols = df.isna().sum()
        missing_info = missing_cols[missing_cols > 0]
        raise ValueError(f"Data contains missing values:\n{missing_info}")
    

def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time series DataFrame into train / validation / test sets.

    Splitting is done chronologically (no shuffling).

    """
    check_missing_values(df)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df




def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
]:
    """
    Scale features using Min-Max scaling (fit on train only).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    return X_train, X_val, X_test, scaler

if __name__ == "__main__":
    tickers = DEFAULT_TICKERS
    data = load_raw_data()
    for ticker in tickers: 
        print(f"Checking missing values for {ticker}...")
        check_missing_values(data[ticker])
