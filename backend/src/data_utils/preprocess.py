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
    print("Checking for missing values...")
    if df.isna().values.any():
        missing_cols = df.isna().sum()
        missing_info = missing_cols[missing_cols > 0]
        raise ValueError(f"Data contains missing values:\n{missing_info}")
    else: 
        print("No missing values found.\n")
    

def add_return_features(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    add return features to the dataframe 
    """
    df = train_df.copy()
    for col in feature_cols:
        df[f"{col}_return"] = np.log(df[col] / df[col].shift(1))
    # drop first row as it will be nan after pct_change
    df.dropna(inplace=True)
    # drop original columns
    df = df.drop(columns=feature_cols)
    print ("Data after adding return features:")
    print(df.head())
    print(f"Total features and samples: {df.shape[1]} features | " f"{df.shape[0]} samples \n")  
    return df
    
    
def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.10,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale features using Min-Max scaling (fit on train only).
    """
    print("Scaling features to range [-1, 1]...\n")
    scaler = MinMaxScaler(feature_range=(-1, 1))

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
    df = data["AAPL"]
    print("Adding return features...")
    df = add_return_features(df, ["Open", "High", "Low", "Close", "Volume"])
    print("Splitting data...")
    train_df, val_df, test_df = split_time_series(df)
    print("Scaling features...")
