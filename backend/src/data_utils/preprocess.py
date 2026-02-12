import pandas as pd 
import numpy as np
 
from sklearn.preprocessing import MinMaxScaler
from src.utils.config import DEFAULT_TICKERS

from src.data_utils.data_loader import load_raw_data
from src.utils.config import FEATURE_COLS, RETURN_FEATURES



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
    return_cols = [f"{col}_return" for col in feature_cols]
    df = df[return_cols]
    print ("Data after adding return features:")
    print(df.head())
    print(f"Total features and samples: {df.shape[1]} features | " f"{df.shape[0]} samples \n")  
    return df
    
    
def split_time_series(
    df: pd.DataFrame | pd.Series,
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
    feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale features using Standard scaling (fit on train only).
    """
    # use standard scaler instead of min max scaler 
    # print("Scaling Features to mean = 0 and std=1...\n")
    # scaler = StandardScaler()
    print("Scaling features to range [0, 1]...\n")
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    return X_train, X_val, X_test, scaler

def scale_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale target variable using Min-Max scaling (fit on train only).
    """
    print(f"Scaling target variable '{target_col}' to range [0, 1]...\n")
    scaler = MinMaxScaler(feature_range=(0, 1))

    y_train = scaler.fit_transform(train_df[[target_col]]).flatten()
    y_val = scaler.transform(val_df[[target_col]]).flatten()
    y_test = scaler.transform(test_df[[target_col]]).flatten()

    return y_train, y_val, y_test, scaler



def load_and_preprocess_data():
    """
    Load and preprocess data for training
    """
    data = load_raw_data()
    df = data["AAPL"]
    raw_prices = df["Close"].copy()
    
    # Feature engineering pipeline: add return features and scale them
    df = add_return_features(df, FEATURE_COLS)
    df = df.dropna()
    print ("Data after adding return features:")
    print(df.head())
    print(f"Total features and samples: {df.shape[1]} features | " f"{df.shape[0]} samples \n")
    
    # Split Data into train, validation, and test sets      
    train_df, val_df, test_df = split_time_series(df)
    
    #Scale features and target variable
    X_train, X_val, X_test, feature_scaler = scale_features(train_df, val_df, test_df, RETURN_FEATURES)
    y_train, y_val, y_test, target_scaler = scale_targets(train_df, val_df, test_df, target_col="Close_return")

    return {"X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "raw_prices": raw_prices,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df
            }
    




    
    
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


# def add_more_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add more features to the dataframe, such as moving averages and volatility.
#     """
    
#     df2 = df.copy()
    
#     # Rolling volatility (10days)
#     df2["volatility_10"] = df2["Close_return"].rolling(10).std()
    
#     # Momentum (5days) how much price has changed in the last 5 days
#     df2["momentum_5"] = np.log(df2["Close"] / df2["Close"].shift(5))
    
#     # Volume surprice (z-score of volume compared to 20-day rolling mean) is today volume unusually high or low compared to the last 20 days?
#     df2["vol_mean_20"] = df2["Volume"].rolling(20).mean()
#     df2["vol_std_20"] = df2["Volume"].rolling(20).std()
#     df2["volume_z"] = (df2["Volume"] - df2["vol_mean_20"]) / (df2["vol_std_20"] + 1e-8)
#     df2 = df2.drop(columns=["vol_mean_20", "vol_std_20"])
#     # drop rows with NaN values created by rolling    
#     print("Data after adding more features:")
#     print(df2.head())
#     print(f"Total features and samples: {df2.shape[1]} features | " f"{df2.shape[0]} samples \n")
      
   # return df2