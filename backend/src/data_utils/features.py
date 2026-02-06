import pandas as pd 
import numpy as np

def add_more_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add more features to the dataframe, such as moving averages and volatility.
    """
    
    df2 = df.copy()
    
    # Rolling volatility (10days)
    df2["volatility_10"] = df2["Close_return"].rolling(10).std()
    
    # Momentum (5days) how much price has changed in the last 5 days
    df2["momentum_5"] = np.log(df2["Close"] / df2["Close"].shift(5))
    
    # Volume surprice (z-score of volume compared to 20-day rolling mean) is today volume unusually high or low compared to the last 20 days?
    df2["vol_mean_20"] = df2["Volume"].rolling(20).mean()
    df2["vol_std_20"] = df2["Volume"].rolling(20).std()
    df2["volume_z"] = (df2["Volume"] - df2["vol_mean_20"]) / (df2["vol_std_20"] + 1e-8)
    
    # drop rows with NaN values created by rolling    
    print("Data after adding more features:")
    print(df.head())
    print(f"Total features and samples: {df.shape[1]} features | " f"{df.shape[0]} samples \n")
      
    return df2
