import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from src.data_utils.data_loader import load_raw_data
from src.data_utils.preprocess import add_return_features
from src.utils.config import DEFAULT_TICKERS, FEATURE_COLS


def plot_close_prices_stock(data: dict[str, pd.DataFrame], tickers: List[str]= DEFAULT_TICKERS) -> None:
    """
    Plot closing prices over time for the specified tickers.
    """
    
    for ticker in tickers:
        df = data[ticker]

        plt.figure(figsize=(14, 6))
        plt.plot(df["Close"], label="Close Price", linewidth=1.2)
        plt.title(f"{ticker} Close Price Over Time", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_return_prices_stock(data: dict[str, pd.DataFrame], tickers: List[str] = DEFAULT_TICKERS, features: List[str] = FEATURE_COLS) -> None:
    """
    Plot return prices over time for the specified tickers.
    """
    
    for ticker in tickers:
        df = data[ticker]
        df = add_return_features(df, features)
        plt.figure(figsize=(14, 6))
        plt.plot(df["Close_return"], label="Close Return Price")
        plt.title(f"{ticker} Close Return Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Return Price")
        plt.grid(True)
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    data = load_raw_data()
    plot_return_prices_stock(data, tickers=DEFAULT_TICKERS, features=FEATURE_COLS)
    # data2 = load_raw_data()
    # plot_close_prices(data2, tickers=DEFAULT_TICKERS)
    


    