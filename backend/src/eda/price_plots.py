import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from src.data_utils.data_loader import load_raw_data



def plot_close_prices(
    data: dict[str, pd.DataFrame],
    tickers: List[str],
) -> None:
    """
    Plot closing prices over time for the specified tickers.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to DataFrames.
    tickers : Iterable[str]
        Ticker symbols to plot.
    """
    for ticker in tickers:
        df = data[ticker]

        plt.figure(figsize=(14, 6))
        plt.plot(df["Close"], label="Close Price", linewidth=1.2)
        plt.title(f"{ticker} Close Price Over Time", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    data = load_raw_data()
    plot_close_prices(data, ["AAPL", "MSFT", "GOOGL", "AMZN"])