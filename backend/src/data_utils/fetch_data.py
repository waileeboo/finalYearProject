import yfinance as yf
from datetime import datetime
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_data(ticker: str, start_date: str, end_date: str | None, interval: str = "1d") -> None:
    if end_date is None: 
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Downloading {ticker}.......")
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    
    if df.empty:
        print(f"No data for {ticker}.")
        return
    # remove multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    save_path = RAW_DATA_DIR / f"{ticker.replace('^', '')}_{interval}.csv"
    df.to_csv(save_path)
    print(f"Saved to {save_path}\n")

def main():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN"]
    start_date = "2000-01-01"
    end_date = "2025-01-01"
    interval = "1d"

    for ticker in tickers:
        download_data(ticker, start_date, end_date, interval)
  
                
if __name__ == "__main__":
    main()
    
    