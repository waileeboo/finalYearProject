import yfinance as yf
import os 
from datetime import datetime
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def download_data(data: str, start_date: str, end_date: str | None, interval: str = "1d") -> None:
    if end_date is None: 
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Downloading {data}.......")
    
    df = yf.download(data, start=start_date, end=end_date, interval=interval)
    
    
    if df.empty:
        print(f"No data for {data}.")
        return
    # remove multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    save_path = os.path.join(RAW_DATA_DIR, f"{data.replace('^', '')}_{interval}.csv")
    df.to_csv(save_path)
    print(f"Saved to {save_path}")

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
    
    