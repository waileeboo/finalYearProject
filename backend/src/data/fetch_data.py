import yfinance as yf
import os 
from datetime import datetime

RAW_DATA_DIR = './backend/data/raw'
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def download_data(data: str, start_date: str, end_date: str | None, interval: str = "1d") -> None:
    if end_date is None: 
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Downloading {data}.......")
    
    df = yf.download(data, start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        print(f"[WARNING] No data for {data}.")
        return
    
    save_path = os.path.join(RAW_DATA_DIR, f"{data.replace('^', '')}_{interval}.csv")
    df.to_csv(save_path)
    print(f"[SUCCESS] Saved → {save_path}")

def main():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN"]
    start_date = "1990-01-01"
    end_date = "2025-01-01"
    interval = "1d"

    for ticker in tickers:
        download_data(ticker, start_date, end_date, interval)
  
                
if __name__ == "__main__":
    main()
    
    