from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import copy
from src.data_utils.preprocess import add_return_features
from src.data_utils.data_loader import load_raw_data
from src.utils.config import FEATURE_COLS
from src.models.baselines.arima_models import check_stationary, auto_arima


def main():
    data = load_raw_data()
    df = data["AAPL"]
    raw_prices = df["Close"].copy()
    df = add_return_features(df, FEATURE_COLS)
    df = df["Close_return"]
    print(df.head())
    
    check_stationary(df)
    
        
    
if __name__ == "__main__":
    main()