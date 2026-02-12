import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

from src.data_utils.preprocess import add_return_features, split_time_series
from src.data_utils.data_loader import load_raw_data
from src.utils.config import FEATURE_COLS
from src.utils.evaluation import evaluate_prices
from src.models.baselines.arima_base import arima_forecast, check_stationary, fit_auto_arima


def load_and_preprocess_data(ticker:str = "AAPL")-> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load and preprocess data for ARIMA training
    """
    
    data = load_raw_data()
    df = data[ticker]
    raw_prices = df["Close"].copy()
    
    df = add_return_features(df, FEATURE_COLS)
    df = df["Close_return"]
    
    # split data 
    train_df, val_df, test_df = split_time_series(df)
    train_df = pd.concat([train_df, val_df])
    print(f"Total rows in dataset: {len(train_df)+len(test_df)}")
    print(f"Training rows and shape: {len(train_df)}, Shape: {train_df.shape} | Test rows and shape: {len(test_df)}, Shape: {test_df.shape}\n")
    
    return raw_prices, train_df, test_df


def train_arima_model(train_data: pd.Series)-> ARIMA:
    """
    Check Stationary and fit auto arima model
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_stationary(train_data)
        model = fit_auto_arima(train_data)
        print("ARIMA model training complete.\n")
    return model


def generate_arima_forecast(model: ARIMA, raw_prices: pd.Series, train_df: pd.Series, test_df: pd.Series)-> tuple[pd.Series, pd.Series]:
    """
    Generate ARIMA forecasted prices
    """
    last_train_price = raw_prices.iloc[len(train_df)]
    steps = len(test_df)
    print(f"Steps:{steps} | last_train_price: {last_train_price}\n")
    
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecasted_prices = arima_forecast(model, steps, last_train_price)
    
    forecasted_prices = pd.Series(forecasted_prices.values, index=test_df.index, name="ARIMA_Forecasted_Price")
     
    actual_prices = raw_prices.loc[test_df.index]
    
    print("Comparison Check...")
    print(f"First Predicted Date & Price: {forecasted_prices.index[0]} | {forecasted_prices.iloc[0]}")
    print(f"First Actual Date & Price:    {actual_prices.index[0]} | {actual_prices.iloc[0]}\n")    
    
    return forecasted_prices, actual_prices
       
        
def evaluate_model(actual_prices: pd.Series, forecasted_prices: pd.Series)-> dict:
    """
    Evaluate ARIMA model performance
    """
    evaluation_metrics = evaluate_prices(actual_prices, forecasted_prices)
    print("Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    return evaluation_metrics


def plot_results(raw_prices: pd.Series, train_df: pd.Series, actual_prices: pd.Series, forecasted_prices: pd.Series)-> None:
    """
    Plot ARIMA forecast vs actual prices
    """
    plt.figure(figsize=(12,6))
    
    # plot the last 100 training prices for context 
    plt.plot(raw_prices.loc[train_df.index].tail(100), label="Training Prices", color="gray", alpha=0.5)
    
    # plot the Actual Prices during test period and Forecasted Prices
    plt.plot(actual_prices, label="Actual Prices", color="blue", linewidth=2)
    plt.plot(forecasted_prices, label="ARIMA Forecasted Prices", linestyle="--", color="red", linewidth=2)
    
    
    plt.title("ARIMA Model Forecast vs Actual Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    


def main():
    print("#######################################################################")
    print("Starting ARIMA training script...\n")
    
    print("Step 1: Load Data and Preprocessing Data (add return features and only keep close returns)...\n")
    raw_prices, train_df, test_df = load_and_preprocess_data()
    
    print("Step 2: Check if dataset is Stationary and Fit auto ARIMA model...\n")
    model = train_arima_model(train_df)
    
    print("Step 3: Forecasting using ARIMA model...\n")
    # Getting the first raw price of the test set 
    forecasted_prices, actual_prices = generate_arima_forecast(model, raw_prices, train_df, test_df)
    
    print("Step 4: Evaluating ARIMA model performance...\n")
    evaluation_metrics = evaluate_model(actual_prices, forecasted_prices)
    print("Step 5: Plotting the results...\n")
    plot_results(raw_prices, train_df, actual_prices, forecasted_prices)
    
    print("ARIMA Baseline DONE.")
    print("####################################################################\n")
    return evaluation_metrics
        
    
if __name__ == "__main__":
    main()