import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from tqdm import tqdm

from src.utils.results_logger import log_results
from src.data_utils.preprocess import add_return_features, split_time_series
from src.data_utils.data_loader import load_raw_data, load_synthetic_series
from src.utils.config import FEATURE_COLS
from src.utils.evaluation import evaluate_prices, evaluate_returns
from src.models.baselines.arima_base import arima_forecast, check_stationary, fit_auto_arima


def load_and_preprocess_data(ticker:str = "GSPC")-> tuple[pd.Series, pd.Series, pd.Series]:
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
        forecasted_prices, forecasted_log_returns = arima_forecast(model, steps, last_train_price)
    
    forecasted_prices = pd.Series(forecasted_prices.values, index=test_df.index, name="ARIMA_Forecasted_Price")
     
    actual_prices = raw_prices.loc[test_df.index]
    
    print("Comparison Check...")
    print(f"First Predicted Date & Price: {forecasted_prices.index[0]} | {forecasted_prices.iloc[0]}")
    print(f"First Actual Date & Price:    {actual_prices.index[0]} | {actual_prices.iloc[0]}\n")    
    
    return forecasted_prices, actual_prices, forecasted_log_returns
       
        
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
    


def train_arima_real():
    print("#######################################################################")
    print("Starting ARIMA training script on GSPC...\n")
    
    print("Step 1: Load Data and Preprocessing Data (add return features and only keep close returns)...\n")
    raw_prices, train_df, test_df = load_and_preprocess_data()
    
    print("Step 2: Check if dataset is Stationary and Fit auto ARIMA model...\n")
    model = train_arima_model(train_df)
    
    print("Step 3: Forecasting using ARIMA model...\n")
    # Getting the first raw price of the test set 
    forecasted_prices, actual_prices, forecasted_log_returns = generate_arima_forecast(model, raw_prices, train_df, test_df)
    
    print("Step 4: Evaluating ARIMA model performance...\n")
    evaluation_metrics = evaluate_model(actual_prices, forecasted_prices)
    return_metrics = evaluate_returns(test_df.values, forecasted_log_returns.values)
    evaluation_metrics.update(return_metrics)
    
    log_results("ARIMA_Baseline", "GSPC", evaluation_metrics)
    

    # print("Step 5: Plotting the results...\n")
    # plot_results(raw_prices, train_df, actual_prices, forecasted_prices)
    
    print("ARIMA Baseline - Real Data DONE.")
    print("####################################################################\n")
    

def train_arima_synthetic(series_name: str = "linear_gradual_drift", series_number: int = 1):
    print("#######################################################################")
    print(f"Starting ARIMA training script on synthetic data: {series_name} #{series_number}\n")
    
    # Step 1: Load and preprocess synthetic data
    print("Step 1: Load and Preprocessing Synthetic Data..")
    series = load_synthetic_series(series_name, series_number)
    print(f"Series length: {len(series)}")
    
    series_sr = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_sr)
    train_series = pd.concat([train_series, val_series])
    val_end = len(train_series)
    print(f"Train (inc. val): {len(train_series)} | Test: {len(test_series)}")
    
    # Step 2: Check Stationarity and Fit ARIMA model
    print("\nStep 2: Check Stationarity and Fit ARIMA model...")
    model = train_arima_model(train_series)
    
    # Step 3: Forecasting using ARIMA model
    print("\nStep 3: Forecasting using ARIMA model...")
    steps = len(test_series)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecasted_values = model.predict(n_periods=steps)
    
    forecasted_values = np.array(forecasted_values)  # strip pandas index so plot aligns

    actual_values = test_series.values
    print(f"Steps: {steps}")
    print(f"Predicted Range: {forecasted_values.min():.4f} to {forecasted_values.max():.4f}")
    print(f"Actual Range:    {actual_values.min():.4f} to {actual_values.max():.4f}")
    
    # Step 4: Evaluating ARIMA model performance
    print("\nStep 4: Evaluating ARIMA model performance...")
    metrics = evaluate_returns(actual_values, forecasted_values)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    log_results("ARIMA_Baseline", f"{series_name}_{series_number}", metrics)
    
    # Step 5: Plotting the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(actual_values, label="Actual", color="blue")
    # plt.plot(forecasted_values, label="ARIMA Forecast", color="red", linestyle="--", alpha=0.7)

    # plt.title(f"ARIMA Baseline: {series_name} #{series_number}")
    # plt.xlabel("Time Step")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    
    # print("ARIMA Baseline - Synthetic Data DONE.")
    print("####################################################################\n")


def main():
    # train_arima_real()
    synthetic_series = [
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    for name in synthetic_series:
        for i in tqdm(range(30), desc=f"Training ARIMA on {name} series"):
            train_arima_synthetic(name, series_number=i+1)
        
    
if __name__ == "__main__":
    main()