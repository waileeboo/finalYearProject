from logging import warning
import pandas as pd
from src.data_utils.preprocess import add_return_features, split_time_series
from src.data_utils.data_loader import load_raw_data
from src.utils.config import FEATURE_COLS
from src.utils.evaluation import model_evaluate_metrics
from src.models.baselines.arima_models import arima_forecast, check_stationary, fit_auto_arima
import matplotlib.pyplot as plt
import warnings

def train_base_arima():
    print("#######################################################################")
    print("Starting ARIMA training script...\n")
    data = load_raw_data()
    df = data["AAPL"]
    raw_prices = df["Close"].copy()
    
    print("Step 1: Preprocessing Data (add return features and only keep close returns)...\n")
    df = add_return_features(df, FEATURE_COLS)
    df = df["Close_return"]
    
    print("Step 2: Splitting Data into Train, Validation and Test sets (Combining train and validation sets)...\n")
    train_df, val_df , test_df = split_time_series(df)
    train_df = pd.concat([train_df, val_df])
    print(f"Total rows in dataset: {len(train_df)+len(test_df)}")
    print(f"Training rows and shape: {len(train_df)}, Shape: {train_df.shape} | Test rows and shape: {len(test_df)}, Shape: {test_df.shape}\n")
    
    print("Step 3: Check if dataset is Stationary...\n")
    check_stationary(train_df)
    
    print("Step 4: Fit Auto ARIMA model on training data...\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_auto_arima(train_df)
    
    print("ARIMA model training complete.\n")
    
    print("Step 5: Forecasting using ARIMA model...\n")
    # Getting the first raw price of the test set 
    last_train_price = raw_prices.iloc[len(train_df)]
    # number of steps to forecast 
    steps = len(test_df)
    print(f"Steps:{steps} | last_train_price: {last_train_price}\n") 
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecasted_prices = arima_forecast(model, steps, last_train_price)
    print(f"Forecasted Prices for {steps} steps:\n{forecasted_prices}\n")
    # Convert forcasted prices to a pandas series using the test_df index
    forecasted_prices = pd.Series(forecasted_prices.values, index=test_df.index, name="ARIMA_Forecasted_Price")
    print(f"Forecasted Prices with Dates:\n{forecasted_prices.head()}\n")
    
    actual_test_prices = raw_prices.loc[test_df.index]
    print("Comparison Check...")
    print(f"First Predicted Date & Price: {forecasted_prices.index[0]} | {forecasted_prices.iloc[0]}")
    print(f"First Actual Date & Price:    {actual_test_prices.index[0]} | {actual_test_prices.iloc[0]}\n")
    
    print("Step 6: Evaluating ARIMA model performance...\n")
    evaluation_metrics = model_evaluate_metrics(actual_test_prices, forecasted_prices)
    print("Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("Step 7: Plotting the results...\n")
    plt.figure(figsize=(12,6))
    
    # plot the last 100 training prices for context 
    plt.plot(raw_prices.loc[train_df.index].tail(100), label="Training Prices", color="gray", alpha=0.5)
    
    # plot the Actual Prices during test period and Forecasted Prices
    plt.plot(actual_test_prices, label="Actual Prices", color="blue", linewidth=2)
    plt.plot(forecasted_prices, label="ARIMA Forecasted Prices", linestyle="--", color="red", linewidth=2)
    
    
    plt.title("ARIMA Model Forecast vs Actual Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("ARIMA Baseline DONE.")
    print("####################################################################\n")
        
    
if __name__ == "__main__":
    train_base_arima()