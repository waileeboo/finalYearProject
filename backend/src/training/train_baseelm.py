import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from src.data_utils.data_loader import  load_synthetic_series
from src.data_utils.preprocess import load_and_preprocess_data, split_time_series
from src.data_utils.windowing import create_windows 
from src.models.baselines.elm_base import ELMBase
from src.utils.config import FEATURE_COLS, RETURN_FEATURES
from src.utils.evaluation import evaluate_returns, evaluate_prices
from src.utils.results_logger import log_results

WINDOW_SIZE = 10
HIDDEN_NEURONS = 10

def flatten_windows(X_windows: np.ndarray) -> np.ndarray:
    return X_windows.reshape(X_windows.shape[0], -1)

def train_elm_real():
    print("#############################################################")
    print("Training ELM on real stock data...\n")

    # -------------------------------------------------------------
    # Step 1: Load and preprocess data (add return features, 
    #         split data, scale features and targets)
    # -------------------------------------------------------------
    print("Step 1: Load and Preprocessing Data...\n")
    data_dict = load_and_preprocess_data(ticker="GSPC")
    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]
    y_test = data_dict["y_test"]
    raw_prices = data_dict["raw_prices"]
    target_scaler = data_dict["target_scaler"]
    test_df = data_dict["test_df"]
    
    # -------------------------------------------------------------
    # Step 2: Create and flatten sliding windows for ELM
    # -------------------------------------------------------------
    print("Step 2: Create and Flatten Sliding Windows for ELM...\n")
    X_train_win, y_train_win = create_windows(X_train, y_train, WINDOW_SIZE)
    X_val_win, y_val_win = create_windows(X_val, y_val, WINDOW_SIZE)
    X_test_win, y_test_win = create_windows(X_test, y_test, WINDOW_SIZE)
    print(X_train_win)
    print (f"X_train_win shape: {X_train_win.shape} | y_train_win shape: {y_train_win.shape}")
    print (f"X_val_win shape: {X_val_win.shape} | y_val_win shape: {y_val_win.shape}")
    print (f"X_test_win shape: {X_test_win.shape} | y_test_win shape: {y_test_win.shape}\n")
    
    # Flatten windows for ELM input
    X_train_flat = flatten_windows(X_train_win)
    X_val_flat = flatten_windows(X_val_win)
    X_test_flat = flatten_windows(X_test_win)
    print(f"ELM input shape — Train: {X_train_flat.shape} | Val: {X_val_flat.shape} | Test: {X_test_flat.shape}\n")
    
    
    # -------------------------------------------------------------
    # Step 3: Train ELM Model 
    # -------------------------------------------------------------
    print(f"Step 3: Train ELM Model with {HIDDEN_NEURONS} hidden neurons...")
    elm = ELMBase(hidden_neurons=HIDDEN_NEURONS, seed=42)
    elm.train(X_train_flat, y_train_win)
    
    print("ELM training complete.\n")
    
    
    # -------------------------------------------------------------
    # Step 4: Evaluation ELM Predictions 
    # -------------------------------------------------------------
    print("Step 4: Evaluate ELM on test data...\n")
    
    # Predict scaled returns and inverse transform reshape becasue predict returns a 1D array but we need to inverse transform it back to 2D array for the scaler. Then flatten it back to 1D array for evaluation and price reconstruction.
    preds_scaled = elm.predict(X_test_flat).reshape(-1, 1)
    preds_returns = target_scaler.inverse_transform(preds_scaled).flatten()

    # Also inverse transform actual scaled returns for return-level metrics
    actual_returns = target_scaler.inverse_transform(y_test_win.reshape(-1, 1)).flatten()

    # Reconstruct prices: Price_t = Price_{t-1} * exp(Return_t)
    test_dates = test_df.index[WINDOW_SIZE:]
    prev_dates = test_df.index[WINDOW_SIZE - 1: -1]

    prev_prices = raw_prices.loc[prev_dates].values
    actual_prices = raw_prices.loc[test_dates].values
    pred_prices = prev_prices * np.exp(preds_returns)
    
    print(f"First date: {test_dates[0]}")
    print(f"Actual price: {actual_prices[0]:.2f}")
    print(f"Predicted price: {pred_prices[0]:.2f}")
    print(f"Raw price at that date: {data_dict['raw_prices'].loc[test_dates[0]]:.2f}")

    print(f"Predicted Price Range: {pred_prices.min():.2f} to {pred_prices.max():.2f}")
    print(f"Actual Price Range:    {actual_prices.min():.2f} to {actual_prices.max():.2f}")

    # Return-level metrics
    return_metrics = evaluate_returns(actual_returns, preds_returns)
    print("\nReturn-Level Metrics:")
    for key, value in return_metrics.items():
        print(f"  {key}: {value:.4f}")
    # Price-level metrics
    price_metrics = evaluate_prices(actual_prices, pred_prices)
    print("\nPrice-Level Metrics:")
    for key, value in price_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    log_results("ELM_Baseline", "GSPC", {**return_metrics, **price_metrics})

    # -------------------------------------------------------------
    # Step 5: Plot 
    # -------------------------------------------------------------
    # print("\nStep 5: Plotting results...")
    # plt.figure(figsize=(12, 6))
    # plt.plot(test_dates, actual_prices, label="Actual Price", color="blue")
    # plt.plot(test_dates, pred_prices, label="ELM Predicted Price", color="red", alpha=0.7)
    # plt.title("ELM Baseline: Actual vs Predicted Stock Prices (AAPL)")
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    print("ELM Baseline — Real Data Complete.")
    print("###################################################################\n")


        

def train_elm_synthetic(series_name: str = "linear_gradual_drift", series_number: int = 1): 
    print("#############################################################\n")
    print(f"ELM Baseline — Synthetic Data ({series_name} #{series_number})")
    # -------------------------------------------------------------
    # Step 1: Load and preprocess data (add return features, 
    #         split data, scale features and targets)
    # -------------------------------------------------------------
    print("Step 1: Load and Preprocessing Data..")
    series = load_synthetic_series(series_name, series_number)
    print(f"Series length: {len(series)}")
    
    series_sr = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_sr)
    # Convert back to numpy
    train_series = train_series.values
    val_series = val_series.values
    test_series = test_series.values
    val_end = len(train_series) + len(val_series)

    print(f"Train length: {len(train_series)} | Val length: {len(val_series)} | Test length: {len(test_series)}\n")
    
    
    # -------------------------------------------------------------
    # Step 2: Create and flatten sliding windows for ELM
    # -------------------------------------------------------------
    print("\nStep 2: Creating sliding windows...")
    # reshape the series to 2D array for windowing function, then flatten it back to 1D array for the target variable. This is because the create_windows function expects a 2D array for features and a 1D array for targets.
    X_train_win, y_train_win = create_windows(
        train_series.reshape(-1, 1), train_series, WINDOW_SIZE
    )
    X_val_win, y_val_win = create_windows(
        val_series.reshape(-1, 1), val_series, WINDOW_SIZE
    )
    X_test_win, y_test_win = create_windows(
        test_series.reshape(-1, 1), test_series, WINDOW_SIZE
    )

    X_train_flat = flatten_windows(X_train_win)
    X_val_flat = flatten_windows(X_val_win)
    X_test_flat = flatten_windows(X_test_win)
    
    print(f"ELM input shape — Train: {X_train_flat.shape} | Val: {X_val_flat.shape} | Test: {X_test_flat.shape}")
    
    # -------------------------------------------------------------
    # Step 3: Train ELM Model 
    # -------------------------------------------------------------
    
    print(f"\nStep 3: Training ELM with {HIDDEN_NEURONS} hidden neurons...")
    elm = ELMBase(hidden_neurons=HIDDEN_NEURONS, seed=42)
    elm.train(X_train_flat, y_train_win)
    print("ELM training complete.")

    # -------------------------------------------------------------
    # Step 4: Evaauation ELM Predictions 
    # -------------------------------------------------------------
    print("\nStep 4: Evaluating on test data...")

    preds_actual = elm.predict(X_test_flat)
    actual_values = test_series[WINDOW_SIZE:]

    print(f"Predicted Range: {preds_actual.min():.4f} to {preds_actual.max():.4f}")
    print(f"Actual Range:    {actual_values.min():.4f} to {actual_values.max():.4f}")

    metrics = evaluate_returns(actual_values, preds_actual)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
        
    log_results(model_name="ELM_Baseline", dataset=f"{series_name}_{series_number}", metrics=metrics)
    
    
    # -------------------------------------------------------------
    # Step 5: Plot 
    # -------------------------------------------------------------
    print("\nStep 5: Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label="Actual", color="blue")
    plt.plot(preds_actual, label="ELM Predicted", color="red", alpha=0.7)

    # Mark known drift points in test region
    concept_size = 2000
    total_concepts = 10
    drift_points = [concept_size * i for i in range(1, total_concepts)]
    for dp in drift_points:
        dp_relative = dp - val_end - WINDOW_SIZE
        if 0 <= dp_relative < len(actual_values):
            plt.axvline(dp_relative, color="green", linestyle="--", alpha=0.5,
                        label="Drift Point" if dp == drift_points[0] else "")

    plt.title(f"ELM Baseline: {series_name} #{series_number}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    
    print("ELM Baseline — Synthetic Data Complete.")
    print("###################################################################\n")



if __name__ == "__main__": 
    train_elm_real()
    synthetic_series = [
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    for name in synthetic_series:
        train_elm_synthetic(name, series_number=1)
    
    