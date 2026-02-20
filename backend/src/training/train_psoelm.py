import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data_utils.data_loader import  load_synthetic_series
from src.data_utils.preprocess import load_and_preprocess_data, split_time_series
from src.data_utils.windowing import create_windows 
from src.utils.evaluation import evaluate_returns, evaluate_prices
from src.utils.results_logger import log_results
from src.models.PSO_ELM import PSO_ELM

WINDOW_SIZE = 10
HIDDEN_NEURONS = 10
NUM_PARTICLES = 30
MAX_ITERATIONS = 1000
STOPPING_PATIENCE = 50


# Flatten the window except for for the first dimension (number of samples). This is because the PSO-ELM model expects a 2D array of shape (num_samples, window_size * num_features) as input.
def flatten_windows(X_windows: np.ndarray) -> np.ndarray:
    return X_windows.reshape(X_windows.shape[0], -1)

# Train on Real Data 
def train_pso_elm_real(ticker: str = "GSPC", seed: int | None = 42) -> None:
    
    # Step 1: Load and preprocess data (add return features, split data, scale features and targets)
    print("\n###################################################################")
    print(f"Step 1: Training PSO-ELM on real stock data ({ticker})...\n")
    data_dict = load_and_preprocess_data(ticker=ticker)
    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]
    y_test = data_dict["y_test"]
    raw_prices = data_dict["raw_prices"]
    target_scaler = data_dict["target_scaler"]
    test_df = data_dict["test_df"]


    # Step 2: Create and flatten sliding windows for PSO ELM
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
    
    # Step 3: Train PSO-ELM Model
    print(f"\nStep 3: Training PSO-ELM ({NUM_PARTICLES} particles, {HIDDEN_NEURONS} hidden neurons)...")

    pso_elm = PSO_ELM(
        hidden_neurons=HIDDEN_NEURONS,
        window_size=WINDOW_SIZE,
        num_features=1,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        stopping_patience=STOPPING_PATIENCE,
        seed=seed,
    )

    pso_elm.train(X_train_flat, y_train_win, X_val_flat, y_val_win)
    print("PSO-ELM training complete.\n")
    
    # Step 4: Evaluation PSO-ELM Predictions
    print("Step 4: Evaluating on test data...")

    # Predict scaled returns and inverse transform
    preds_scaled = pso_elm.predict(X_test_flat).reshape(-1, 1)
    preds_returns = target_scaler.inverse_transform(preds_scaled).flatten()

    # Inverse transform actual scaled returns
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

    # Log results
    log_results("PSO_ELM", ticker, {**return_metrics, **price_metrics})
    
    # Step 5: Plot
    # print("\nStep 5: Plotting results...")

    
        

    print("###################################################################\n")

    pass 
    
    
# Train on Syntehtic Data 
def train_pso_elm_synthetic(series_name: str = "linear_gradual_drift", series_number: int = 1, seed: int | None = 42)-> None:
    # Step 1: Load and preprocess data (add return features, split data, scale features and targets)
    print("\n###################################################################")
    print(f"Training PSO-ELM on synthetic data ({series_name}, {series_number})...\n")
    series = load_synthetic_series(series_name, series_number)
    print(f"Series length: {len(series)}")

    series_sr = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_sr)
    train_series = train_series.values
    val_series = val_series.values
    test_series = test_series.values
    val_end = len(train_series) + len(val_series)
    print(f"Train: {len(train_series)} | Val: {len(val_series)} | Test: {len(test_series)}")


    # Step 2: Create and flatten sliding windows for PSO ELM
    print("\nStep 2: Creating and flattening sliding windows...")
    X_train_win, y_train_win = create_windows(train_series.reshape(-1, 1), train_series, WINDOW_SIZE)
    X_val_win, y_val_win = create_windows(val_series.reshape(-1, 1), val_series, WINDOW_SIZE)
    X_test_win, y_test_win = create_windows(test_series.reshape(-1, 1), test_series, WINDOW_SIZE)

    X_train_flat = flatten_windows(X_train_win)
    X_val_flat = flatten_windows(X_val_win)
    X_test_flat = flatten_windows(X_test_win)
    print(f"ELM input shape — Train: {X_train_flat.shape} | Val: {X_val_flat.shape} | Test: {X_test_flat.shape}")
    # Step 3: Train PSO-ELM Model
    print(f"\nStep 3: Training PSO-ELM ({NUM_PARTICLES} particles, {HIDDEN_NEURONS} hidden neurons)...")

    pso_elm = PSO_ELM(
        hidden_neurons=HIDDEN_NEURONS,
        window_size=WINDOW_SIZE,
        num_features=1,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        stopping_patience=STOPPING_PATIENCE,
        seed=seed,
    )

    pso_elm.train(X_train_flat, y_train_win, X_val_flat, y_val_win)
    print("PSO-ELM training complete.\n")

    
    # Step 4: Evaluation PSO-ELM Predictions
    
    print("Step 4: Evaluating on test data...")

    preds_actual = pso_elm.predict(X_test_flat)
    actual_values = test_series[WINDOW_SIZE:]

    print(f"Predicted Range: {preds_actual.min():.4f} to {preds_actual.max():.4f}")
    print(f"Actual Range:    {actual_values.min():.4f} to {actual_values.max():.4f}")

    metrics = evaluate_returns(actual_values, preds_actual)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Log results
    log_results("PSO_ELM", f"{series_name}_{series_number}", metrics)

    # Step 5: Plot
    # print("\nStep 5: Plotting results...")


    print("\nPSO-ELM — Synthetic Data Complete.")    
        
    print("###################################################################\n")
    pass

if __name__ == "__main__": 
    for i in tqdm(range(1, 31), desc="Training PSO-ELM on Real Data", ncols=100):
        train_pso_elm_real(ticker="GSPC", seed=i)
         
    synthetic_series = [
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    for name in synthetic_series:
        for i in tqdm(range(1, 31), desc=f"Training PSO-ELM on Synthetic: {name}", ncols=100):
            train_pso_elm_synthetic(name, series_number=i, seed=i)
