import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_utils.data_loader import load_synthetic_series
from src.data_utils.preprocess import load_and_preprocess_data, split_time_series
from src.data_utils.windowing import create_windows
from src.models.baselines.svr_base import SVRBase
from src.utils.evaluation import evaluate_returns, evaluate_prices
from src.utils.results_logger import log_results
from src.utils.paths import RESULTS_FILE_RQ4_BASELINE as RESULTS_FILE

# Configuration
WINDOW_SIZE = 10
SVR_C = 1.0
SVR_EPSILON = 0.1


def flatten_windows(X_windows: np.ndarray) -> np.ndarray:
    return X_windows.reshape(X_windows.shape[0], -1)

def train_svr_real(ticker: str = "GSPC", seed: int | None = 42) -> None:
    print("##################################################")
    print(f"SVR Baesline = Real data - {ticker}")
    # step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(ticker=ticker)
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    raw_prices = data_dict["raw_prices"]
    target_scaler = data_dict["target_scaler"]
    test_df = data_dict["test_df"]
    
    # step 2: Create and flatten sliding windows 
    print("Step 2: Create and Flattening Slinding windows")
    X_train_win, y_train_win = create_windows(X_train, y_train, window_size=WINDOW_SIZE)
    X_test_win, y_test_win = create_windows(X_test, y_test, window_size=WINDOW_SIZE)
    
    X_train_flat = flatten_windows(X_train_win)
    X_test_flat = flatten_windows(X_test_win)
    print(f"SVR input shape - Train: {X_train_flat.shape}, Test: {X_test_flat.shape}\n")
    
    # step 3: train 
    print("Step 3: Training SVR...")
    svr = SVRBase(kernel='rbf', C=SVR_C, epsilon=SVR_EPSILON, seed=seed)
    svr.train(X_train_flat, y_train_win)
    print("Training completed.\n")
    
    # Step 4 Evaluation 
    print("Step 4: Evaluating model on Test Data...")
    preds_scaled = svr.predict(X_test_flat).reshape(-1,1)
    preds_returns = target_scaler.inverse_transform(preds_scaled).flatten()
    actual_returns = target_scaler.inverse_transform(y_test_win.reshape(-1,1)).flatten()
    
    test_dates = test_df.index[WINDOW_SIZE:]
    prev_dates = test_df.index[WINDOW_SIZE-1:-1]
    prev_prices = raw_prices.loc[prev_dates].values
    actual_prices = raw_prices.loc[test_dates].values
    pred_prices = prev_prices * np.exp(preds_returns)
    
    print(f"First data : {test_dates[0]}")
    print(f"Actual price : {actual_prices[0]}, Predicted price : {pred_prices[0]}")
    print(f"Predicted Range : {pred_prices.min()} - {pred_prices.max()}")
    print(f"Actual Range : {actual_prices.min()} - {actual_prices.max()}")
    
    return_metrics = evaluate_returns(actual_returns, preds_returns)
    print("Return_level Metrics:")
    for metric, value in return_metrics.items():
        print(f"{metric}: {value}")

    price_metrics = evaluate_prices(actual_prices, pred_prices)
    print("\nPrice_level Metrics:")
    for metric, value in price_metrics.items():
        print(f"{metric}: {value}")
        
    log_results("SVR_Baseline", ticker, {**return_metrics, **price_metrics}, path=RESULTS_FILE)    
    
    
    
def train_svr_synthetic(series_name: str = "linear_gradual_drift", series_number: int = 1, seed: int | None = 42) -> None:
    print("\n##################################################")
    print(f"SVR Baesline = Synthetic data - {series_name} - Series {series_number}")
    
    print("Step 1: Loading synthetic data...")
    series = load_synthetic_series(series_name=series_name, series_number=series_number)
    series_sr = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_sr, train_ratio=0.5, val_ratio=0.1)
    train_series = train_series.values
    val_series = val_series.values
    test_series = test_series.values
    print(f"Train: {len(train_series)}, Val: {len(val_series)}, Test: {len(test_series)}\n")
    
    print("Step 2: Create and flatten sliding windows...")
    X_train_win, y_train_win = create_windows(train_series.reshape(-1,1), train_series, window_size=WINDOW_SIZE)
    X_test_win, y_test_win = create_windows(test_series.reshape(-1,1), test_series, window_size=WINDOW_SIZE)
    X_train_flat = flatten_windows(X_train_win)
    X_test_flat = flatten_windows(X_test_win)
    print(f"RF input shape - Train: {X_train_flat.shape}, Test: {X_test_flat.shape}\n")
    
    # Step 3: Train
    svr = SVRBase(kernel="rbf", C=SVR_C, epsilon=SVR_EPSILON)
    svr.train(X_train_flat, y_train_win)
    print("SVR training complete.\n")
    
    print("Step 4: Evaluating model on Test Data...")
    preds = svr.predict(X_test_flat)
    actual_values = test_series[WINDOW_SIZE:]
    print(f"Predicted Range : {preds.min()} - {preds.max()}")
    print(f"Actual Range : {actual_values.min()} - {actual_values.max()}")
    
    metrics = evaluate_returns(actual_values, preds)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
        
    log_results(model_name="SVR_Baseline", dataset=f"{series_name}_{series_number}", metrics=metrics, path=RESULTS_FILE)

    print("SVR Baseline synthetic data done.")
    print("##################################################\n")        

def main():
    for i in tqdm(range(1,31), desc="Training SVR on Real Data", ncols=100):
        train_svr_real(ticker="GSPC", seed=i)
        
    synthetic_series = [
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    
    for name in synthetic_series:
        for i in tqdm(range(1,31), desc=f"Training SVR on Synthetic Data - {name}", ncols=100):
            train_svr_synthetic(series_name=name, series_number=i, seed=i)
            
if __name__ == "__main__":
    main()