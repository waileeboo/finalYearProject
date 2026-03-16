import time
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
 
from src.data_utils.preprocess import load_and_preprocess_data, split_time_series
from src.data_utils.windowing import create_windows
from src.data_utils.data_loader import load_synthetic_series
from src.utils.evaluation import evaluate_returns, evaluate_prices
from src.utils.results_logger import log_results
from src.models.baselines.rf_base import RFBase
from src.models.baselines.svr_base import SVRBase
from src.detectors.drift_detector import DriftDetector
from src.training.online_eval import (
    online_evaluation_loop_adaptive,
    online_evaluation_loop_no_detector,
    flatten_windows,
    get_true_drift_points_synthetic,
)
from src.utils.paths import RESULTS_FILE_RQ4_REAL, RESULTS_FILE_RQ4_SYNTHETIC


# Configuration
BEST_DETECTOR = "kswin"
WINDOW_SIZE = 10
N_ESTIMATORS = 100      # RF trees
SVR_C = 1.0             # SVR regularisation
SVR_EPSILON = 0.1       # SVR epsilon tube
RETRAIN_WINDOW = 200
RETRAIN_INTERVAL = 200  # periodic retrain interval for no-detector condition
NUM_SEEDS = 30
 
# Synthetic data settings
CONCEPT_LENGTH = 2000
TOTAL_CONCEPTS = 10
 
 
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    
# model initilaisers
def train_initial_rf(X_train_flat: np.ndarray, y_train_win: np.ndarray, seed: int) -> RFBase:
    """Train initial Random forest on full training data"""
    rf = RFBase(n_estimators=N_ESTIMATORS, seed=seed)
    rf.train(X_train_flat, y_train_win)
    return rf

def train_initial_svr(X_train_flat: np.ndarray, y_train_win: np.ndarray, seed: int) -> SVRBase:
    """Train initial SVR on full training data"""
    svr = SVRBase(kernel='rbf', C=SVR_C, epsilon=SVR_EPSILON, seed=seed)
    svr.train(X_train_flat, y_train_win)
    return svr 

# load real data
def load_real_data(ticker: str) -> dict:
    """Load and preprocess real stock data"""
    return load_and_preprocess_data(ticker=ticker)

def load_synthetic_data(series_name: str, series_number: int):
    """Load synthetic series and split into train/val/test"""
    series = load_synthetic_series(series_name=series_name, series_number=series_number)
    series_pd = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_pd, train_ratio=0.5, val_ratio=0.1)
    train_series = train_series.values
    val_series = val_series.values
    test_series = test_series.values
    test_start_idx = len(train_series) + len(val_series)
    return train_series, val_series, test_series, test_start_idx
    
    
def reconstruct_prices(
    preds_scaled: np.ndarray,
    actuals_scaled: np.ndarray,
    target_scaler,
    raw_prices: pd.Series,
    test_df_index: pd.DatetimeIndex,
) -> dict:
    """Inverse transform scaled prediction and reconstruct real prices"""
    preds_returns = target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    actual_returns = target_scaler.inverse_transform(actuals_scaled.reshape(-1,1)).flatten()
    dates = test_df_index[WINDOW_SIZE:]
    prev_dates = test_df_index[WINDOW_SIZE-1:-1]
    prev_prices = raw_prices.loc[prev_dates].values
    actual_prices = raw_prices.loc[dates].values
    pred_prices = prev_prices * np.exp(preds_returns)
    return {
        "pred_returns": preds_returns,
        "actual_returns": actual_returns,
        "dates": dates,
        "actual_prices": actual_prices,
        "pred_prices": pred_prices
    }
    
def _log_real_result(result:dict, model_name: str, ticker: str, data_dict: dict, elapsed: float) -> None:
    """Log results for real data experiments"""
    result["metrics"]["total_time"] = elapsed
    price_info = reconstruct_prices(
        result["predictions"], result["actuals"], data_dict["target_scaler"], data_dict["raw_prices"], data_dict["test_df"].index)
    
    unscaled = evaluate_returns(price_info["actual_returns"], price_info["pred_returns"])
    result["metrics"]["Return_MAE"] = unscaled["Return_MAE"]
    result["metrics"]["Return_MSE"] = unscaled["Return_MSE"]
    result["metrics"]["Return_RMSE"] = unscaled["Return_RMSE"]
    
    price_metrics = evaluate_prices(
        pd.Series(price_info["actual_prices"]),
        pd.Series(price_info["pred_prices"]),
    )
    result["metrics"].update({f"price_{k}": v for k, v in price_metrics.items()})
 
    log_results(
        model_name=model_name,
        dataset=ticker,
        metrics=result["metrics"],
        path=RESULTS_FILE_RQ4_REAL,
    )
    
def _build_real_flat_windows(data_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build flattened training window from real data dict"""
    X_train_win, y_train_win = create_windows(
        data_dict["X_train"], data_dict["y_train"], WINDOW_SIZE
    )
    return flatten_windows(X_train_win), y_train_win
    
def _build_synthetic_flat_windows(train_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build flattened training windows from synthetic series."""
    X_train_win, y_train_win = create_windows(
        train_series.reshape(-1, 1), train_series, WINDOW_SIZE
    )
    return flatten_windows(X_train_win), y_train_win
    
    
def run_rq4_real_with_detector(data_dict: dict, seed: int, ticker: str = "GSPC") -> None:
    """
    RQ4: on real data 
    RF and SVR retrain when KSWIN detects drift
    Results saved to RESULTS_FILE_RQ4_REAL
    """
    set_seed(seed)
    test_series = data_dict["y_test"]
    X_train_flat, y_train_win = _build_real_flat_windows(data_dict)
    models = {
        "RF": (train_initial_rf(X_train_flat, y_train_win, seed), "rf"),
        "SVR": (train_initial_svr(X_train_flat, y_train_win, seed), "svr")
    }
    
    for model_name, (model, model_type) in models.items():
        print(f"RQ4 Real data with detector. Running model: {model_name}")
        detector = DriftDetector(method=BEST_DETECTOR)
        start = time.time()
        result = online_evaluation_loop_adaptive(
            model=model,
            model_type=model_type,
            test_series=test_series,
            detector=detector,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            true_drift_points=None, 
            cooldown_period=200
        )
        
        _log_real_result(
            result,
            model_name=f"{model_name}_{BEST_DETECTOR}",
            ticker=ticker,
            data_dict=data_dict,
            elapsed=time.time() - start,
        )
    
        print(f"MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Drifts: {result['metrics']['num_drifts_detected']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {result['metrics']['total_time']:.2f}s")


def run_rq4_real_no_detector(data_dict: dict, seed: int, ticker: str = "GSPC", retrain_interval: int = RETRAIN_INTERVAL) -> None:
    """RQ4 on real data without detector"""
    set_seed(seed)
    test_series = data_dict["y_test"]
    X_train_flat, y_train_win = _build_real_flat_windows(data_dict)
    models = {
        "RF": (train_initial_rf(X_train_flat, y_train_win, seed), "rf"),
        "SVR": (train_initial_svr(X_train_flat, y_train_win, seed), "svr")
    }
    
    for model_name, (model, model_type) in models.items():
        print(f"RQ4 Real data without detector. Running model: {model_name}")
        start = time.time()
 
        result = online_evaluation_loop_no_detector(
            model=model,
            model_type=model_type,
            test_series=test_series,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            retrain_interval=retrain_interval,
        )
 
        _log_real_result(
            result,
            model_name=f"{model_name}_no_detector",
            ticker=ticker,
            data_dict=data_dict,
            elapsed=time.time() - start,
        )
        print(f"    MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Retrains: {result['metrics']['num_retrains']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {result['metrics']['total_time']:.2f}s")
        

def run_rq4_synthetic_with_detector(series_name: str, series_number: int, seed: int) -> None:
    print(f"RQ4 Synthetic with detector. {series_name} - Series {series_number}")
    set_seed(seed)
    train_series, _, test_series, test_start_idx = load_synthetic_data(
        series_name, series_number
    )
    
    true_drifts = get_true_drift_points_synthetic(
        concept_length=CONCEPT_LENGTH,
        total_concepts=TOTAL_CONCEPTS,
        test_start_idx=test_start_idx,
        test_length=len(test_series),
        window_size=WINDOW_SIZE,
    )
    X_train_flat, y_train_win = _build_synthetic_flat_windows(train_series)
 
    models = {
        "RF":  (train_initial_rf(X_train_flat, y_train_win, seed=seed), "rf"),
        "SVR": (train_initial_svr(X_train_flat, y_train_win, seed=seed), "svr"),
    }
 
    for model_name, (model, model_type) in models.items():
        print(f"Running {model_name}...")
        detector = DriftDetector(method=BEST_DETECTOR)
        start = time.time()
 
        result = online_evaluation_loop_adaptive(
            model=model,
            model_type=model_type,
            test_series=test_series,
            detector=detector,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            true_drift_points=true_drifts,
        )
 
        result["metrics"]["total_time"] = time.time() - start
        log_results(
            model_name=f"{model_name}_{BEST_DETECTOR}",
            dataset=f"{series_name}_{series_number}",
            metrics=result["metrics"],
            path=RESULTS_FILE_RQ4_SYNTHETIC,
        )
 
        print(f"MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Drifts: {result['metrics']['num_drifts_detected']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {result['metrics']['total_time']:.2f}s")
    

def run_rq4_synthetic_no_detector(series_name: str, series_number: int, seed: int, retrain_interval: int) -> None:
    """RQ4 on synthetic data without detector"""
    print(f"RQ4 Synthetic without detector. {series_name} - Series {series_number}")
    set_seed(seed)
    train_series, _, test_series, test_start_idx = load_synthetic_data(
        series_name, series_number
    )

    X_train_flat, y_train_win = _build_synthetic_flat_windows(train_series)
 
    models = {
        "RF":  (train_initial_rf(X_train_flat, y_train_win, seed=seed), "rf"),
        "SVR": (train_initial_svr(X_train_flat, y_train_win, seed=seed), "svr"),
    }
    for model_name, (model, model_type) in models.items():
        print(f"Running {model_name}...")
        start = time.time()
 
        result = online_evaluation_loop_no_detector(
            model=model,
            model_type=model_type,
            test_series=test_series,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            retrain_interval=retrain_interval,
        )
 
        result["metrics"]["total_time"] = time.time() - start
        log_results(
            model_name=f"{model_name}_no_detector",
            dataset=f"{series_name}_{series_number}",
            metrics=result["metrics"],
            path=RESULTS_FILE_RQ4_SYNTHETIC,
        )
 
        print(f"MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Retrains: {result['metrics']['num_retrains']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {result['metrics']['total_time']:.2f}s")


def main():
    drift_types =[
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    
    ticker = "GSPC"
    
    # Real Data
    print("###################################################")
    print("Training RQ4: Real Data")
    print(f"Detector: {BEST_DETECTOR}, Seeds: {NUM_SEEDS} | Ticker: {ticker}")
    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(ticker=ticker)
    
    for seed in tqdm(range(1, NUM_SEEDS + 1), desc="RQ4 Real Data with detector", ncols=100):
        run_rq4_real_with_detector(data_dict=data_dict, seed=seed, ticker=ticker)
    
    # for seed in tqdm(range(1, NUM_SEEDS + 1), desc="RQ4 Real Data no detector", ncols=100):
    #     run_rq4_real_no_detector(data_dict=data_dict, seed=seed, ticker=ticker, retrain_interval=RETRAIN_INTERVAL)
    
    print("RQ4 Real data complete.")
    print("###################################################")
    
    
    print("RQ4 - Synthetic Data")
    for dt in drift_types:
        print(f"Drift Type: {dt}")
        for seed in tqdm(range(1, NUM_SEEDS + 1), desc=f"RQ4 Synthetic Data - {dt} with detector", ncols=100):
            run_rq4_synthetic_with_detector(series_name=dt, series_number=seed, seed=seed)
        
        # for seed in tqdm(range(1, NUM_SEEDS + 1), desc=f"RQ4 Synthetic Data - {dt} no detector", ncols=100):
        #     run_rq4_synthetic_no_detector(series_name=dt, series_number=seed, seed=seed, retrain_interval=RETRAIN_INTERVAL)
    
if __name__ == "__main__":
    main()