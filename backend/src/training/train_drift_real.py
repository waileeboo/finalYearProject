# RQ2: Drift-adaptive comparison on Real Financial Data

"""
Online evaluation loop for drift-adaptive models on real stock data (GSPC).

KSWIN is used as the drift detector — selected from Phase 1 synthetic
experiments and applied consistently here for comparability.

Runs 30 seeds per model, comparing:
    PSO_LSTM, PSO_ELM, LSTM, ELM  — all using trial-based adaptive loop

Results saved to: rq2_phase2_real_results.csv
"""

import time
import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data_utils.preprocess import load_and_preprocess_data
from src.data_utils.windowing import create_windows
from src.utils.evaluation import evaluate_returns, evaluate_prices
from src.utils.results_logger import log_results
from src.models.PSO_LSTM import PSO_LSTM
from src.models.PSO_ELM import PSO_ELM
from src.models.baselines.lstm_base import LSTMBase
from src.models.baselines.elm_base import ELMBase
from src.training.train_baselstm import train_model, create_data_loaders
from src.detectors.drift_detector import DriftDetector
from src.training.online_eval import online_evaluation_loop_adaptive, flatten_windows
from src.utils.paths import RESULTS_FILE_RQ2_PHASE2_REAL 

# Configuration

# KSWIN selected from Phase 1 synthetic experiments — applied consistently
BEST_DETECTOR = "kswin"

WINDOW_SIZE = 10
BATCH_SIZE = 16

# LSTM backbone settings
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3
BACKBONE_EPOCHS = 30
BACKBONE_LR = 1e-4
BACKBONE_PATIENCE = 5

# PSO settings
NUM_PARTICLES = 30
MAX_ITERATIONS = 1000
STOPPING_PATIENCE = 50

# ELM settings
HIDDEN_NEURONS = 10

# Drift adaptation settings
RETRAIN_WINDOW = 200

# Number of seeds for repeated experiments
NUM_SEEDS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")


# reproducibility
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# data loading and preprocess 
def load_real_data(ticker: str = "GSPC") -> dict:
    """Load and preprocess real stock data."""
    return load_and_preprocess_data(ticker=ticker)


# Initial model training
def train_initial_pso_lstm(data_dict: dict, seed: int = 42) -> PSO_LSTM:
    """Train LSTM backbone via backprop, then PSO-optimise the FC layer."""
    set_seed(seed)

    train_loader, val_loader, _ = create_data_loaders(
        data_dict, WINDOW_SIZE, BATCH_SIZE
    )

    trained_model, _, _, _ = train_model(
        train_loader, val_loader,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=BACKBONE_LR,
        epochs=BACKBONE_EPOCHS,
        patience=BACKBONE_PATIENCE,
        num_features=data_dict["X_train"].shape[1],
        use_amp=(DEVICE.type == "cuda"),
    )

    X_train_win, y_train_win = create_windows(
        data_dict["X_train"], data_dict["y_train"], WINDOW_SIZE
    )
    X_val_win, y_val_win = create_windows(
        data_dict["X_val"], data_dict["y_val"], WINDOW_SIZE
    )

    pso_lstm = PSO_LSTM(
        trained_model=trained_model,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        stopping_patience=STOPPING_PATIENCE,
        seed=seed,
        device=DEVICE,
    )
    pso_lstm.train(X_train_win, y_train_win, X_val_win, y_val_win)

    return pso_lstm


def train_initial_pso_elm(data_dict: dict, seed: int = 42) -> PSO_ELM:
    """Train PSO-ELM on initial data."""
    X_train_win, y_train_win = create_windows(
        data_dict["X_train"], data_dict["y_train"], WINDOW_SIZE
    )
    X_val_win, y_val_win = create_windows(
        data_dict["X_val"], data_dict["y_val"], WINDOW_SIZE
    )

    X_train_flat = flatten_windows(X_train_win)
    X_val_flat   = flatten_windows(X_val_win)

    pso_elm = PSO_ELM(
        hidden_neurons=HIDDEN_NEURONS,
        window_size=WINDOW_SIZE,
        num_features=data_dict["X_train"].shape[1],
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        stopping_patience=STOPPING_PATIENCE,
        seed=seed,
    )
    pso_elm.train(X_train_flat, y_train_win, X_val_flat, y_val_win)

    return pso_elm


def train_initial_lstm(data_dict: dict, seed: int = 42) -> LSTMBase:
    """Train baseline LSTM via backprop."""
    set_seed(seed)

    train_loader, val_loader, _ = create_data_loaders(
        data_dict, WINDOW_SIZE, BATCH_SIZE
    )

    trained_model, _, _, _ = train_model(
        train_loader, val_loader,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=BACKBONE_LR,
        epochs=BACKBONE_EPOCHS,
        patience=BACKBONE_PATIENCE,
        num_features=data_dict["X_train"].shape[1],
        use_amp=(DEVICE.type == "cuda"),
    )

    return trained_model


def train_initial_elm(data_dict: dict, seed: int = 42) -> ELMBase:
    """Train initial ELM with random weights (no PSO optimisation)"""
    X_train_win, y_train_win = create_windows(
        data_dict["X_train"], data_dict["y_train"], WINDOW_SIZE
    )
    X_train_flat = flatten_windows(X_train_win)

    elm = ELMBase(hidden_neurons=HIDDEN_NEURONS, seed=seed)
    elm.train(X_train_flat, y_train_win)

    return elm


# price reconstruct 
def reconstruct_prices(
    preds_scaled: np.ndarray,
    actuals_scaled: np.ndarray,
    target_scaler,
    raw_prices: pd.Series,
    test_df_index: pd.DatetimeIndex,
) -> dict:
    """
    Inverse-transform scaled predictions back to log returns,
    then reconstruct prices using Price_t = Price_{t-1} * exp(Return_t).
    """
    preds_returns = target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()
    actual_returns = target_scaler.inverse_transform(
        actuals_scaled.reshape(-1, 1)
    ).flatten()

    dates = test_df_index[WINDOW_SIZE:]
    prev_dates = test_df_index[WINDOW_SIZE - 1:-1]
    prev_prices = raw_prices.loc[prev_dates].values
    actual_prices = raw_prices.loc[dates].values
    pred_prices = prev_prices * np.exp(preds_returns)

    return {
        "pred_prices":    pred_prices,
        "actual_prices":  actual_prices,
        "pred_returns":   preds_returns,
        "actual_returns": actual_returns,
        "dates":          dates,
    }


# Model comparison
def run_phase2_model_comparison(
    data_dict: dict,
    seed: int = 1,
    ticker: str = "GSPC",
) -> None:
    """
    Compare all 4 adaptive models on real stock data using drift detector.
    Results are logged to RESULTS_FILE_RQ2_PHASE2_REAL.
    """
    
    test_series = data_dict["y_test"]
    
    models = {
        # "PSO_LSTM":(train_initial_pso_lstm(data_dict, seed=seed), "pso_lstm"),
        "PSO_ELM":(train_initial_pso_elm(data_dict, seed=seed), "pso_elm"),
        # "LSTM":(train_initial_lstm(data_dict, seed=seed), "lstm"),
        "ELM":(train_initial_elm(data_dict, seed=seed), "elm"),
    }


    for model_name, (model, model_type) in models.items():
        print(f"  Running {model_name}...")
        detector    = DriftDetector(method=BEST_DETECTOR)
        model_start_time = time.time()

        result = online_evaluation_loop_adaptive(
            model=model,
            model_type=model_type,
            test_series=test_series,
            detector=detector,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            true_drift_points=None,
            cooldown_period=200,
        )

        model_elapsed = time.time() - model_start_time
        result["metrics"]["total_time"] = model_elapsed

        # Reconstruct prices and compute price-level metrics
        price_info = reconstruct_prices(
            result["predictions"], result["actuals"],
            data_dict["target_scaler"],
            data_dict["raw_prices"],
            data_dict["test_df"].index,
        )
        # Overwrite scaled return metrics with unscaled for fair comparison with RQ1
        unscaled_return_metrics = evaluate_returns(price_info["actual_returns"], price_info["pred_returns"])
        result["metrics"]["Return_MAE"] = unscaled_return_metrics["Return_MAE"]
        result["metrics"]["Return_MSE"] = unscaled_return_metrics["Return_MSE"]
        result["metrics"]["Return_RMSE"] = unscaled_return_metrics["Return_RMSE"]

        price_metrics = evaluate_prices(
            pd.Series(price_info["actual_prices"]),
            pd.Series(price_info["pred_prices"]),
        )
        result["metrics"].update({f"price_{k}": v for k, v in price_metrics.items()})

        log_results(
            model_name=f"{model_name}_{BEST_DETECTOR}",
            dataset=ticker,
            metrics=result["metrics"],
            path=RESULTS_FILE_RQ2_PHASE2_REAL,
        )

        print(f"MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Drifts: {result['metrics']['num_drifts_detected']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {model_elapsed:.2f}s")


def main():
    ticker = "GSPC"

    print("#####################################################################")
    print("RQ2 — Drift-Adaptive Model Comparison on Real Data")
    print(f"Detector : {BEST_DETECTOR} | Seeds: {NUM_SEEDS} | Ticker: {ticker}")

    # Load data once, reused across all seeds and models
    print("Loading and preprocessing real data...")
    data_dict = load_real_data(ticker=ticker)
    print(f"Train: {len(data_dict['X_train'])} | "
          f"Val  : {len(data_dict['X_val'])} | "
          f"Test : {len(data_dict['X_test'])}\n")

    # Run 30 seeds
    for seed in tqdm(range(1, NUM_SEEDS + 1), desc="Seeds", ncols=100):
        run_phase2_model_comparison(data_dict, seed=seed, ticker=ticker)

    print("All real data experiments complete.")
    print(f"Results saved to: {RESULTS_FILE_RQ2_PHASE2_REAL}")
    print("\n" + "#####################################################################")


if __name__ == "__main__":
    main()