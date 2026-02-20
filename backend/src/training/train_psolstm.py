import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from tqdm import tqdm

from src.data_utils.data_loader import load_synthetic_series
from src.data_utils.preprocess import load_and_preprocess_data, split_time_series
from src.data_utils.windowing import create_windows
from src.utils.evaluation import evaluate_returns, evaluate_prices
from src.utils.results_logger import log_results
from src.models.PSO_LSTM import PSO_LSTM
from src.training.train_baselstm import train_model, create_data_loaders, reconstruct_prices

# Configuration
WINDOW_SIZE = 10
BATCH_SIZE = 16

# LSTM backbone settings (same as train_baselstm for fair comparison)
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3
BACKBONE_EPOCHS = 30
BACKBONE_LR = 1e-4
BACKBONE_PATIENCE = 5

# PSO settings for FC layer optimisation
NUM_PARTICLES = 30
MAX_ITERATIONS = 1000
STOPPING_PATIENCE = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")


def set_seed(seed: int | None = 42):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Real Data Training
def train_pso_lstm_real(ticker: str = "GSPC", seed: int | None = 42) -> None:
    print("\n###################################################################")
    print(f"PSO-LSTM Training on Real Data ({ticker})")
    set_seed(seed)

    # Step 1: Load and preprocess data
    print("Step 1: Loading and Preprocessing Data...\n")
    data_dict = load_and_preprocess_data(ticker=ticker)
    print(f"X_train: {data_dict['X_train'].shape} | X_val: {data_dict['X_val'].shape} | "
          f"X_test: {data_dict['X_test'].shape}\n")

    # Step 2: Create sliding windows and DataLoaders
    print("Step 2: Creating Sliding Windows and DataLoaders...\n")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dict, WINDOW_SIZE, BATCH_SIZE
    )

    # Step 3: Train LSTM backbone via backpropagation
    # Reuses the same train_model function from train_baselstm
    print("Step 3: Training LSTM Backbone via Backpropagation...\n")
    trained_model, best_val_loss, train_losses, val_losses = train_model(
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
    print(f"Backbone training complete. Best val loss: {best_val_loss:.6f}\n")

    # Step 4: PSO optimisation of FC layer
    # Extract windowed numpy arrays for PSO_LSTM interface
    print("Step 4: PSO Optimisation of FC Layer...\n")

    # Get windowed numpy arrays (PSO_LSTM expects numpy, not DataLoaders)
    X_train_win, y_train_win = create_windows(data_dict["X_train"], data_dict["y_train"], WINDOW_SIZE)
    X_val_win, y_val_win = create_windows(data_dict["X_val"], data_dict["y_val"], WINDOW_SIZE)
    X_test_win, y_test_win = create_windows(data_dict["X_test"], data_dict["y_test"], WINDOW_SIZE)

    pso_lstm = PSO_LSTM(
        trained_model=trained_model,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        stopping_patience=STOPPING_PATIENCE,
        seed=seed,
        device=DEVICE,
    )

    pso_lstm.train(X_train_win, y_train_win, X_val_win, y_val_win)

    # Step 5: Generate predictions on test set
    print("\nStep 5: Generating Predictions on Test Set...\n")
    preds_scaled = pso_lstm.predict(X_test_win)
    targets_scaled = y_test_win

    # Step 6: Reconstruct prices and evaluate
    print("Step 6: Reconstructing Prices and Evaluating...\n")

    raw_prices = data_dict["raw_prices"]
    target_scaler = data_dict["target_scaler"]
    test_df = data_dict["test_df"]

    pred_prices, actual_prices, dates, preds_returns, actual_returns = reconstruct_prices(
        preds_scaled, targets_scaled,
        target_scaler, raw_prices,
        test_df.index, window_size=WINDOW_SIZE,
    )

    print(f"First date: {dates[0]}")
    print(f"Actual price: {actual_prices[0]:.2f}")
    print(f"Predicted price: {pred_prices[0]:.2f}")
    print(f"Raw price at that date: {raw_prices.loc[dates[0]]:.2f}")
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
    log_results("PSO_LSTM", ticker, {**return_metrics, **price_metrics})

    # Step 7: Plot results
    # print("\nStep 7: Plotting results...")

    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # # Backbone training loss
    # axes[0, 0].plot(train_losses, label="Train Loss")
    # axes[0, 0].plot(val_losses, label="Val Loss")
    # axes[0, 0].set_title("LSTM Backbone: Training vs Validation Loss")
    # axes[0, 0].set_xlabel("Epoch")
    # axes[0, 0].set_ylabel("Loss")
    # axes[0, 0].legend()
    # axes[0, 0].grid(True, alpha=0.3)

    # # PSO convergence
    # axes[0, 1].plot(pso_lstm.pso.fitness_history)
    # axes[0, 1].set_title("PSO Convergence (FC Layer)")
    # axes[0, 1].set_xlabel("Iteration")
    # axes[0, 1].set_ylabel("Best Fitness (MAE)")
    # axes[0, 1].grid(True, alpha=0.3)

    # # Actual vs Predicted prices
    # axes[1, 0].plot(dates, actual_prices, label="Actual Price", color="blue")
    # axes[1, 0].plot(dates, pred_prices, label="PSO-LSTM Predicted", color="red", alpha=0.7)
    # axes[1, 0].set_title(f"PSO-LSTM: Actual vs Predicted ({ticker})")
    # axes[1, 0].set_xlabel("Date")
    # axes[1, 0].set_ylabel("Price")
    # axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)

    # # Prediction error over time
    # errors = np.abs(actual_prices - pred_prices)
    # axes[1, 1].plot(dates, errors, color="orange", alpha=0.7)
    # axes[1, 1].set_title("Absolute Prediction Error Over Time")
    # axes[1, 1].set_xlabel("Date")
    # axes[1, 1].set_ylabel("Absolute Error")
    # axes[1, 1].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    print("\nPSO-LSTM — Real Data Complete.")
    print("###################################################################\n")


# Synthetic Data Training
def train_pso_lstm_synthetic(series_name: str = "linear_gradual_drift", series_number: int = 1, seed: int | None = 42) -> None:
    print("\n###################################################################")
    print(f"PSO-LSTM Training on Synthetic Data ({series_name} #{series_number})")
    set_seed(seed)

    # Step 1: Load and split synthetic data
    print("Step 1: Loading and Splitting Synthetic Data...\n")
    series = load_synthetic_series(series_name, series_number)
    print(f"Series length: {len(series)}")

    series_sr = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_sr)
    train_series = train_series.values
    val_series = val_series.values
    test_series = test_series.values
    val_end = len(train_series) + len(val_series)
    print(f"Train: {len(train_series)} | Val: {len(val_series)} | Test: {len(test_series)}\n")

    # Step 2: Create sliding windows and DataLoaders
    # Reshape to 2D (n_samples, 1) for windowing — LSTM keeps 3D
    print("Step 2: Creating Sliding Windows and DataLoaders...\n")
    synthetic_dict = {
        "X_train": train_series.reshape(-1, 1),
        "y_train": train_series,
        "X_val": val_series.reshape(-1, 1),
        "y_val": val_series,
        "X_test": test_series.reshape(-1, 1),
        "y_test": test_series,
    }
    train_loader, val_loader, test_loader = create_data_loaders(
        synthetic_dict, WINDOW_SIZE, BATCH_SIZE
    )

    # Step 3: Train LSTM backbone via backpropagation
    print("Step 3: Training LSTM Backbone via Backpropagation...\n")
    trained_model, best_val_loss, train_losses, val_losses = train_model(
        train_loader, val_loader,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=BACKBONE_LR,
        epochs=BACKBONE_EPOCHS,
        patience=BACKBONE_PATIENCE,
        num_features=1,
        use_amp=(DEVICE.type == "cuda"),
    )
    print(f"Backbone training complete. Best val loss: {best_val_loss:.6f}\n")

    # Step 4: PSO optimisation of FC layer
    print("Step 4: PSO Optimisation of FC Layer...\n")

    # Get windowed numpy arrays for PSO_LSTM
    X_train_win, y_train_win = create_windows(
        train_series.reshape(-1, 1), train_series, WINDOW_SIZE
    )
    X_val_win, y_val_win = create_windows(
        val_series.reshape(-1, 1), val_series, WINDOW_SIZE
    )
    X_test_win, y_test_win = create_windows(
        test_series.reshape(-1, 1), test_series, WINDOW_SIZE
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

    # Step 5: Generate predictions on test set
    print("\nStep 5: Generating Predictions on Test Set...\n")
    preds = pso_lstm.predict(X_test_win)

    # Step 6: Evaluate (no price reconstruction for synthetic)
    print("Step 6: Evaluating on Test Data...\n")
    actual_values = test_series[WINDOW_SIZE:]

    print(f"Predicted Range: {preds.min():.4f} to {preds.max():.4f}")
    print(f"Actual Range:    {actual_values.min():.4f} to {actual_values.max():.4f}")

    metrics = evaluate_returns(actual_values, preds)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    log_results("PSO_LSTM", f"{series_name}_{series_number}", metrics)

    # Step 7: Plot results
    
    # print("\nStep 7: Plotting results...")

    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # # Backbone training loss
    # axes[0, 0].plot(train_losses, label="Train Loss")
    # axes[0, 0].plot(val_losses, label="Val Loss")
    # axes[0, 0].set_title("LSTM Backbone: Training vs Validation Loss")
    # axes[0, 0].set_xlabel("Epoch")
    # axes[0, 0].set_ylabel("Loss")
    # axes[0, 0].legend()
    # axes[0, 0].grid(True, alpha=0.3)

    # # PSO convergence
    # axes[0, 1].plot(pso_lstm.pso.fitness_history)
    # axes[0, 1].set_title("PSO Convergence (FC Layer)")
    # axes[0, 1].set_xlabel("Iteration")
    # axes[0, 1].set_ylabel("Best Fitness (MAE)")
    # axes[0, 1].grid(True, alpha=0.3)

    # # Actual vs Predicted
    # axes[1, 0].plot(actual_values, label="Actual", color="blue")
    # axes[1, 0].plot(preds, label="PSO-LSTM Predicted", color="red", alpha=0.7)

    # # Mark known drift points in test region
    # concept_size = 2000
    # total_concepts = 10
    # drift_points = [concept_size * i for i in range(1, total_concepts)]
    # for dp in drift_points:
    #     dp_relative = dp - val_end - WINDOW_SIZE
    #     if 0 <= dp_relative < len(actual_values):
    #         axes[1, 0].axvline(
    #             dp_relative, color="green", linestyle="--", alpha=0.5,
    #             label="Drift Point" if dp == drift_points[0] else "",
    #         )

    # axes[1, 0].set_title(f"PSO-LSTM: {series_name} #{series_number}")
    # axes[1, 0].set_xlabel("Time Step")
    # axes[1, 0].set_ylabel("Value")
    # axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)

    # # Prediction error over time
    # errors = np.abs(actual_values - preds)
    # axes[1, 1].plot(errors, color="orange", alpha=0.7)
    # axes[1, 1].set_title("Absolute Prediction Error Over Time")
    # axes[1, 1].set_xlabel("Time Step")
    # axes[1, 1].set_ylabel("Absolute Error")
    # axes[1, 1].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    print("\nPSO-LSTM — Synthetic Data Complete.")
    print("###################################################################\n")


# Main
def main():
    for i in tqdm(range(1, 31), desc="Training PSO-LSTM on Real Data", ncols=100): 
        train_pso_lstm_real(ticker="GSPC", seed=i)

    synthetic_series = [
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    for name in synthetic_series:
        for i in tqdm(range(1, 31), desc=f"Training PSO-LSTM on Synthetic: {name}", ncols=100):
            train_pso_lstm_synthetic(name, series_number=i, seed=i)


if __name__ == "__main__":
    main()