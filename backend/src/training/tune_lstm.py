"""
Optuna hyperparameter tuning for LSTM model.

Usage:
    python tune_lstm.py
"""
import optuna
from optuna.trial import Trial
import json
from datetime import datetime
import torch

from src.training.train_baselstm import (
    load_and_preprocess_data, 
    create_data_loaders, 
    train_model
)
from src.utils.config import FEATURE_COLS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial: Trial, data_dict: dict) -> float:
    """
    Optuna objective function for hyperparameter optimization
    """
    # Hyperparameters to tune
    window_size = trial.suggest_int("window_size", 20, 100, step=10)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # Create dataloaders with trial hyperparameters
    train_loader, val_loader, _ = create_data_loaders(
        data_dict, window_size, batch_size
    )
    
    # Train model
    _, best_val_loss, _, _ = train_model(
        train_loader, val_loader,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=20,  # More epochs for tuning
        num_features=len(FEATURE_COLS),
        use_amp=True,
    )
    
    return best_val_loss


def main():
    print("################################################################")
    print("LSTM Hyperparameter Tuning with Optuna\n")
    print(f"Device: {DEVICE}\n")
    
    # Load data ONCE (reused across all trials)
    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data()
    print("Data loaded successfully!\n")
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name="lstm_hpo",
        # Pruner steps bad traials early to save time( don't prune first 5 trials and don't prune until 5 steps into training)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run 50 Trials 
    n_trials = 50
    print(f"Starting optimization with {n_trials} trials...\n")
    
    study.optimize(
        lambda trial: objective(trial, data_dict),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Results
    print("\n################################################" )
    print("OPTIMIZATION COMPLETE")
    
    print(f"\nBest trial:")
    print(f"  Value (Val Loss): {study.best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "best_val_loss": study.best_trial.value,
        "best_params": study.best_params,
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "device": str(DEVICE)
    }
    
    output_file = "best_lstm_params.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Optionally show top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value").head(5)
    print(trials_df[["number", "value", "params_hidden_size", "params_num_layers", 
                     "params_dropout", "params_learning_rate"]].to_string(index=False))
    
    return study


if __name__ == "__main__":
    study = main()