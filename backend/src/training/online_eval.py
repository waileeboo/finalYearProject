# RQ2: Drift-adaptive comparison

"""
Online evaluation loop for drift adaptive Models. 

it trigger model retrainning when drift it detected
"""
   
import numpy as np 
import time 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.data_utils.windowing import create_windows
from src.detectors.drift_detector import DriftDetector, evaluate_detector
from src.models.baselines.lstm_base import LSTMBase
from src.utils.evaluation import evaluate_returns


# retraining setup
RETRAIN_EPOCHS = 10
RETRAIN_PATIENCE = 3 # early stopping patience for LSTM retrain
RETRAIN_LR = 1e-4
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building retrain data from sliding buffer 
def build_retrain_windows(
    buffer: np.ndarray,
    window_size: int,
    val_ratio: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    take the 1d buffer of the recent observations and create windows for train/val splits.
    :param buffer: 1d array of recent observations (length should be >= window_size)
    :param window_size: size of the sliding window
    :param val_ratio: proportion of windows to use for validation
    :return: X_train, y_train, X_val, y_val
    """
    
    X = buffer.reshape(-1, 1) # reshape to 2d for windowing
    y = buffer 
    X_win, y_win = create_windows(X, y, window_size)
    
    # split chronologically into train_val sets
    n = len(X_win)
    val_size = max(int(n * val_ratio), 1) # at least 1 sample for val
    train_size = n - val_size
    
    X_train = X_win[:train_size]
    y_train = y_win[:train_size]
    X_val = X_win[train_size:]
    y_val = y_win[train_size:]
    
    return X_train, y_train, X_val, y_val

# Flatten window for ELM input 
def flatten_windows(X_windows: np.ndarray) -> np.ndarray:
    """Flatten 3D windowed data into 2D for ELM model input."""
    return X_windows.reshape(X_windows.shape[0], -1)

# Online evaluation loop 
def online_evaluation_loop(
    model,
    model_type: str,
    test_series: np.ndarray,
    detector: DriftDetector,
    window_size: int,
    retrain_window: int,
    true_drift_points: list[int],
    cooldown_period: int = 100, # minimum steps between retrains to avoid overfitting to noise
    ) -> dict:
    """Go through test set one at a time, make prediction and feed error into drift detector and retrain model when drift is detected. Return dict of results."""
    
    n_test = len(test_series)
    predictions = []
    actuals = []
    drift_detected_points = []
    retrain_times = []
    
    # slidding buffer : accumulates recent observations for retraining 
    buffer = []
    
    # cooldown counter: skip drift detection for N step after retrain to avoid overfitting to noise
    cooldown_counter = 0
    
    print(f"\nStarting online evalution loop at {n_test - window_size} steps | detector = {detector.method} | model = {model_type}")
    print(f"Retrain window: {retrain_window} Observation | True drift points (relative to test start): {true_drift_points}")
    
    for t in range(window_size, n_test):
        # build input windwo from the last window size observations 
        #create window size input during online evaluation 
        input_window = test_series[t-window_size:t]
        actual_value = test_series[t]
        
        # predict single step 
        pred = _predict_single_step(model, model_type, input_window, window_size)
        predictions.append(pred)
        actuals.append(actual_value)
        buffer.append(actual_value)
        
        # feed error into detector 
        error = abs(pred - actual_value)
        if cooldown_counter > 0:
            # Still in cooldown: skip detection, just decrement counter
            cooldown_counter -= 1
            continue
        drift_flag = detector.update(error)
        
        if drift_flag:
            drift_step = t - window_size # account for initial window offset
            drift_detected_points.append(drift_step)
            print(f"Drift detected at step {drift_step} (t={t})")
            
            # Get recent data for retraining 
            if len(buffer) >= retrain_window:
                retrain_data = np.array(buffer[-retrain_window:]) # take the most recent retrain_window observations
            else: 
                retrain_data = np.array(buffer) # take all available data
                
            # retrain if enough data is available
            # if len(retrain_data) >= window_size + 2: # need at least one window of data to retrain
            if len(retrain_data) >= retrain_window: 
                start_time = time.time()
                _retrain_model(model, model_type, retrain_data, window_size)
                elapsed = time.time() - start_time
                retrain_times.append(elapsed)
                print(f"Retraing in {elapsed:.2f} seconds on {len(retrain_data)} samples.")
                cooldown_counter = cooldown_period # set cooldown counter to avoid overfitting to noise
            else:
                print(f"Not enough data to retrain (have {len(retrain_data)}, need at least {retrain_window}). Skipping retrain.")
                
            detector.reset()
    
    # after loop evaluate the result 
    predictions = np.array(predictions)
    actuals = np.array(actuals) 
    
    metrics = evaluate_returns(actuals, predictions)
    metrics["num_drifts_detected"] = len(drift_detected_points)
    metrics["total_retrain_time"] = sum(retrain_times)
    metrics["average_retrain_time"] = np.mean(retrain_times) if retrain_times else 0.0
    
    if true_drift_points is not None: 
        detector_metrics = evaluate_detector(
            detected_points=drift_detected_points,
            true_drift_points=true_drift_points,
            tolerance= 100, # allow some tolerance in detection timing
            total_steps = len(predictions),
        )
        metrics.update({
            f"detect_{k}": v for k, v in detector_metrics.items()
            if k != "detection_delays"
        })
    
    print(f"\n  MAE: {metrics['Return_MAE']:.6f}")
    print(f"  Drifts detected: {len(drift_detected_points)}")
    print(f"  Total retrain time: {sum(retrain_times):.2f}s")

    return {
        "predictions": predictions,
        "actuals": actuals,
        "metrics": metrics,
        "drift_detected_points": drift_detected_points,
        "retrain_times": retrain_times,
    }
    


# Predict single step with current model
def _predict_single_step(model, model_type: str, input_window: np.ndarray, window_size: int) -> float:
    """Naje a subgke oreductiion fromn a window of observations using the current model"""
    if model_type == "pso_lstm":
        X_input = input_window.reshape(1, window_size, 1)
        return model.predict(X_input)[0] 
    
    elif model_type == "lstm":
        X_input = input_window.reshape(1, window_size, 1)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_input, dtype=torch.float32).to(DEVICE)
            return model(X_t).cpu().numpy().flatten()[0]
    
    elif model_type == "pso_elm":
        X_input = input_window.reshape(1, -1)
        return model.predict(X_input)[0]
    
    elif model_type == "elm":
        X_input = input_window.reshape(1, -1)
        return model.predict(X_input)[0]
    else: 
        raise ValueError(f"Unknown model type: {model_type}")
    


# Retrain model 
def _retrain_model(model, model_type: str, retrain_data: np.ndarray, window_size: int) -> None:
    """Retrain the model using recent data."""
    X_train, y_train, X_val, y_val = build_retrain_windows(retrain_data, window_size)
    
    if model_type == "pso_lstm":
        #pso reoptimises FC layer only 
        model.retrain(X_train, y_train, X_val, y_val)
        
    elif model_type == "pso_elm":
        # PSO scatters particles and re-optimises input weights
        X_train_flat = flatten_windows(X_train)
        X_val_flat = flatten_windows(X_val)
        model.retrain(X_train_flat, y_train, X_val_flat, y_val)
        
    elif model_type == "lstm":
        # Full backprop retraining on recent window
        _retrain_lstm_full(model, X_train, y_train, X_val, y_val)
        
    elif model_type == "elm":
        # Full pseudo-inverse retraining on recent window
        X_train_flat = flatten_windows(X_train)
        model.train(X_train_flat, y_train)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
        
# Retrain lstm full
def _retrain_lstm_full(model:LSTMBase, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Full retraining of LSTM model on recent data."""
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Create DataLoaders from numpy arrays
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=RETRAIN_LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    model.to(DEVICE)
    for epoch in range(RETRAIN_EPOCHS):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= RETRAIN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    

# get true drift point relative to test start
def get_true_drift_points_synthetic(
    concept_length:int,
    total_concepts: int,
    test_start_idx: int,
    test_length: int,
    window_size: int
) -> list[int]:
    """Compute drift point indices relative to the test prediction start"""
    absolute_drifts = [concept_length * i for i in range(1, total_concepts)]
    relative_drifts = []
    for dp in absolute_drifts:
        dp_rel = dp - test_start_idx - window_size # account for window size offset
        if 0 <= dp_rel < test_length:
            relative_drifts.append(dp_rel)
    return relative_drifts                                    
    
    
    
# plotting 
def plot_drift_results(results: dict, true_drift_points: list[int], title: str = "Drift Adaptive Results") -> None:
    
    # count number of model to plot 
    n_models = len(results)
    # each subplot will stack vertically for each model
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    
    # Loop through each model where ax is the current subplot axis, name is model name and res is the model results dictionary 
    for ax, (name, res) in zip(axes, results.items()):
        # extract actuals, preds and detected drift points from the result dictionary
        actuals = res["actuals"]
        preds = res["predictions"]
        detected = res["drift_detected_points"]

        # Plot absolute error at each step
        errors = np.abs(np.array(actuals) - np.array(preds))
        ax.plot(errors, label="Absolute Error", color="orange", alpha=0.7)

        # True drift points (green dashed)
        for i, dp in enumerate(true_drift_points):
            ax.axvline(dp, color="green", linestyle="--", alpha=0.5, linewidth=1,
                        label="True Drift" if i == 0 else "")

        # Detected drift points (red dotted)
        for i, dp in enumerate(detected):
            ax.axvline(dp, color="red", linestyle=":", alpha=0.6, linewidth=1.5,
                        label="Detected Drift" if i == 0 else "")

        mae = res["metrics"]["Return_MAE"]
        n_drifts = res["metrics"]["num_drifts_detected"]
        ax.set_title(f"{name} — MAE: {mae:.6f}, Drifts detected: {n_drifts}")
        ax.set_ylabel("Absolute Error")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()