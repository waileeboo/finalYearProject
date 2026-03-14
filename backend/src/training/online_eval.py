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
from collections import deque
import copy
import warnings

from src.data_utils.windowing import create_windows
from src.detectors.drift_detector import DriftDetector, evaluate_detector
from src.models.baselines.lstm_base import LSTMBase
from src.utils.evaluation import evaluate_returns


# retraining setup
RETRAIN_EPOCHS = 10
RETRAIN_PATIENCE = 3 # early stopping patience for LSTM retrain
RETRAIN_LR = 1e-4
BATCH_SIZE = 16

MAX_CHALLENGERS = 2
TRIAL_STEPS = 20 
ERROR_WINDOW_SIZE = 20 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


warnings.filterwarnings("ignore",message="RNN module weights are not part of single contiguous chunk of memory") 
class ModelCandidate: 
    """Helper class to store model with its rolling error history and version label"""
    def __init__(self, model, model_type:str, label:str):
        self.model = model 
        self.model_type = model_type 
        self.label = label 
        self.errors = deque(maxlen=ERROR_WINDOW_SIZE)
    
    def record_error(self, error: float) -> None:
        """Record a new error value for this model."""
        self.errors.append(error)
        
    @property
    def mean_error(self) -> float:
        """Rolling MAE. Returns inf if no error recorded"""
        if len(self.errors) == 0:
            return float("inf")
        return float(np.mean(self.errors))

class ModelPool: 
    """Holder for multiple model cnadiates running concurrently. When drift is detected, the current model is added to the pool and a new model is trained in parallel. The pool manages the lifecycle of these candidates and selects the best performing one after a certain evaluation period."""

    
    def __init__(self, initial_model, model_type:str):
        self.current = ModelCandidate(initial_model, model_type, label="v0")
        self.challengers: list[ModelCandidate] = [] # list of ModelCandidate
        self._trial_step = 0 # count steps to call resolve_trial (compare model)
        self._trial_active = False 
        self._retrain_counter = 0 #(count step to generate label)
        self._last_challenger_preds : list[float] = [] # store predictions of the challenger 
    
    # make prediction
    def predict(self, input_window: np.ndarray, window_size: int) -> float:
        """Make prediction with the current and challenger models."""
        current_pred = _predict_single_step(self.current.model, self.current.model_type, input_window, window_size)
        
        self._last_challenger_preds = [_predict_single_step(c.model, c.model_type, input_window, window_size) for c in self.challengers]
        
        return current_pred
         
    # Error tracking 
    def update_errors(self, current_pred:float, actual: float) -> None:
        """Record errors for current and all challengers"""
        self.current.record_error(abs(current_pred - actual))
        for c, p in zip(self.challengers, self._last_challenger_preds):
            c.record_error(abs(p - actual))
        if self._trial_active and self.challengers:
            self._trial_step += 1
    
    # add challneger 
    def add_challenger(self, new_model, model_type:str) -> str:
        """add a retrained challenger. If already at MAX_CHALLENGERS, evict the worst-performing one first. Return the label of the new challenger"""
        self._retrain_counter += 1
        label = f"v{self._retrain_counter}"
                
        if len(self.challengers) >= MAX_CHALLENGERS:
            # Evict the worst performing challenger (highest MAE)
            worst_idx = 0
            for i in range(len(self.challengers)):
                if self.challengers[i].mean_error > self.challengers[worst_idx].mean_error:
                    worst_idx = i
            evicted = self.challengers.pop(worst_idx)
            
            print(f"Challegner pool full. Evicting {evicted.label}.")
        
        # Clear error of all existing challengers and current so every candiate competes fairly in the new trials 
        self.current.errors.clear()
        for c in self.challengers:
            c.errors.clear()
        
        self.challengers.append(ModelCandidate(new_model, model_type, label=label))
        self._trial_active = True 
        self._trial_step = 0
        self._last_challenger_preds = []
        return label
    
    def resolve_trial(self) -> str:
        """
        Compare current vs all challengers after TRIAL_STEPS. The model with the lowest mean error wins and becomes the new current.  Returns 'promoted' if challenger won, 'held' if current held on, 'no_trial' if no challenger.
        """
        if not self.challengers: 
            self._trial_active = False
            self._trial_step = 0
            return "no_trial"
        
        all_candidates = [self.current] + self.challengers
        best = min(all_candidates, key=lambda c: c.mean_error)
        
        # check if best is the current model or a challenger
        if best.label != self.current.label:
            print(f"Trial verdict: {best.label} wins | MAE {best.mean_error:.6f} < current {self.current.mean_error:.6f}")
            
            # clear old current's error 
            old_current = self.current 
            old_current.errors.clear()
            self.current = best 
            
            # keep old current as a challenger so it can compete if drift reverts
            self.challengers = [c for c in self.challengers if c.label != best.label]
            self.challengers.append(old_current)
            verdict = "promoted"
        
        # current is still the best 
        else:
            print(f"Trial verdict: {self.current.label} holds | MAE {self.current.mean_error:.6f} <= challengers {[c.mean_error for c in self.challengers]}")
            verdict = "held"
            
        self._last_challenger_preds = []
        self._trial_step = 0 
        self._trial_active = False
        return verdict
         
    @property
    def in_trial(self) -> bool:
        return self._trial_active
    
    @property
    def trial_step(self) -> int:
        return self._trial_step
    
    @property
    def current_label(self) -> str:
        return self.current.label


# Online evaluation loop 
def online_evaluation_loop(
    model,
    model_type: str,
    test_series: np.ndarray,
    detector: DriftDetector,
    window_size: int,
    retrain_window: int,
    true_drift_points: list[int] | None = None, # for evaluation only, not used in detection
    cooldown_period: int = 150, # minimum steps between retrains to avoid overfitting to noise
    ) -> dict:
    """
    Phase1
    Go throught the test series one by one with no trial-based Model pool. 
    1. when drift detected 
    2. retrain model with recent data (if enough data is available) and replace the current model immediately.
    """
    
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
    
    # Post Loop evaluation 
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
            tolerance= 250, # allow some tolerance in detection timing
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
    
def online_evaluation_loop_adaptive(
    model, 
    model_type: str,
    test_series: np.ndarray,
    detector: DriftDetector,
    window_size: int,
    retrain_window: int,
    true_drift_points: list[int] | None = None, # for evaluation only, not used in detection
    cooldown_period: int = 150, # minimum steps between retrains to avoid overfitting to noise
) -> dict:
    """
    Phase2
    Go throught the test series one by one using a trial-based Model pool. 
    
    When drift detected: 
    1. copy current model and retrain a new model as a challenger 
    2. Both run concurrently for TRIAL_STEPS (current model is still in used)
    3. After TRIAL_STEPS, compare the mean error of the challenger vs current. If challenger win, promote it to be the new current model. If current wins, keep challenger in pool for future trials. If pool exceed max size, evict the worst performing challenger.
    """
    
    n_test = len(test_series) # used as loop bound 
    predictions = [] # store the predicted values at each step 
    actuals = [] # store the actual values at each step 
    drift_detected_points = [] # store the step index every time drift is detected 
    # store how long each retrain took in second. Used to compute total and average retrain time at the end
    retrain_times = [] 
    # store which model version(label) is active at each step to used for plot_pool activity at the end
    active_model_history = []
    # store "promoted " or "held" for each trial that completed. Used for metrics and plotting 
    trial_verdicts = []
    # store how many times the current model was replaced by a challenger. A metric logged to CSV
    model_switches = 0
    # Count down after a trail resolves. While above 0, drift detection is skipped
    cooldown_counter = 0
    
    # slidding buffer : accumulates recent observations for retraining 
    buffer: list[float] = []
    
    # Creates the model pool with the initial model as current (labelled v0)
    pool = ModelPool(model, model_type)
    # Tracks the previous model label so we can detect when a switch happens by comparing to pool.current_label each step.
    prev_label = pool.current_label
    
    print(f"Starting online evalution loop | steps = {n_test - window_size} | detector = {detector.method} | model = {model_type} | Trial_Steps = {TRIAL_STEPS}\n")
    
    print(f"Retrain window: {retrain_window}, True drift points (relative to test start): {true_drift_points}\n")
    
    for t in range(window_size, n_test):
        input_window = test_series[t-window_size:t]
        actual_value = test_series[t]  
        
        # predict with current (challengers also predict silently inside)
        pred = pool.predict(input_window, window_size)
        
        # track model switches
        if pool.current_label != prev_label: 
            model_switches += 1
            print(f"[Step {t - window_size}] Model promoted: {prev_label} -> {pool.current_label}")
            prev_label = pool.current_label 
            
        predictions.append(pred)
        actuals.append(actual_value)
        buffer.append(actual_value)
        active_model_history.append(pool.current_label)
        
        # Update errors for current and all challengers
        pool.update_errors(pred, actual_value)
        
        # Resolve trial after TRIAL_STEPS 
        if pool.in_trial and pool.trial_step >= TRIAL_STEPS:
            verdict = pool.resolve_trial()
            trial_verdicts.append(verdict)
            cooldown_counter = cooldown_period
            continue
        
        # Skip detection during cooldown 
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue
        
        # Skip detection while a trial is still running 
        if pool.in_trial:
            continue
        
        # Normal drift detection 
        error = abs(pred - actual_value)
        drift_flag = detector.update(error)
        
        if drift_flag:
            drift_step = t - window_size
            drift_detected_points.append(drift_step)
            print(f"Drift detected at step {drift_step}")
            
            retrain_data = np.array(buffer[-retrain_window:]) if len(buffer) >= retrain_window else np.array(buffer)
        
            if len(retrain_data) >= retrain_window:
                # Deep copy current model before retraining 
                new_model = copy.deepcopy(pool.current.model)
                start = time.time()
                _retrain_model(new_model, pool.current.model_type, retrain_data, window_size)
                elapsed = time.time() - start
                retrain_times.append(elapsed)
                
                new_label = pool.add_challenger(new_model, pool.current.model_type)
                print(f" Challenger {new_label} added to pool. Retrain time: {elapsed:.2f}s. | Trial started for {TRIAL_STEPS} steps.")
            else: 
                print(f"Not enough data to retrain (need at least {retrain_window}, have {len(retrain_data)}). Skipping retrain.")
            
            detector.reset()
    
    # Post Loop evaluation 
    predictions = np.array(predictions)
    actuals     = np.array(actuals)
    
    metrics = evaluate_returns(actuals, predictions)
    metrics["num_drifts_detected"] = len(drift_detected_points)
    metrics["total_retrain_time"] = sum(retrain_times)
    metrics["average_retrain_time"] = float(np.mean(retrain_times)) if retrain_times else 0.0
    metrics["model_switches"] = model_switches
    metrics["trial_verdicts"] = trial_verdicts
    metrics["trials_promoted"] = trial_verdicts.count("promoted")
    metrics["trials_held"] = trial_verdicts.count("held")
    metrics["final_model"] = pool.current_label
    metrics["num_retrains"] = pool._retrain_counter

    
    if true_drift_points is not None:
        det_metrics = evaluate_detector(
            detected_points=drift_detected_points,
            true_drift_points=true_drift_points,
            tolerance=250,
            total_steps=len(predictions),
        )
        # det_metrics contains a dict. SO loop through all the dict item and add them to metric
        metrics.update({
            f"detect_{k}": v for k, v in det_metrics.items()
            if k != "detection_delays"
        })
        
        
    print(f"\n  MAE: {metrics['Return_MAE']:.6f}")
    print(f"  Drifts detected: {len(drift_detected_points)}")
    print(f"  Model switches: {model_switches}")
    print(f"  Trials promoted: {trial_verdicts.count('promoted')}")
    print(f"  Trials held: {trial_verdicts.count('held')}")
    print(f"  Total retrain: {sum(retrain_times):.2f}s")
    print(f"  Final model: {pool.current_label}")

    return {
        "predictions": predictions,
        "actuals": actuals,
        "metrics": metrics,
        "drift_detected_points": drift_detected_points,
        "retrain_times": retrain_times,
        "active_model_history": active_model_history,
        "trial_verdicts": trial_verdicts,
    }


def online_evaluation_loop_no_detector(
    model, 
    model_type: str,
    test_series: np.ndarray,
    window_size: int,
    retrain_window: int,
    retrain_interval: int = 200, # retrain every N steps regardless of drift detection
) -> dict:
    """RQ3 abilation: trail based adaptive loop with no drift detector. Rretrains are triggered at t fixed periodic interval instead of on detected drift."""
    n_test = len(test_series) # used as loop bound 
    predictions:list[float] = [] # store the predicted values at each step 
    actuals:list[float] = [] # store the actual values at each step 
    retrain_times:list[float] = [] # store how long each retrain took in second. Used to compute total and average retrain time at the end

    # store which model version(label) is active at each step to used for plot_pool activity at the end
    active_model_history:list[str] = []
    # store "promoted " or "held" for each trial that completed. Used for metrics and plotting 
    trial_verdicts:list[str] = []
    # store how many times the current model was replaced by a challenger. A metric logged to CSV
    model_switches = 0
    # Count down after a trail resolves. While above 0, drift detection is skipped
    steps_since_retrain = 0 
    # slidding buffer : accumulates recent observations for retraining 
    buffer: list[float] = []
    # Creates the model pool with the initial model as current (labelled v0)
    pool = ModelPool(model, model_type)
    # Tracks the previous model label so we can detect when a switch happens by comparing to pool.current_label each step.
    prev_label = pool.current_label
    
    print(f"Starting RQ3 no detector loop | Steps = {n_test - window_size} | model = {model_type} | Retrain interval = {retrain_interval}\n")
    for t in range(window_size, n_test):
        input_window = test_series[t-window_size:t]
        actual_value = test_series[t]  
        
        # predict with current (challengers also predict silently inside)
        pred = pool.predict(input_window, window_size)
        
        
        # Track model switches 
        if pool.current_label != prev_label:
            model_switches += 1
            print(f"[Step {t - window_size}] Model promoted: {prev_label} -> {pool.current_label}")
            prev_label = pool.current_label
            
        predictions.append(pred)
        actuals.append(actual_value)
        buffer.append(actual_value)
        active_model_history.append(pool.current_label)
        
        # Update rolling error for current and all challenger 
        pool.update_errors(pred, actual_value)
        
        # Resolve trail after Trails steps - reset step counter 
        if pool.in_trial and pool.trial_step >= TRIAL_STEPS:
            verdict = pool.resolve_trial()
            trial_verdicts.append(verdict)
            steps_since_retrain = 0 
            continue
        
        # skip periodic trigger whila a trail is stil running 
        if pool.in_trial:
            continue
        
        # periodic retrain trigger 
        steps_since_retrain += 1
        if steps_since_retrain < retrain_interval:
            continue
        
        # Interval reached: retrain adn reset counter 
        steps_since_retrain = 0
        retrain_data = np.array(buffer[-retrain_window:]) if len(buffer) >= retrain_window else np.array(buffer)
        
        if len(retrain_data) >= retrain_window:
            new_model = copy.deepcopy(pool.current.model)
            start = time.time()
            _retrain_model(new_model, pool.current.model_type, retrain_data, window_size)
            elapsed = time.time() - start
            retrain_times.append(elapsed)
            
            new_label = pool.add_challenger(new_model, pool.current.model_type)
            print(f"[Step {t - window_size}] Periodic retrain: Challenger {new_label} added to pool. Retrain time: {elapsed:.2f}s. | Trial started for {TRIAL_STEPS} steps.")
        else:
            raise ValueError(f"Not enough data to retrain (have {len(retrain_data)}, need at least {retrain_window}). Consider reducing retrain_window or waiting for more data to accumulate.")  
        
    # Post loop evlaution 
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    metrics = evaluate_returns(actuals, predictions)    
    metrics["num_drifts_detected"] = 0 # no detector, so no drifts detected
    metrics["total_retrain_time"] = sum(retrain_times)
    metrics["average_retrain_time"] = float(np.mean(retrain_times)) if retrain_times else 0.0
    metrics["model_switches"] = model_switches
    metrics["trial_verdicts"] = trial_verdicts
    metrics["trials_promoted"] = trial_verdicts.count("promoted")
    metrics["trials_held"] = trial_verdicts.count("held")
    metrics["final_model"] = pool.current_label
    metrics["num_retrains"] = pool._retrain_counter
    
    print(f"\n  MAE: {metrics['Return_MAE']:.6f}")
    return {
        "predictions": predictions,
        "actuals": actuals, 
        "metrics": metrics,
        "drift_detected_points": [],
        "retrain_times": retrain_times,
        "active_model_history": active_model_history,
        "trial_verdicts": trial_verdicts
    }    
        
        
        
        
        

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


# Predict single step with current model
def _predict_single_step(model, model_type: str, input_window: np.ndarray, window_size: int) -> float:
    """make a prediction for a single step using the current model"""
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
            best_state = copy.deepcopy(model.state_dict())
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
def plot_drift_results(
    results: dict,
    true_drift_points: list[int],
    title: str = "Drift Adaptive Results",
) -> None:
    """
    the absolute prediction error over time (orange)
    
    Green Dashed lines: where true drifts occur 
    
    Red Dotted lines: where drifts are detected by the model
    
    """
    # count number of model to plot 
    n_models = len(results)
        
    # each subplot will stack vertically for each model
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    
    # Loop through each model where ax is the current subplot axis, name is model name and res is the model results dictionary 
    for ax, (name, res) in zip(axes, results.items()):
        errors   = np.abs(np.array(res["actuals"]) - np.array(res["predictions"]))
        detected = res["drift_detected_points"]

        ax.plot(errors, label="Absolute Error", color="orange", alpha=0.7)

        for i, dp in enumerate(true_drift_points):
            ax.axvline(dp, color="green", linestyle="--", alpha=0.5, linewidth=1,
                       label="True Drift" if i == 0 else "")
        for i, dp in enumerate(detected):
            ax.axvline(dp, color="red", linestyle=":", alpha=0.6, linewidth=1.5,
                       label="Detected Drift" if i == 0 else "")

        mae      = res["metrics"]["Return_MAE"]
        n_drifts = res["metrics"]["num_drifts_detected"]
        switches = res["metrics"].get("model_switches", "n/a")
        ax.set_title(f"{name} — MAE: {mae:.6f}  |  Drifts: {n_drifts}  |  Switches: {switches}")
        ax.set_ylabel("Absolute Error")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    
