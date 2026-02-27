# RQ2: Drift-adaptive comparison

"""
Drift adaptive comparison on Synthetic Data 

2 Phase: 
1. test 3 drift detector (Adwin, Page-Hinkley, KSWIN) on PSO-LSTM using 1 series per drift type (4 types). Then pick teh best performing detector. 

2. Compare all 4 models (PSO-LSTM, PSO-ELM, LSTM, ELM) using the best detector on 30series per drift type
"""
import numpy as np
import pandas as pd
import random
import torch

from src.data_utils.data_loader import load_synthetic_series
from src.data_utils.preprocess import split_time_series
from src.data_utils.windowing import create_windows
from src.utils.results_logger import log_results
from src.models.PSO_LSTM import PSO_LSTM
from src.models.PSO_ELM import PSO_ELM
from src.models.baselines.lstm_base import LSTMBase
from src.models.baselines.elm_base import ELMBase
from src.training.train_baselstm import train_model, create_data_loaders
from src.detectors.drift_detector import DriftDetector
from src.training.online_eval import (online_evaluation_loop, get_true_drift_points_synthetic,flatten_windows, plot_drift_results)

# Configuration 
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
RETRAIN_WINDOW = 100

# Synthetic data settings
CONCEPT_LENGTH = 2000
TOTAL_CONCEPTS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_initial_pso_lstm(train_series: np.ndarray, val_series: np.ndarray, seed: int = 42) -> PSO_LSTM:
    """train PSO-LSTM"""
    set_seed(seed)
    
    # create Dataloadersf ro backbone LSTM training
    # X_test and y_test are not used for training, but need a placeholder to create the dataloader 
    synthetic_dict = {
        "X_train": train_series.reshape(-1, 1), # reshape to 2d for windowing
        "y_train": train_series, 
        "X_val": val_series.reshape(-1, 1),
        "y_val": val_series,
        "X_test": val_series.reshape(-1, 1),
        "y_test": val_series,
    }
    
    train_loader, val_loader, _ = create_data_loaders(synthetic_dict, WINDOW_SIZE, BATCH_SIZE)
    
    # train the model 
    trained_model, _, _, _ = train_model(
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
    
    # PSO Optimise FC Layer 
    X_train_win, y_train_win = create_windows(train_series.reshape(-1, 1), train_series, WINDOW_SIZE)
    X_val_win, y_val_win = create_windows(val_series.reshape(-1, 1), val_series, WINDOW_SIZE)
    
    pso_lstm = PSO_LSTM(
        trained_model = trained_model,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        stopping_patience=STOPPING_PATIENCE,
        seed=seed,
        device=DEVICE
    )
    
    pso_lstm.train(X_train_win, y_train_win, X_val_win, y_val_win)
    
    return pso_lstm
    
    

def train_initial_pso_elm(train_series: np.ndarray, val_series: np.ndarray, seed: int = 42) -> PSO_ELM:
    """Train PSO-ELM"""
    X_train_win, y_train_win = create_windows(train_series.reshape(-1, 1), train_series, WINDOW_SIZE)
    X_val_win, y_val_win = create_windows(val_series.reshape(-1, 1), val_series, WINDOW_SIZE)
    
    X_train_flat = flatten_windows(X_train_win)
    X_val_flat = flatten_windows(X_val_win)
    
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
    return pso_elm
    

def train_initial_elm(train_series: np.ndarray, val_series: np.ndarray, seed: int = 42) -> ELMBase:
    "Train initial ELM with random weights (no PSO optimisation)"
    X_train_win, y_train_win = create_windows(train_series.reshape(-1, 1), train_series, WINDOW_SIZE)
    
    X_train_flat = flatten_windows(X_train_win)
    
    elm = ELMBase(hidden_neurons=HIDDEN_NEURONS, seed=seed)
    elm.train(X_train_flat, y_train_win)
    return elm

def train_initial_lstm(train_series: np.ndarray, val_series: np.ndarray, seed: int = 42) -> LSTMBase:
    """Train inital LSTM with backprop only (no PSO optimisation)"""
    set_seed(seed)
    
    synthetic_dict = {
        "X_train": train_series.reshape(-1, 1), # reshape to 2d for windowing
        "y_train": train_series, 
        "X_val": val_series.reshape(-1, 1),
        "y_val": val_series,
        "X_test": val_series.reshape(-1, 1),
        "y_test": val_series,
    }
    train_loader, val_loader, _ = create_data_loaders(synthetic_dict, WINDOW_SIZE, BATCH_SIZE)
    
    trained_model, _, _, _ = train_model(
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
    return trained_model


def load_synthetic_data(series_name: str, series_number: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load synthetic data adn split into train/val/test sets"""
    
    series = load_synthetic_series(series_name, series_number)
    # Convert to pandas Series for splitting, then back to numpy arrays
    series_pd = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_pd, train_ratio=0.5, val_ratio=0.1)
    train_series= train_series.values
    val_series = val_series.values
    test_series = test_series.values
    test_start_idx = len(train_series) + len(val_series)
    return train_series, val_series, test_series, test_start_idx


# Phase 1: Test 3 detector on PSO-LSTM
def run_phase1_detector_comparison(series_name:str = "linear_gradual_drift", series_number: int = 1) -> dict:
    """Test 3 drift detectors on PSO-LSTM and return their scores for comparison"""
    print(f"Running Phase 1 Detector Comparison on {series_name} (Series #{series_number})")
    
    train_series, val_series, test_series, test_start_idx = load_synthetic_data(
        series_name, series_number
    )
    
    # Compute true drift points relative to test predictions because online loop need it
    true_drifts = get_true_drift_points_synthetic(
        concept_length=CONCEPT_LENGTH,
        total_concepts=TOTAL_CONCEPTS,
        test_start_idx=test_start_idx,
        test_length=len(test_series),
        window_size=WINDOW_SIZE,
    )
    print(f"True drift points (relative to test): {true_drifts}")
    

    # Create Detectors
    results = {}
    detector_methods = ["adwin", "page_hinkley", "kswin"]
    for method in detector_methods:
        print(f"Testing Detector: {method}")
        
        # Train initial PSO-LSTM for each detector 
        pso_lstm = train_initial_pso_lstm(train_series, val_series, seed=series_number)
        
        # create detector 
        detector = DriftDetector(method=method)
        
        # Run online evaluation loooop
        result = online_evaluation_loop(
            model=pso_lstm,
            model_type="pso_lstm",
            test_series=test_series,
            detector = detector,
            window_size = WINDOW_SIZE,
            retrain_window = RETRAIN_WINDOW,
            true_drift_points = true_drifts,
        )
        
        results[method] = result
        
        log_results(model_name=f"PSO_LSTM_{method}", dataset=f"{series_name}_{series_number}", metrics=result["metrics"])
        
        
        # print Comparison table 
    print("Detector Comparison Summary")
    header = (f"{'Detector':<20} {'MAE':>10} {'Drifts':>8} "
            f"{'Precision':>10} {'Recall':>10} {'Avg Delay':>10} {'Time(s)':>10}")
    print(header)
    print("-" * len(header))
    for method, res in results.items():
        m = res["metrics"]
        print(f"{method:<20} "
              f"{m['Return_MAE']:>10.6f} "
              f"{m['num_drifts_detected']:>8d} "
              f"{m.get('detect_precision', 0):>10.3f} "
              f"{m.get('detect_recall', 0):>10.3f} "
              f"{m.get('detect_avg_detection_delay', float('nan')):>10.1f} "
              f"{m['total_retrain_time']:>10.2f}")

    return results
   
        
    

def run_phase2_model_comparison():
    pass


def main():
    drift_types = [
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    
    print("##############################################################")
    print("Phase 1: Step 1: Detector comparison on PSO-LSTM (1 series per drift type)\n")
    
    phase1_all = {}
    for dt in drift_types: 
        results = run_phase1_detector_comparison(dt, series_number=1)
        phase1_all[dt] = results
        
    # Pick best performing detector from phase 1 results   
    detector_scores = {}
    for method in ["adwin", "page_hinkley", "kswin"]:
        method_scores = []
        for dt in drift_types:
            method_scores.append(phase1_all[dt][method]["metrics"]["Return_MAE"])
        detector_scores[method] = np.mean(method_scores)
        
    print("Step 2: Average MAE per Detector")
    # convert the dictionary to a sorted list of tuple. key is needed to sort the score value else it will sort by method name instead of score.
    for method, score in sorted(detector_scores.items(), key=lambda x: x[1]):
        print(f"{method}: {score:.4f}")
        
    best_detector = min(detector_scores, key=detector_scores.get)
    print(f"\nBest performing detector: {best_detector}\n")
    
    print("PHase 1 Done.")
    print("##############################################################\n")
    
    print("##############################################################")
    print("Phase 2: Model comparison using best detector (30 series per drift type)\n")
    
    
    
if __name__ == "__main__":
    main()