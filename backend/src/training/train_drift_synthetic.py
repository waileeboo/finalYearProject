# RQ2: Drift-adaptive comparison

"""
Drift adaptive comparison on Synthetic Data 

2 Phase: 
1. test 3 drift detector (Adwin, Page-Hinkley, KSWIN) on PSO-LSTM using 1 series per drift type (4 types). Then pick teh best performing detector. 

2. Compare all 4 models (PSO-LSTM, PSO-ELM, LSTM, ELM) using the best detector on 30series per drift type
"""
import time

import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm


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
from src.training.online_eval import (online_evaluation_loop,online_evaluation_loop_adaptive, get_true_drift_points_synthetic,flatten_windows, plot_drift_results, online_evaluation_loop_no_detector)
from src.utils.paths import RESULTS_FILE_RQ2_PHASE1, RESULTS_FILE_RQ2_PHASE2_SYNTHETIC, RESULTS_FILE_RQ3_SYNTHETIC, RESULTS_FILE_RQ5_SYNTHETIC

# Configuration 
WINDOW_SIZE = 10
BATCH_SIZE = 16

# LSTM backbone settings
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3
BACKBONE_EPOCHS = 1
BACKBONE_LR = 1e-4
BACKBONE_PATIENCE = 5

# PSO settings
NUM_PARTICLES = 30
MAX_ITERATIONS = 100
STOPPING_PATIENCE = 50

# ELM settings
HIDDEN_NEURONS = 10

# Drift adaptation settings
RETRAIN_WINDOW = 200

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
    """Train baseline LSTM via backprop"""
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


def load_synthetic_data_rq5(series_name: str, series_number: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load synthetic data and split into train/val/test sets for RQ5"""
    
    series = load_synthetic_series(series_name, series_number)
    # Convert to pandas Series for splitting, then back to numpy arrays
    series_pd = pd.Series(series)
    train_series, val_series, test_series = split_time_series(series_pd, train_ratio=0.1, val_ratio=0.1)
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
        log_results(model_name=f"PSO_LSTM_{method}", dataset=f"{series_name}_{series_number}", metrics=result["metrics"],path=RESULTS_FILE_RQ2_PHASE1)
    
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
    # Print true vs detected drift points per detector
    print(f"\n  True drift points:  {true_drifts}")
    for method, res in results.items():
        print(f"  [{method}] Detected: {res['drift_detected_points']}")

    # # Plot absolute error with drift markers for visual inspection
    # plot_drift_results(
    #     results=results,
    #     true_drift_points=true_drifts,
    #     title=f"Detector Comparison — {series_name} #{series_number}",
    # )
    
    return results
  
    

def run_phase2_model_comparison(best_detector: str, series_name: str, series_number: int):
    
    print(f"{series_name}-{series_number}- Running all models with best detector: {best_detector}")
    
    train_series, val_series, test_series, test_start_idx = load_synthetic_data(series_name, series_number)
    
    true_drifts = get_true_drift_points_synthetic(concept_length=CONCEPT_LENGTH,        total_concepts=TOTAL_CONCEPTS, 
                                                  test_start_idx=test_start_idx, test_length=len(test_series), window_size=WINDOW_SIZE)
    
    # # Train all 4 models with the same seed for reproducibility 
    models = {
        # "PSO_LSTM": (train_initial_pso_lstm(train_series, val_series, seed=series_number), "pso_lstm"),
        # "PSO_ELM": (train_initial_pso_elm(train_series, val_series, seed=series_number), "pso_elm"),
        "LSTM": (train_initial_lstm(train_series, val_series, seed=series_number), "lstm"),
        "ELM": (train_initial_elm(train_series, val_series, seed=series_number), "elm"),
    }    
        
    for model_name, (model, model_type) in models.items():
        print(f"    Running {model_name}...")
        model_start_time = time.time()
        detector = DriftDetector(method=best_detector)

        result = online_evaluation_loop_adaptive(
            model=model,
            model_type=model_type,
            test_series=test_series,
            detector=detector,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            true_drift_points=true_drifts,
        )
        model_elapsed = time.time() - model_start_time
        result["metrics"]["total_time"] = model_elapsed
        
        log_results(
            model_name=f"{model_name}_{best_detector}",
            dataset=f"{series_name}_{series_number}",
            metrics=result["metrics"],path=RESULTS_FILE_RQ2_PHASE2_SYNTHETIC
        )

        print(f"MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Drifts: {result['metrics']['num_drifts_detected']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {model_elapsed:.2f}s")
        
    

def run_rq3_synthetic_no_detector(
    series_name: str, 
    series_number: int,
    retrain_interval: int = 200
) -> None:
    """
    RQ3 ablation: Same models, same synthetic data, but no drift detector. 
    Results saved to RESULTS_FILE_RQ3_SYNTHETIC 
    """
    print(f"{series_name}-{series_number} — RQ3 no-detector | retrain_interval={retrain_interval}")
 
    train_series, val_series, test_series, _ = load_synthetic_data(series_name, series_number)
 
    models = {
        "PSO_LSTM": (train_initial_pso_lstm(train_series, val_series, seed=series_number), "pso_lstm"),
        "PSO_ELM": (train_initial_pso_elm(train_series, val_series, seed=series_number), "pso_elm"),
        "LSTM": (train_initial_lstm(train_series, val_series, seed=series_number), "lstm"),
        "ELM": (train_initial_elm(train_series, val_series, seed=series_number), "elm"),
    }
    
    for model_name, (model, model_type) in models.items():
        print(f"Running {model_name} without detector...")
        model_start_time = time.time()
        
        result = online_evaluation_loop_no_detector(
            model=model,
            model_type=model_type,
            test_series=test_series,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            retrain_interval=retrain_interval,
        )
        
        model_elapsed = time.time() - model_start_time
        result["metrics"]["total_time"] = model_elapsed
        
        log_results(
            model_name=f"{model_name}_no_detector",
            dataset=f"{series_name}_{series_number}",
            metrics=result["metrics"],
            path=RESULTS_FILE_RQ3_SYNTHETIC,
        )
        
        print(f"MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Retrains: {result['metrics']['num_retrains']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {model_elapsed:.2f}s")
    
    
    
def run_rq5_model_comparison(best_detector: str, series_name: str, series_number: int):
    
    print(f"{series_name}-{series_number}- Running all models with best detector: {best_detector}")
    
    train_series, val_series, test_series, test_start_idx = load_synthetic_data_rq5(series_name, series_number)
    
    true_drifts = get_true_drift_points_synthetic(concept_length=CONCEPT_LENGTH, total_concepts=TOTAL_CONCEPTS, 
                                                  test_start_idx=test_start_idx, test_length=len(test_series), window_size=WINDOW_SIZE)
    
    # # Train all 4 models with the same seed for reproducibility 
    models = {
        "PSO_LSTM": (train_initial_pso_lstm(train_series, val_series, seed=series_number), "pso_lstm"),
        "PSO_ELM": (train_initial_pso_elm(train_series, val_series, seed=series_number), "pso_elm"),
        "LSTM": (train_initial_lstm(train_series, val_series, seed=series_number), "lstm"),
        "ELM": (train_initial_elm(train_series, val_series, seed=series_number), "elm"),
    }    
        
    for model_name, (model, model_type) in models.items():
        print(f"    Running {model_name}...")
        model_start_time = time.time()
        detector = DriftDetector(method=best_detector)

        result = online_evaluation_loop_adaptive(
            model=model,
            model_type=model_type,
            test_series=test_series,
            detector=detector,
            window_size=WINDOW_SIZE,
            retrain_window=RETRAIN_WINDOW,
            true_drift_points=true_drifts,
        )
        model_elapsed = time.time() - model_start_time
        result["metrics"]["total_time"] = model_elapsed
        
        log_results(
            model_name=f"{model_name}_{best_detector}",
            dataset=f"{series_name}_{series_number}",
            metrics=result["metrics"],path=RESULTS_FILE_RQ5_SYNTHETIC
        )

        print(f"MAE: {result['metrics']['Return_MAE']:.6f} | "
              f"Drifts: {result['metrics']['num_drifts_detected']} | "
              f"Switches: {result['metrics']['model_switches']} | "
              f"Time: {model_elapsed:.2f}s")
        
        
def main():
    drift_types = [
        "linear_gradual_drift",
        "linear_abrupt_drift",
        "nonlinear_gradual_drift",
        "nonlinear_abrupt_drift",
    ]
    
    # print("##############################################################")
    # print("Phase 1: Step 1: Detector comparison on PSO-LSTM (1 series per drift type)\n")
    
    # phase1_all = {}
    # for dt in drift_types: 
    #     results = run_phase1_detector_comparison(dt, series_number=1)
    #     phase1_all[dt] = results
        
    # # Pick best performing detector from phase 1 results   
    # detector_scores = {}
    # detector_recalls = {}
    # for method in ["adwin", "page_hinkley", "kswin"]:
    #     method_scores = []
    #     method_recalls = []
    #     for dt in drift_types:
    #         method_scores.append(phase1_all[dt][method]["metrics"]["Return_MAE"])
    #         method_recalls.append(phase1_all[dt][method]["metrics"]["detect_recall"])
    #     detector_scores[method] = np.mean(method_scores)
    #     detector_recalls[method] = np.mean(method_recalls)
        
    # print("Step 2: Average MAE and Recall per Detector")
    # print(f"{'Detector':<20} {'Avg MAE':>10} {'Avg Recall':>12}")
    
    # # convert the dictionary to a sorted list of tuple. key is needed to sort the score value else it will sort by method name instead of score.
    # for method in ["adwin", "page_hinkley", "kswin"]:
    #     print(f"{method:<20} {detector_scores[method]:>10.4f} {detector_recalls[method]:>12.3f}")

    # # Pick best by recall (higher is better), break ties by MAE (lower is better)
    # best_detector = max(detector_recalls, key=lambda m: (detector_recalls[m], -detector_scores[m]))
    # print(f"\nBest performing detector: {best_detector}")
    
    # print("PHase 1 Done.")
    # print("##############################################################\n")
    best_detector = "kswin" 
    print("##############################################################")
    print("Phase 2: Model comparison using best detector (30 series per drift type)\n")
    for dt in drift_types:
        print(f"Drift Type: {dt}")
        for series_num in tqdm(range(1,3), desc=f"Phase 2 - {dt}", ncols=100):
            run_phase2_model_comparison(best_detector, dt, series_num)
    print("\nAll synthetic experiments complete.")
    print("##############################################################\n")
    
    # print("\n###############################################################")
    # print("RQ3 - No detector abilation on Synthetic data")
    # for dt in drift_types:
    #     print(f"Drift Type: {dt}")
    #     for series_num in tqdm(range(1,31), desc=f"RQ3 - No Detector - {dt}", ncols=100):
    #         run_rq3_synthetic_no_detector(series_name=dt, series_number=series_num, retrain_interval=300)
    # print("Completed RQ3 synthetic experiments complete\n")
    # print(f"Results saved to {RESULTS_FILE_RQ3_SYNTHETIC}")
    # print("###############################################################\n")
    # print("##############################################################")
    # print("RQ5: Model comparison using best detector \n")
    # for dt in drift_types:
    #     print(f"Drift Type: {dt}")
    #     for series_num in tqdm(range(1,31), desc=f"RQ5 - {dt}", ncols=100):
    #         run_rq5_model_comparison(best_detector, dt, series_num)
    # print("\nAll synthetic experiments complete.")
    # print("##############################################################\n")
    
if __name__ == "__main__":
    main()