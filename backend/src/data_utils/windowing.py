import numpy as np 
from src.utils.config import NUM_FEATURES
from typing import Tuple

def create_windows(X: np.ndarray, y:np.ndarray, window_size: int) ->tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for stock data
    """
    
    if NUM_FEATURES != X.shape[1]:
        raise ValueError(f"Expected X to have {NUM_FEATURES} features, but got {X.shape[1]}")
    
    if window_size <= 0:
        raise ValueError("window_size must be positive integer")
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    num_samples = len(X) - window_size
    
    if num_samples <= 0:
        raise ValueError("window_size larger than number of samples in X")
    
    X_windows = np.zeros((num_samples, window_size, NUM_FEATURES))
    y_windows = np.zeros(num_samples)
    
    for i in range(num_samples):
        X_windows[i] = X[i:i+window_size]
        y_windows[i] = y[i+window_size]

    return X_windows, y_windows