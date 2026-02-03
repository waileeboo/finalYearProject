import numpy as np 

def create_windows(X: np.ndarray, y:np.ndarray, window_size: int) ->tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for stock data
    """
    
    if window_size <= 0:
        raise ValueError("window_size must be positive integer")
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    num_samples = len(X) - window_size
    
    if num_samples <= 0:
        raise ValueError("window_size larger than number of samples in X")
    
    X_windows = np.zeros((num_samples, window_size, X.shape[1]))
    y_windows = np.zeros(num_samples)
    
    print(f"Creating {num_samples} windows of size {window_size}...\n")
    
    for i in range(num_samples):
        X_windows[i] = X[i:i+window_size]
        y_windows[i] = y[i+window_size]

    return X_windows, y_windows