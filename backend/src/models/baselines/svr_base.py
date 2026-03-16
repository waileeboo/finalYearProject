import numpy as np 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SVRBase:
    """SVR baseline model"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1, seed: int | None = 42):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.seed = seed
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        self.scaler = StandardScaler()
        
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVR model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained SVR model"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)