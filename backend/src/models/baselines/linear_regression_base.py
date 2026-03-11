import numpy as np
from sklearn.linear_model import LinearRegression


class LRBase:
    """
    Linear Regression baseline model.
    Takes flattened windowed input (same as ELM).
    """

    def __init__(self):
        self.model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)