import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RFBase:
    """
    Random Forest baseline model.
    Takes flattened windowed input (same as ELM).
    """

    def __init__(self, n_estimators: int = 100, seed: int | None = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit random forest on flattened windowed input"""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on flattened windowed input"""
        return self.model.predict(X)