"""
PSO-ELM: Particle Swarm Optimisation applied to Extreme Learning Machine.

PSO searches for optimal ELM input weights. Each particle's position represents
a flattened input weight matrix. The fitness function trains an ELM with those
weights and evaluates on the validation set.

After PSO converges, the best ELM is extracted for predictions.
The swarm stays alive for retraining when drift is detected.
"""

import numpy as np
from src.models.optimisers.PSO import PSO
from src.models.baselines.elm_base import ELMBase
from sklearn.metrics import mean_absolute_error


class PSO_ELM:
    def __init__(
        self,
        hidden_neurons: int = 10,
        window_size: int = 5,
        num_features: int = 1,
        num_particles: int = 30,
        max_iterations: int = 1000,
        inertia: float = 0.5,
        c1: float = 2.4,
        c2: float = 1.4,
        vel_max: float = 5.0,
        pos_min: float = -1.0,
        pos_max: float = 1.0,
        stopping_patience: int = 50,
        scatter_rate: float = 0.25,
        seed: int = 42,
    ):
        """
        :param hidden_neurons: number of ELM hidden neurons
        :param window_size: number of time lags in the input window
        :param num_features: number of input features
        :param num_particles: number of PSO particles
        :param max_iterations: max PSO iterations
        :param inertia: PSO inertia weight
        :param c1: cognitive coefficient
        :param c2: social coefficient
        :param vel_max: maximum velocity
        :param pos_min: minimum position bound
        :param pos_max: maximum position bound
        :param stopping_patience: early stopping patience
        :param scatter_rate: fraction of particles to reinitialise on drift
        :param seed: random seed
        """
        self.hidden_neurons = hidden_neurons
        self.seed = seed

        # Input dimension = flattened window + bias
        self.input_dim = (window_size * num_features) + 1
        # Total weights PSO needs to search over
        self.num_dimensions = self.input_dim * hidden_neurons

        # Data references (set during train)
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None

        # Best ELM model after PSO converges
        self.best_elm: ELMBase | None = None

        # PSO instance
        self.pso = PSO(
            num_dimensions=self.num_dimensions,
            fitness_fn=self._fitness,
            num_particles=num_particles,
            max_iterations=max_iterations,
            inertia=inertia,
            c1=c1,
            c2=c2,
            vel_max=vel_max,
            pos_min=pos_min,
            pos_max=pos_max,
            stopping_patience=stopping_patience,
            scatter_rate=scatter_rate,
            seed=seed,
        )

    def _fitness(self, position: np.ndarray) -> float:
        """
        Fitness function for PSO. Takes a particle's position (flattened weights),
        reshapes into ELM input weight matrix, trains ELM, and returns MAE on
        validation set.

        :param position: flattened weight vector from a PSO particle
        :return: MAE on validation set (lower is better)
        """
        # Reshape particle position into weight matrix (input_dim x hidden_neurons)
        weights = position.reshape(self.input_dim, self.hidden_neurons)

        # Train ELM with these weights
        elm = ELMBase(hidden_neurons=self.hidden_neurons, seed=self.seed)
        elm.train(self.X_train, self.y_train, weights=weights)

        # Evaluate on validation set
        predictions = elm.predict(self.X_val)
        return mean_absolute_error(self.y_val, predictions)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Train PSO-ELM: run PSO to find optimal input weights, then extract
        the best ELM.

        :param X_train: training features (n_samples, window_size * n_features)
        :param y_train: training targets (n_samples,)
        :param X_val: validation features
        :param y_val: validation targets
        """
        # Store data references for fitness function
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        print(f"PSO-ELM Training — {self.pso.num_particles} particles, "
              f"{self.num_dimensions} dimensions, "
              f"{self.hidden_neurons} hidden neurons")

        # Run PSO
        best_weights = self.pso.train()

        # Extract best ELM
        self._build_best_elm(best_weights)

    def _build_best_elm(self, weights: np.ndarray) -> None:
        """
        Create the final ELM using the best weights found by PSO.

        :param weights: flattened weight vector (gbest position)
        """
        weight_matrix = weights.reshape(self.input_dim, self.hidden_neurons)
        self.best_elm = ELMBase(hidden_neurons=self.hidden_neurons, seed=self.seed)
        self.best_elm.train(self.X_train, self.y_train, weights=weight_matrix)
        print("Best ELM extracted from PSO.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the best ELM found by PSO.

        :param X: input features (n_samples, window_size * n_features)
        :return: predictions (n_samples,)
        """
        if self.best_elm is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self.best_elm.predict(X)



    def retrain(
        self,
        X_train_new: np.ndarray,
        y_train_new: np.ndarray,
        X_val_new: np.ndarray,
        y_val_new: np.ndarray,
    ) -> None:
        """
        Retrain after drift detection. Updates training data, scatters
        some particles, then runs PSO again. Swarm retains knowledge
        from surviving particles.
    
        :param X_train_new: new training features after drift
        :param y_train_new: new training targets after drift
        :param X_val_new: new validation features after drift
        :param y_val_new: new validation targets after drift
        """
        # Update data references
        self.X_train = X_train_new
        self.y_train = y_train_new
        self.X_val = X_val_new
        self.y_val = y_val_new
    
        print("PSO-ELM Retraining after drift...")
    
        # Retrain PSO (scatter + optimise)
        best_weights = self.pso.retrain()
    
        # Extract new best ELM
        self._build_best_elm(best_weights)