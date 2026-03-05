import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from src.models.baselines.lstm_base import LSTMBase
from src.models.optimisers.PSO import PSO

class PSO_LSTM:
    """
    PSO-optimised LSTM.
    
    Trains LSTM with backprop first, then freezes LSTM layers
    and uses PSO to optimise the FC (output) layer weights.
    """

    def __init__(
        self,
        trained_model: LSTMBase,
        num_particles: int = 30,
        max_iterations: int = 1000,
        stopping_patience: int = 50,
        seed: int | None = 42,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.model = trained_model.to(self.device)
        self.seed = seed

        # Freeze all LSTM layers — only FC layer will be optimised
        for name, param in self.model.named_parameters():
            # every parameter is not just a tensor but a object with metadata (name, requires_grad, etc.) so we check if "fc" is in the name to identify the FC layer and only allow those parameters to be optimised by PSO
            if "fc" not in name:
                param.requires_grad = False

        # Calculate FC layer dimensions
        # fc.weight shape: (output_size, hidden_size) + fc.bias shape: (output_size,)
        fc_weight_size = self.model.fc.weight.numel()
        fc_bias_size = self.model.fc.bias.numel()
        self.fc_weight_shape = self.model.fc.weight.shape
        self.fc_bias_shape = self.model.fc.bias.shape
        self.num_dimensions = fc_weight_size + fc_bias_size

        print(f"PSO-LSTM — FC weight shape: {self.fc_weight_shape}, "
              f"FC bias shape: {self.fc_bias_shape}, "
              f"Total dimensions: {self.num_dimensions}")

        # Create PSO with fitness function
        self.pso = PSO(
            num_dimensions=self.num_dimensions,
            fitness_fn=self._fitness,
            num_particles=num_particles,
            max_iterations=max_iterations,
            stopping_patience=stopping_patience,
            seed=seed,
        )

        # Data references (set during train)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def _inject_fc_weights(self, position: np.ndarray) -> None:
        """Inject PSO particle position into the FC layer weights and bias."""
        fc_weight_size = self.model.fc.weight.numel()

        weight_values = position[:fc_weight_size]
        bias_values = position[fc_weight_size:]

        with torch.no_grad():
            # in place copy by replacing the contents of this tensor with the new value. Pytorch track the parameter as a leaf node in the computational graph, so we can't just assign a new tensor to self.model.fc.weight because that would break the reference. Instead, we copy the values into the existing tensor to preserve the reference and ensure it works with the frozen LSTM layers.
            self.model.fc.weight.copy_(
                torch.tensor(weight_values, dtype=torch.float32).reshape(self.fc_weight_shape)
            )
            self.model.fc.bias.copy_(
                torch.tensor(bias_values, dtype=torch.float32).reshape(self.fc_bias_shape)
            )

    def _fitness(self, position: np.ndarray) -> float:
        """
        Fitness function for PSO.
        Injects FC weights, runs forward pass through frozen LSTM, returns MAE.
        """
        self._inject_fc_weights(position)

        self.model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
            preds = self.model(X_val_t).cpu().numpy()

        return mean_absolute_error(self.y_val, preds)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Run PSO to optimise FC layer weights.
        
        :param X_train: training features (not used during PSO, kept for retrain)
        :param y_train: training targets
        :param X_val: validation features (used for fitness evaluation)
        :param y_val: validation targets
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        print(f"PSO-LSTM Training — {self.pso.num_particles} particles, "
              f"{self.num_dimensions} dimensions (FC layer)")

        best_position = self.pso.train()
        self._inject_fc_weights(best_position)
        print("Best FC weights injected into LSTM.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using frozen LSTM + optimised FC layer."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_t).cpu().numpy()
        return preds

    def retrain(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Retrain FC layer after drift detection."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        best_position = self.pso.retrain()
        self._inject_fc_weights(best_position)
        print("PSO-LSTM retrained after drift.")