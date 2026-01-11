import torch 
import torch.nn as nn

class LSTMBase(nn.Module):
    """
    Baseline LSTM model for time series forecasting.
    """
    
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        y_hat = self.fc(last_out)
        return y_hat.squeeze(-1)  