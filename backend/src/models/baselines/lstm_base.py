import torch 
import torch.nn as nn


class LSTMBase(nn.Module):
    """
    Baseline LSTM model for time series forecasting.
    """
    
    def __init__(self, num_features: int, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.3, output_size: int = 1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            # Data shape: (batch, seq_len, input_size) when using DataLoader so set Batch_first=True
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)  
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        y_hat = self.fc(last_out)
        return y_hat.squeeze(-1)  
    
