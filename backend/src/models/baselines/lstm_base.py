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
    
    
    
    
    # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(32, output_size)
        # )
        
        
        
        


# class BiLSTMBase(nn.Module):
#     def __init__(self, num_features: int = NUM_FEATURES, hidden_size=64, num_layers=1, dropout=0.3):
#         super().__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(
#             input_size=num_features,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout,
#             bidirectional=True  
#         )
        
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 1)
#         )
        
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
        
     
#         last_out = lstm_out[:, -1, :]
        
#         return self.fc(last_out)