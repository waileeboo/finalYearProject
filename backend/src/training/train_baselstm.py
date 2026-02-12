from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import GradScaler
import random 
import pandas as pd


from src.data_utils.windowing import create_windows
from src.data_utils.preprocess import scale_features, split_time_series, scale_targets, build_feature_pipeline
from src.data_utils.data_loader import load_raw_data
from src.models.baselines.lstm_base import LSTMBase
from src.training.train_utils import train_one_epoch, evaluate
from src.utils.config import FEATURE_COLS, RETURN_FEATURES
from src.utils.evaluation import evaluate_prices, evaluate_returns

#Configuration for the training 
WINDOW_SIZE = 20
BATCH_SIZE =16
EPOCHS =10
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3
PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")
scaler = GradScaler() if DEVICE.type == "cuda" else None


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# Load and preprocess data
def load_and_preprocess_data():
    """
    Load and preprocess data for training
    """
    data = load_raw_data()
    df = data["AAPL"]
    raw_prices = df["Close"].copy()
    df = build_feature_pipeline(df, FEATURE_COLS, RETURN_FEATURES)
        
    train_df, val_df, test_df = split_time_series(df)
    X_train, X_val, X_test, feature_scaler = scale_features(train_df, val_df, test_df, RETURN_FEATURES)
    y_train, y_val, y_test, target_scaler = scale_targets(train_df, val_df, test_df, target_col="Close_return")

    return {"X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "raw_prices": raw_prices,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df
            }


# Create Sliding Wondows and DataLoaders    
def create_data_loaders(data_dict: dict, WINDOW_SIZE: int, BATCH_SIZE: int):
    """
    Create DataLoaders for training, validation and testing
    """
    
    # Create sliding windows
    X_train_win, y_train_win = create_windows(
        data_dict['X_train'],
        data_dict['y_train'],
        window_size=WINDOW_SIZE,
    )
    
    X_val_win, y_val_win = create_windows(
        data_dict['X_val'],
        data_dict['y_val'],
        window_size=WINDOW_SIZE,
    )
    
    X_test_win, y_test_win = create_windows(
        data_dict['X_test'],
        data_dict['y_test'],
        window_size=WINDOW_SIZE,
    )
    
    # Create Tensor Datasets
    X_train_t = torch.tensor(X_train_win, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_win, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_win, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_win, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_win, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_win, dtype=torch.float32)
    
    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=False, # set shuffle to False for time series data to preserve temporal order
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE,shuffle=False,
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE,shuffle=False,
    )
    
    return train_loader, val_loader, test_loader


# Train the Base LSTM model
def train_model(train_loader: DataLoader, val_loader: DataLoader, hidden_size: int, num_layers: int, dropout: float, learning_rate: float, epochs: int, patience: int = 5, num_features: int = len(FEATURE_COLS), use_amp: bool = False):
    """
    Train the Base LSTM model
    """
    model = LSTMBase(num_features=num_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_size=1)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Mixed Precision Training Setup (Scaling up the loss before backward pass and scaling down the gradients after optimizer step)
    # set a large best_val_loss to start with so that the first val loss is always better
    best_model_state = None
    patience_counter = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    
    grad_scaler = GradScaler() if (DEVICE.type == "cuda" and use_amp) else None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler=grad_scaler) 
        val_loss = evaluate(
            model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} <-- New Best\n")
        else: 
            patience_counter += 1
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | patience: {patience_counter}/{patience}\n")
            
            if patience_counter >= patience:
                print("Early stopping triggered.\n")
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    
    return model, best_val_loss, train_losses, val_losses


def generate_predictions(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions using the trained model
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    return preds, targets

def reconstruct_prices(preds: np.ndarray, targets:np.ndarray, target_scaler: MinMaxScaler, raw_prices: pd.Series, test_df_index: pd.DatetimeIndex, window_size: int) ->tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Reconstruct prices from predicted returns
    """

    preds_returns = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_returns = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    dates = test_df_index[window_size:]

    actual_prices = raw_prices.loc[dates].values
    # Take the last price from the training + validation period as the starting point til last of the test period 
    prev_dates = test_df_index[window_size-1:-1]
    prev_prices = raw_prices.loc[prev_dates].values
    pred_prices = prev_prices * np.exp(preds_returns)
    return pred_prices, actual_prices, dates, preds_returns, actual_returns


def plot_results(train_losses: list[float], val_losses: list[float], actual_prices: np.ndarray, pred_prices: np.ndarray, dates: pd.DatetimeIndex)-> None:
    
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(dates,actual_prices, label="Actual", color='blue')
    plt.plot(dates, pred_prices, label="Predicted", color='red', alpha=0.7)
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
        

def main():
    
    print("####################################################################\n")
    set_seed(42)
    print("Starting Baseline LSTM Training...\n")
    print("Step 1: Loading and Preprocessing Data\n")
    data_dict = load_and_preprocess_data()
    print(f"X_train shape: {data_dict['X_train'].shape} | y_train shape: {data_dict['y_train'].shape}\n")

    print("Step 2: Creating Sliding Windows and Setting up DataLoaders\n")
    train_loader, val_loader, test_loader = create_data_loaders(data_dict, WINDOW_SIZE, BATCH_SIZE)
    
    print("Step 3: Training Model\n")
    model, best_val_loss, train_losses, val_losses = train_model(
        train_loader, val_loader,
        hidden_size= HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        patience=PATIENCE,
        num_features=data_dict['X_train'].shape[1],
        use_amp=(DEVICE.type=="cuda"))
    
    
    print("Step 4: Evaluating on Test Data\n")
    test_loss = evaluate(
        model, test_loader,
        criterion=nn.MSELoss(),
        device=DEVICE
    )
    print(f"Test MSE Loss: {test_loss:.6f}\n")
    
    
    print("Step 5: Generating Predictions\n")
    preds, targets = generate_predictions(model, test_loader, DEVICE)
    
    print("Step 6: Reconstructing Prices and Calculating Metrics\n")
    test_start_idx = 1 + len(data_dict['X_train']) + len(data_dict['X_val']) + WINDOW_SIZE
    print(f"Test start index: {test_start_idx}")
    print(f"First prediction date: {data_dict['raw_prices'].index[test_start_idx]}")
    print(f"Number of predictions: {len(preds)}")
    print(f"Expected predictions: {len(data_dict['X_test']) - WINDOW_SIZE}")
    # Account for: 1 row dropped by add_return_features + train + val + window_size

    pred_prices, actual_prices, dates, pred_returns, actual_returns = reconstruct_prices(
        preds, targets,
        data_dict['target_scaler'],
        data_dict['raw_prices'],
        data_dict['test_df'].index,
        window_size=WINDOW_SIZE
    )
    print(f"First date: {dates[0]}")
    print(f"Actual price: {actual_prices[0]:.2f}")
    print(f"Predicted price: {pred_prices[0]:.2f}")
    print(f"Raw price at that date: {data_dict['raw_prices'].loc[dates[0]]:.2f}")

    # These should match:
    assert actual_prices[0] == data_dict['raw_prices'].loc[dates[0]]
    
    metrics = evaluate_prices(pd.Series(actual_prices), pd.Series(pred_prices))
    print("Price Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()

    return_metrics = evaluate_returns(actual_returns, pred_returns)
    print("Return Evaluation Metrics:")
    for metric, value in return_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    print("Step 7: Plotting Actual vs Predicted Prices\n")
    plot_results(train_losses, val_losses, actual_prices, pred_prices, dates)

    print("Train Base LSTM Done")
    print("################################################################\n")


if __name__ == "__main__":
    main()
