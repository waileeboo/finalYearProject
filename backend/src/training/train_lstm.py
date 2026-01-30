from sklearn.metrics import mean_squared_error
import copy
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.data_utils.windowing import create_windows
from src.data_utils.preprocess import scale_features, split_time_series, add_return_features
from src.data_utils.data_loader import load_raw_data
from src.models.baselines.lstm_base import LSTMBase
from src.training.train_utils import train_one_epoch, evaluate
from src.utils.config import FEATURE_COLS, RETURN_FEATURES
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Configuration for the training 
WINDOW_SIZE = 20
BATCH_SIZE =32
EPOCHS = 8
LEARNING_RATE = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")


def main():
    # Load Data 
    data = load_raw_data()
    df = data["AAPL"]

    # Keep raw prices for reconstruction later (before we drop them in preprocessing)
    raw_prices = df["Close"].copy()
    
    # Preprocess 
    # Define the new input columns including return features 
    df = add_return_features(df, FEATURE_COLS)
    # split the dataset 
    train_df, val_df, test_df = split_time_series(df)
    # scale the X features 
    X_train, X_val, X_test, scaler = scale_features(
        train_df, val_df, test_df, RETURN_FEATURES
    )
    print("Feature scaling complete. Scaled features to range [-1, 1].")
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}\n")
    
    # scale the y (return_closed Price)
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train = y_scaler.fit_transform(train_df[["Close_return"]]).flatten()
    y_val = y_scaler.transform(val_df[["Close_return"]]).flatten()
    y_test = y_scaler.transform(test_df[["Close_return"]]).flatten()
    
    print("Feature scaling complete for target variable 'Close_return'.")
    print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}\n")
    
    # Windowing 
    print("Creating sliding windows...")
    print("Creating training windows...")
    X_train_win, y_train_win = create_windows(X_train, y_train, WINDOW_SIZE)
    print(f"X_train_win shape: {X_train_win.shape}, y_train_win shape: {y_train_win.shape}\n")
    print("Creating validation windows...") 
    X_val_win, y_val_win = create_windows(X_val, y_val, WINDOW_SIZE)
    print(f"X_val_win shape: {X_val_win.shape}, y_val_win shape: {y_val_win.shape}\n")
    print("Creating test windows...")
    X_test_win, y_test_win = create_windows(X_test, y_test, WINDOW_SIZE)
    print(f"X_test_win shape: {X_test_win.shape}, y_test_win shape: {y_test_win.shape}\n")
    
    # Convert to tensors 
    print("Converting numpy training windows to tensors...")
    X_train_t = torch.tensor(X_train_win, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_win, dtype=torch.float32)
    
    print("Converting numpy validation windows to tensors...")
    X_val_t = torch.tensor(X_val_win, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_win, dtype=torch.float32)
    
    print("Converting numpy test windows to tensors...")
    X_test_t = torch.tensor(X_test_win, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_win, dtype=torch.float32)
    
    # DataLoaders
    print("Creating DataLoaders...\n")
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),  
        batch_size=BATCH_SIZE,
        shuffle=False,
    )   
    
    # Model, Optimizer, Loss 
    model = LSTMBase(num_features=len(FEATURE_COLS), hidden_size=32, num_layers=2, dropout=0.5, output_size=1)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    


    # Training loop 
    print("Starting BaseLine LSTM training...\n")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss = evaluate(
            model, val_loader, criterion, DEVICE
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
        
    model.eval()
    print("Evaluating on test data...")
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch) 
            all_preds.append(outputs.cpu().numpy())

    # 1. Concatenate into (TotalSamples, 1)
    preds_combined = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    # 2. Inverse transform
    preds_actual = y_scaler.inverse_transform(preds_combined).flatten()

    # 3. Reconstruct Prices: Price_t = Price_{t-1} * exp(Return_t)
    # We need the raw prices corresponding to the test set indices
    # The targets start at WINDOW_SIZE because the first WINDOW_SIZE samples are used as input
    test_dates = test_df.index[WINDOW_SIZE:]
    
    # We need the price from the day BEFORE the target date to apply the return
    # Indices: [WINDOW_SIZE-1] to [len - 1]
    prev_dates = test_df.index[WINDOW_SIZE-1 : -1]
    
    prev_prices = raw_prices.loc[prev_dates].values
    actual_prices = raw_prices.loc[test_dates].values
    
    # Calculate predicted prices using the exponential of log returns
    pred_prices = prev_prices * np.exp(preds_actual)

    # Print ranges to debug
    print(f"Predicted Price Range: {pred_prices.min():.2f} to {pred_prices.max():.2f}")
    print(f"Actual Price Range: {actual_prices.min():.2f} to {actual_prices.max():.2f}")

    # Plot Training vs Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_prices, label="Actual Price", color='blue')
    plt.plot(test_dates, pred_prices, label="Predicted Price", color='red', alpha=0.7)
    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
