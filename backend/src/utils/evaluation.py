from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def directional_accuracy_returns(actual_ret: np.ndarray, pred_ret: np.ndarray) -> float:
    """
    % of days where predicted return sign matches actual return sign.
    """
    a = actual_ret > 0
    p = pred_ret > 0
    return float(np.mean(a == p) * 100.0)


def evaluate_returns(actual_ret: np.ndarray, pred_ret: np.ndarray) -> dict:
    mse = mean_squared_error(actual_ret, pred_ret)
    mae = mean_absolute_error(actual_ret, pred_ret)
    rmse = float(np.sqrt(mse))
    return {
        "Return_MAE": float(mae),
        "Return_MSE": float(mse),
        "Return_RMSE": rmse,
    }


def evaluate_prices(actual: pd.Series, predicted: pd.Series)-> dict:
    
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    rmse = np.sqrt(mse)
    # direction_actual = np.diff(actual) > 0
    # direction_predicted = np.diff(predicted) > 0
    # directional_accuracy = np.mean(direction_actual == direction_predicted) * 100
    return { "MAE": mae,"MSE": mse, "RMSE": rmse, "MAPE": mape}



def plot_rolling_error(actual, prediction, series_name, series_number, model_name="ELM_Baseline"):
    # calculate absolute error for each step 
    errors = np.abs(actual - prediction)
    
    # Calculate rolling MAE ( window of 50 for smoothness)
    rolling_mae = pd.Series(errors).rolling(window=50).mean()
    n = len(errors)
    plt.figure(figsize=(12, 5))
    plt.plot(rolling_mae, label='Rolling MAE (window=50)', color="blue", linewidth=1.5)
    
    # Mark Concept Boundaries (every 2000 steps)
    for x in range(2000, n, 2000):
        plt.axvline(x=x, color="red", linestyle="--", alpha=0.7, label="Drift Point" if x == 2000 else "")
        
    # add shading for Interpretation
    unseen_end = 6000
    plt.axvspan(0, 6000, color='#ff6b6b', alpha=0.2, label='Unseen Concepts (5-7)')
    plt.axvspan(6000, n, color='#51cf66', alpha=0.2, label='Recurring Concepts (8-10)')
    
    # --- INCREASED FONT SIZES ---
    # Default is usually ~12. Setting title to 24, and axis labels to 20.
    plt.title(f"RQ5: {series_name.replace('_', ' ').title()} - Series {series_number} | {model_name}", fontsize=24)
    plt.xlabel("Test Set Time Step", fontsize=20)
    plt.ylabel("Rolling MAE", fontsize=20)
    
    # Make the numbers on the axis larger so they match the new text size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Double the legend font size
    plt.legend(fontsize=16)
    # ----------------------------
    
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()