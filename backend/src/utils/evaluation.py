from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np


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