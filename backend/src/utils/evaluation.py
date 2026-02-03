from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import numpy as np

def model_evaluate_metrics(actual: pd.Series, predicted: pd.Series)-> dict:
    
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    direction_actual = np.diff(actual) > 0
    direction_predicted = np.diff(predicted) > 0
    directional_accuracy = np.mean(direction_actual == direction_predicted) * 100
    return { "MAE": mae,"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2, "Directional Accuracy (%)": directional_accuracy}