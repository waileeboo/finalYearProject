from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np



def check_stationary(train_data: pd.Series)-> None:
    print("Testing for stationary time series using ADF test...")
    result = adfuller(train_data)
    # print results
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    print(f"ADF Statistic: {adf_statistic}")
    print(f"P-value: {p_value}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value}")
        
    if p_value < 0.05:
        print("The time series is stationary (reject H0)\n")
    else:
        print("The time series is non-stationary (fail to reject H0)\n")


def auto_arima(train_data: pd.Series)-> ARIMA:
    print("Finding for optimal Arima parameters...")
    model = auto_arima(train_data, stationary=False, seasonal=False, with_intercept=True, stepwise=True, trace=True)
    print(model.summary())
    print(f"Optimal ARIMA parameters: {model.order}\n")
    print(f"AIC: {model.aic()}\n")
    return model

def arima_forecast(model, steps: int, last_actual_price=None):
    print("Generating ARIMA forecast...")
    print(f"Forecasting {steps} steps ahead...")
    forecast = model.predict(n_periods=steps)
    if last_actual_price is not None:
        forecast = pd.Series(forecast).cumsum() + last_actual_price
        cummulative_returns = np.cumsum(forecast)
        forecast_prices = last_actual_price * np.exp(cummulative_returns)
        return forecast_prices
    return forecast




