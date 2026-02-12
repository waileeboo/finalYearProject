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
    # number of lag which how many previous time steps are used to remove all the correlation
    lag = result[2]
    # number of observations used for the ADF regression and calculation of the test statistic
    nobs = result[3]
    critical_values = result[4]
    print(f"ADF Statistic: {adf_statistic}")
    print(f"P-value: {p_value}")
    print(f"Lag Used: {lag}")
    print(f"Number of Observations Used: {nobs}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"{key}: {value}")
        
    if p_value < 0.05:
        print("The time series is stationary \n")
    else:
        print("The time series is non-stationary \n")


def fit_auto_arima(train_data: pd.Series)-> ARIMA:
    print("Finding for optimal Arima parameters...")
    model = auto_arima(train_data, stationary=False, seasonal=False, with_intercept=True, stepwise=True, trace=True)
    print(model.summary())
    print(f"Optimal ARIMA parameters: {model.order}\n")
    return model


def arima_forecast(model, steps: int, last_train_price: float)-> np.ndarray:
    print(f"Generating ARIMA forecast for {steps} steps...")
    # Generate forecasted log returns
    forecast_log_returns = model.predict(n_periods=steps)
    # Forecast log return give you a numpy of values based on previous day.Use cumsum to get cumulative log returns to convert to prices based on last known price
    cumulative_log_returns = np.cumsum(forecast_log_returns)
    # Convert log returns to prices
    forecast_prices = last_train_price * np.exp(cumulative_log_returns)
    return forecast_prices


