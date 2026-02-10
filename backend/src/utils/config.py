DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
# RETURN_FEATURES = ["Close_return", "High_return", "Low_return", "Open_return", "Volume_return"]
# FEATURE_COLS = ["Close","High", "Low", "Open", "Volume"]

FEATURE_COLS = ["Close","Volume"]

RETURN_FEATURES = ["Close_return", "Volume_return", ]

RETURN_FEATURES = [
    "Close_return", "Volume_return",
    "volatility_10", "momentum_5", "volume_z"
]