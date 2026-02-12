import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.data_utils.data_loader import load_raw_data, load_synthetic_series
from src.data_utils.preprocess import add_return_features, split_time_series
from src.data_utils.windowing import create_windows 
from src.models.baselines.elm_base import ELMBase
from src.utils.config import FEATURE_COLS, RETURN_FEATURES
from src.evaluation.evaluation import evaluation_returns

WINDOW_SISE = 10
HIDDEN_NEURONS = 10

def flatten_windows():
    pass

def train_elm_real():
    pass

def train_elm_synthetic(): 
    pass 



if __name__ == "__main__": 
# train_elm_real() train_elm_synthetic()
    
    