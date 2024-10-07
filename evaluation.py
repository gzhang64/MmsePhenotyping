from itertools import chain
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import math
import os
import pandas as pd
import pickle
import string
import numpy as np


def create_evaluation_dict(test_y, pred_y, optimal_k, model_name):
    evaluation_dict = {"MSE": [mean_squared_error(test_y, pred_y)],
                       "RMSE": [np.sqrt(mean_squared_error(test_y, pred_y))],
                       "optimal_k": [optimal_k],
                       "model": [model_name]
                       }
    evaluation_df = pd.DataFrame(evaluation_dict)
    
    return evaluation_df


def create_evaluation_dict_meta(test_y, pred_y, model_name):
    evaluation_dict = {"MSE": [mean_squared_error(test_y, pred_y)],
                       "RMSE": [np.sqrt(mean_squared_error(test_y, pred_y))],
                       "model": [model_name]
                       }
    evaluation_df = pd.DataFrame(evaluation_dict)
    
    return evaluation_df


if __name__=="__main__":
    pass