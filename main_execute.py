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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from loading_cleaning import get_dataset
from feature_selection import initialize_feature_selection
from evaluation import create_evaluation_dict
from train import model_training, get_TF_vectorizer


def main():
    train_data, train_y, val_data, val_y, test_data, test_y = get_dataset()
    feature_selection_mode = ["feature_reduction", "no_reduction"]
    model_list = [{"model_name": "RandomForestRegressor",
                "model": RandomForestRegressor(),
                "param_grid": {"reg__n_estimators": [100,200,300,500,600],
                                "reg__max_depth": [None, 10, 20, 30]}
                },

                {"model_name": "MLP",
                    "model": MLPRegressor(max_iter=5000),
                    "param_grid": {"reg__solver": ['lbfgs', 'adam'],
                                   "reg__hidden_layer_sizes": [(512, 256, 125, 50),
                                                           (100, 50, 25),
                                                           (512, 256, 125, 50, 25)],
                                "reg__learning_rate": ['constant', 'adaptive'],
                                "reg__alpha": [0.0001, 0.001, 0.01, 0.1]}
                },

                    {"model_name": "SGDRegressor",
                    "model": SGDRegressor(),
                    "param_grid": {'reg__penalty': ['l1', 'l2', 'elasticnet'],
                                'reg__alpha': [0.0001, 0.001, 0.01, 0.1]}
                }
                ]
    whole_evaluation = pd.DataFrame()
    for f_mode in feature_selection_mode:
        for model_dict in model_list:
            print(f"{f_mode}, {model_dict['model_name']}")
            model_name = model_dict["model_name"]
            pred_y, optimal_k = model_training(train_data, train_y, val_data, val_y, 
                                               test_data, model_dict, f_mode)
            
            evaluation_df = create_evaluation_dict(test_y, pred_y, optimal_k, model_name)
            whole_evaluation =pd.concat([whole_evaluation, evaluation_df], axis=0)
            whole_evaluation.to_csv("performance_whole_v2.csv", index=False)
            print("Next")

if __name__=="__main__":
    main()

# 17562


