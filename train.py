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
# from hypopt import GridSearch 
from evaluation import create_evaluation_dict
from scipy.sparse import vstack
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline

def get_TF_vectorizer(train_data, val_data, test_data):
    feature_extractor = TfidfVectorizer(
                                        max_features=None,
                                        stop_words='english',
                                        analyzer='word',
                                        ngram_range=(1,3),
                                        decode_error='ignore',
                                        lowercase=True,
                                        max_df=0.9,
                                        min_df=0.1)

    feature_extractor.fit(train_data)
    feature_names = feature_extractor.get_feature_names_out()
    print(feature_names[:10])
    train_x_raw = feature_extractor.transform(train_data)
    val_x_raw = feature_extractor.transform(val_data)

    whole_train_data = np.concatenate([train_data, val_data], axis=0)
    feature_extractor.fit(whole_train_data)
    feature_names = feature_extractor.get_feature_names_out()
    whole_train_raw = feature_extractor.transform(whole_train_data)
    test_x_raw = feature_extractor.transform(test_data)

    return train_x_raw, val_x_raw, whole_train_raw, test_x_raw, feature_names


def model_training(train_data, train_y, val_data, val_y,
                   test_data, model_dict, f_mode):
    
    model = model_dict["model"]

    whole_train_y = np.concatenate([train_y, val_y], axis=0)
    whole_train_data = np.concatenate([train_data, val_data], axis=0)
    split_index = [-1]*np.array(train_data).shape[0] + [0]*np.array(val_data).shape[0]
    pds = PredefinedSplit(test_fold=split_index)

    if f_mode == "no_reduction":
        pipe  = Pipeline(steps=[("tf_idf",  TfidfVectorizer(
                                        max_features=None,
                                        stop_words='english',
                                        analyzer='word',
                                        ngram_range=(1,3),
                                        decode_error='ignore',
                                        lowercase=True,
                                        max_df=0.9,
                                        min_df=0.1)),
                            ("reg", model)
                            ])

        search = GridSearchCV(pipe,
                              cv = pds,
                              param_grid= model_dict["param_grid"],
                              scoring="neg_mean_squared_error",
                              verbose=3)
        
        search.fit(whole_train_data, whole_train_y)
        

        pred_y = search.predict(test_data)
        optimal_k = f_mode
    
    else:
        mse_score_list = []
        k_list = np.arange(100,1100,100)
        train_x_raw, val_x_raw, whole_train_raw, test_x_raw, feature_names = get_TF_vectorizer(train_data, 
                                                                                             val_data,
                                                                                                   test_data)
        
        for k in k_list:
            train_x, val_x = initialize_feature_selection(k, train_x_raw, train_y, val_x_raw)
            model.fit(train_x, train_y)
            val_pred = model.predict(val_x)
            mse_score = mean_squared_error(val_y, val_pred)
            mse_score_list.append(mse_score)
            print(mse_score)
            
        optimal_k = k_list[np.argmin(mse_score_list)]
        whole_train_x, test_x = initialize_feature_selection(optimal_k, whole_train_raw, whole_train_y, test_x_raw)

        
        pipe = Pipeline(steps=[("reg", model)])

        search = GridSearchCV(pipe,
                              cv = pds,
                              param_grid= model_dict["param_grid"],
                              scoring="neg_mean_squared_error",
                              verbose=3)
        
        search.fit(whole_train_x, whole_train_y)
        pred_y = search.predict(test_x)
        # model.fit(whole_train_x, whole_train_y)
        # pred_y = model.predict(test_x)
            
    return pred_y, optimal_k


if __name__ == "__main__":
    pass