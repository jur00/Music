import pandas as pd
import numpy as np
import pickle
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from os import cpu_count
from ml_razor import Razor


import lightgbm as lgb

READ_BORUTA_FILE = 'BorutaFeatures.sav'
READ_MODEL_DATA_FILE = 'model_data_random50000.sav'
FILE_PATH_BORUTA = '.\\LGBMmodel\\files\\'
FILE_PATH_MODEL_DATA = '.\\createRandomDataset\\files\\'

targets = ['danceability', 'energy', 'valence']
for t in targets:
    with open(FILE_PATH_MODEL_DATA + READ_MODEL_DATA_FILE, 'rb') as file:
        df = pickle.load(file)
    with open(FILE_PATH_BORUTA + t + READ_BORUTA_FILE, 'rb') as file:
        dimp = pickle.load(file)

    feature_importances = {k: v for k, v in zip(dimp['feature'].to_list(), dimp['importance'].to_list())}

    estimator = lgb.LGBMRegressor(max_depth=5, n_jobs=cpu_count() - 1)

    razor = Razor(estimator=estimator, method='correlation', step=.05)
    razor.shave(df=df, target=t, feature_importances=feature_importances)
    razor.plot(plot_type='ks_analysis')
    correlation_features = razor.features_left
    correlation_feature_importances = {k: v for k, v in zip(dimp['feature'].to_list(), dimp['importance'].to_list())
                                       if k in correlation_features}

    razor = Razor(estimator=estimator, method='importance', lower_bound=2)
    razor.shave(df=df, target=t, feature_importances=correlation_feature_importances)
    razor.plot(plot_type='ks_analysis')
    final_features = razor.features_left

    pickle.dump(final_features, open('featureSelectionResults' + t + '.sav', 'wb'))

