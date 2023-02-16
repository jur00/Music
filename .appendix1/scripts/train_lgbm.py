import pandas as pd
import numpy as np
from joblib import dump, load
import os
import re
import random
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(music_dataset):
    music_datasets = ['my', 'random']
    if music_dataset in music_datasets:
        file_dir = f'.\\create{music_dataset.capitalize()}Dataset\\files\\'
        version_number = int(re.sub(r'[^0-9]', '', [file for file in
                                                    os.listdir(file_dir)
                                                    if
                                                    file.startswith(f'{music_dataset}_music_model_')][-1]))
        data = load(file_dir + f'{music_dataset}_music_model_{version_number}.sav')
    else:  # if music_dataset == 'complete':
        file_dirs = {md: f'.\\create{md.capitalize()}Dataset\\files\\' for md in music_datasets}
        version_numbers = {md: int(re.sub(r'[^0-9]', '', [file for file in
                                                          os.listdir(fd)
                                                          if
                                                          file.startswith(f'{md}_music_model_')][-1]))
                           for md, fd in zip(music_datasets, file_dirs.values())}
        datas = {md: load(file_dirs[md] + f'{md}_music_model_{version_numbers[md]}.sav') for md in music_datasets}
        data = datas['my'] + datas['random']
    df = pd.DataFrame(data)
    df = pd.get_dummies(data=df, columns=['key'])
    id = 'File Name'
    targets = ['danceability', 'energy', 'valence']
    sp_targets = ['sp_' + t for t in targets]
    predictors = [f for f in df.columns if f not in sp_targets + [id]]

    return df, predictors, targets, sp_targets


def create_my_tracks_split(k=10, random_state=8):
    file_dir = '.\\createMyDataset\\files\\'
    version_number = int(re.sub(r'[^0-9]', '', [file for file in
                                                os.listdir(file_dir)
                                                if
                                                file.startswith('my_music_model_')][-1]))
    data = load(file_dir + f'my_music_model_{version_number}.sav')
    my_tracks = pd.DataFrame(data)['File Name'].to_list()
    np.random.seed(random_state)
    np.random.shuffle(my_tracks)
    my_tracks_split = np.array_split(my_tracks, k)

    return my_tracks_split


def my_cross_validation(estimator, X_scaled, y, my_tracks_split):
    splits = len(my_tracks_split)
    scores = []
    for i in range(splits):
        test_idxs = np.where(df['File Name'].isin(my_tracks_split[0]).to_list())[0]
        X_train = np.array([X_scaled[i] for i in range(len(X_scaled)) if i not in test_idxs])
        X_test = np.array([X_scaled[i] for i in range(len(X_scaled)) if i in test_idxs])
        y_train = np.array([y[i] for i in range(len(y)) if i not in test_idxs])
        y_test = np.array([y[i] for i in range(len(y)) if i in test_idxs])

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

        return scores


music_dataset = 'complete'

df, predictors, targets, sp_targets = load_data(music_dataset)

my_tracks_split = create_my_tracks_split()

for target, sp_target in zip(targets, sp_targets):
    X = df[predictors]
    y = df[sp_target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # optuna.logging.disable_default_handler()
    study = optuna.create_study(direction='minimize')


    def objective(trial, X_scaled, y, my_tracks_split):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 16, 4096, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 100),
            'num_leaves': trial.suggest_int('num_leaves', 3, 100),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'reg_alpha': trial.suggest_float('reg_alpha', 1, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 2),
            'subsample': trial.suggest_float('subsample', .5, .9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .05, .9)
        }
        lgbm = lgb.LGBMRegressor(**params)
        scores = my_cross_validation(lgbm, X_scaled, y, my_tracks_split)
        rmse = np.mean(scores)

        return rmse


    func = lambda trial: objective(trial, X_scaled, y, my_tracks_split)
    study.optimize(func, n_trials=50)
    fig = optuna.visualization.plot_optimization_history(study, target_name='rmse', error_bar=True)

    best_params = study.best_params


    lgbm = lgb.LGBMRegressor(**best_params)
    lgbm.fit(X, y)

    dump(lgbm, f'.\\LGBMmodel\\files\\LGBMmodel{target.capitalize()}.sav')
