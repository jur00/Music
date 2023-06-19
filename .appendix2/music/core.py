# code for __init__.Music.get_more_training_data

import re

import pandas as pd
import numpy as np

import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_validate

from music.database import MySQLdb
from music.versioning import UpdateArtistsSpotify
from music.other import get_credentials

def find_best_parameters(X, y):
    study = optuna.create_study(direction='maximize')

    def objective(trial, X, y):
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
        lgbm = lgb.LGBMClassifier(**params)
        scores = cross_validate(lgbm, X, y, cv=5, scoring='accuracy')['test_score']
        accuracy = np.mean(scores)

        return accuracy

    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=50)

    return study.best_params

db = MySQLdb(get_credentials('db'))
df = db.load_table('tracks_training')
df_sp = db.load_table('tracks_my_spotify', only_current_tracks=True)
df_rb = db.load_table('tracks_my_rekordbox', only_current_tracks=True)
df['simplified_trackname'] = df['trackname'].str.lower().str.replace(
    r'[^a-z 0-9]', '', regex=True).str.replace(r' +', ' ', regex=True)
df = df.drop_duplicates(subset=['simplified_trackname'])

df_sp['simplified_trackname'] = df_sp['sp_trackname'].str.lower().str.replace(
    r'[^a-z 0-9]', '', regex=True).str.replace(r' +', ' ', regex=True)
df = df.loc[~df['simplified_trackname'].isin(df_sp['simplified_trackname'])]

# ----------------------------------------------------------------------------------------------- #

best_params = {}
features = ['danceability', 'energy', 'valence', 'instrumentalness', 'duration', 'tempo']
df_sp_conc = df_sp.loc[df_sp['sp_dif_type'] == 'same', :].rename(
    columns={f'sp_{feature}': feature for feature in features})
df = df.rename(columns={'track_id': 'sp_id'})
df_sp_conc['my_track'] = 1
df['my_track'] = 0
keep_cols = ['sp_id', 'my_track'] + features

predicted_probas_list = []
n_my_tracks = df_sp_conc.shape[0]
n_train_tracks = df.shape[0]
df['n_in_model'] = np.zeros(n_train_tracks)
df_predicted = df.loc[:, ['sp_id']]
i = 0
while any(df['n_in_model'] == 0):
    perc_part_of_model = (df['n_in_model'] > 0).mean()
    print(f'iteration: {i}', 'percentage of training data been part of '
          f'model: {round(perc_part_of_model*100, 4)}%', sep=', ', end='\r')
    weights = 1 / (df['n_in_model'] + 1)
    df_train_in_model = df[keep_cols].sample(n=n_my_tracks, random_state=i, weights=weights)
    df_train_in_model_ids = df_train_in_model['sp_id']
    df.loc[df['sp_id'].isin(df_train_in_model_ids), 'n_in_model'] += 1
    df_model = pd.concat([df_train_in_model,
                          df_sp_conc[keep_cols]], ignore_index=True)
    df_model = df_model.drop('sp_id', axis=1)
    X = df_model.drop('my_track', axis=1)
    y = df_model['my_track']
    if (i == 0) & (len(best_params) == 0):
        best_params = find_best_parameters(X=X, y=y)

    model = lgb.LGBMClassifier(**best_params).fit(X, y)
    df_predict = df.loc[~df['sp_id'].isin(df_train_in_model_ids), keep_cols].drop(
            ['sp_id', 'my_track'], axis=1)
    my_track_proba_idx = np.where(model.classes_ == 1)[0][0]

    predicted_probas_list.append(pd.Series(
        model.predict_proba(df_predict).T[my_track_proba_idx].T,
        index=df_predict.index, name=f'iteration_{i}'))
    i += 1

df_predicted['avg_proba'] = pd.concat(predicted_probas_list, axis=1).mean(skipna=True, axis=1).sort_index()
df_predicted = df_predicted.sort_values(by='avg_proba', ascending=False).reset_index(drop=True)
# artists_spotify = UpdateArtistsSpotify(db, sp)
# artists_spotify.update()
