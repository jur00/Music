from joblib import dump, load
import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from autokeras import StructuredDataRegressor
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, LayerNormalization
import tensorflow as tf
import time

def get_test_files():
    file_dir = '.\\createMyDataset\\files\\'
    version_number = int(re.sub(r'[^0-9]', '', [file for file in
                                                os.listdir(file_dir)
                                                if
                                                file.startswith('my_music_model_')][-1]))
    data = load(file_dir + f'my_music_model_{version_number}.sav')
    df = pd.DataFrame(data)
    test_filenames = df['File Name'].sample(frac=.1, random_state=8).to_list()

    return test_filenames


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
    df_test = df.loc[df['File Name'].isin(test_files), :]
    df = df.loc[~df['File Name'].isin(test_files), :]
    id = 'File Name'
    targets = ['danceability', 'energy', 'valence']
    sp_targets = ['sp_' + t for t in targets]

    return df, df_test, id, targets, sp_targets


music_datasets = ['my', 'random', 'complete']
results = {'dataset': [],
           'target': [],
           'time_taken': [],
           'r2': [],
           'mae': []}
counter = 1
test_files = get_test_files()
df, df_test, id, targets, sp_targets = load_data('complete')

for target in targets:
    sp_target = 'sp_' + target
    dimp = load(f'.\\LGBMmodel\\files\\borutaFeatureImportances{music_dataset.capitalize()}'
                f'{target.capitalize()}')
    predictors = dimp['feature'].tolist()
    X = df[predictors]
    y = df[sp_target].values.reshape(-1, 1)

    print(f'{counter} / 18, features: {music_dataset}, target: {target}')
    counter += 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    y_train = y.copy()
    X_test = scaler.transform(df_test[predictors])
    y_test = df_test[sp_target].values.reshape(-1, 1)
    t1 = time.time()
    reg = StructuredDataRegressor(max_trials=15, overwrite=True, directory=f'.\\NNmodel\\files\\models\\{target}')
    reg.fit(x=X_train, y=y_train, verbose=0)
    y_hat = reg.predict(X_test, verbose=0)
    r_squared = r2_score(y_test, y_hat)
    mae = mean_absolute_error(y_test, y_hat)
    t2 = time.time()
    timedelta = t2-t1
    minutes = int(timedelta / 60)
    seconds = int(timedelta % 60)
    results['dataset'].append(dataset)
    results['target'].append(target)
    results['time_taken'].append(f'{minutes} minutes and {seconds} seconds')
    results['r2'].append(r_squared)
    results['mae'].append(mae)
    print(f'data: {dataset}, target: {target}, mae: {mae}, r2: {r_squared}, time: {minutes} minutes and {seconds} seconds')
dfr = pd.DataFrame(results)
dump(dfr, '.\\NNmodel\\files\\featureComparisonAllResults.sav')