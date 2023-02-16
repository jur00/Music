from joblib import load, dump
import re
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def get_latest_version(music_dataset, latest=True):
    idx = latest-2
    file_dir = f'.\\create{music_dataset.capitalize()}Dataset\\files\\'
    version_number = int(re.sub(r'[^0-9]', '', [file for file in
                               os.listdir(file_dir)
                               if
                               file.startswith(f'{music_dataset}_music_model_')][idx]))

    return version_number


def load_data(o_n_l):
    music_datasets = ['my', 'random']
    versions = {'random': get_latest_version('random'),
                'my': {'old': get_latest_version('my', False),
                       'new': get_latest_version('my', True)}}
    random_data = load(f'.\\createRandomDataset\\files\\random_music_model_{versions["random"]}.sav')
    my_data = {o_n: load(f'.\\createMyDataset\\files\\my_music_model_{versions["my"][o_n]}.sav') for o_n in o_n_l}
    data = {o_n: my_data[o_n] + random_data for o_n in o_n_l}

    return data


def get_feature_categories(filename):
    feature_categories = load(filename)

    return feature_categories


def get_model_data_features(feature_categories):
    model_data_features = ['File Name']
    model_data_features.extend(feature_categories['librosa'])
    model_data_features.extend(feature_categories['chord'])
    model_data_features.extend(['key', 'mode', 'tempo'])

    return model_data_features


feature_categories_filename = '.\\createMyDataset\\files\\feature_categories_my.sav'

o_n_l = ['old', 'new']
data = load_data(o_n_l)
feature_categories = get_feature_categories(feature_categories_filename)
model_data_features = get_model_data_features(feature_categories)

minmaxs = {o_n: {'feature': [],
                 'min': [],
                 'max': []} for o_n in o_n_l}
for o_n in o_n_l:
    for mdf in model_data_features:
        minmaxs[o_n]['feature'].append(mdf)
        minmaxs[o_n]['min'].append(min([data[o_n][i][mdf] for i in range(len(data[o_n]))]))
        minmaxs[o_n]['max'].append(max([data[o_n][i][mdf] for i in range(len(data[o_n]))]))

lower_min = any(np.array(minmaxs['new']['min']) < np.array(minmaxs['old']['min']))
greater_max = any(np.array(minmaxs['new']['max']) > np.array(minmaxs['old']['max']))

if lower_min | greater_max:
    df = pd.DataFrame(data['new'])
    id = 'File Name'
    targets = ['danceability', 'energy', 'valence']
    sp_targets = ['sp_' + t for t in targets]
    predictors = [f for f in df.columns if f not in sp_targets + [id]]
    X = df[predictors]
    X = pd.get_dummies(data=X, columns=['key'])
    scaler = StandardScaler()
    scaler.fit(X)
    dump(scaler, './createMyDataset/files/standard_scaler_8.sav')