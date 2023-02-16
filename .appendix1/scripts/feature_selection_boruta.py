import pandas as pd
import numpy as np
from joblib import dump, load
import os
import re
import lightgbm as lgb
from BorutaShap import BorutaShap


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


def fit_boruta_shap(Xdf, y):
    regressor = lgb.LGBMRegressor(n_jobs=3, max_depth=5)
    shap_selector = BorutaShap(model=regressor,
                               importance_measure='shap',
                               classification=False)
    shap_selector.fit(Xdf, y, random_state=8, n_trials=100, verbose=True)
    shap_selector.TentativeRoughFix()
    accepted = shap_selector.columns
    fimp = shap_selector.feature_importance(normalize=True)[0]

    dimp = pd.DataFrame(np.array([accepted, fimp]).T, columns=['feature', 'importance'])
    dimp = dimp.sort_values(by='importance', ascending=False).reset_index(drop=True)
    if any(dimp['feature'].str.contains('key')):
        add_keys = ['key_' + str(i) for i in range(11) if 'key_' + str(i) not in dimp['feature'].to_list()]
        add_importances = [min(dimp['importance']) for _ in range(len(add_keys))]
        df_add = pd.DataFrame({'feature': add_keys, 'importance': add_importances})
        dimp = pd.concat([dimp, df_add], axis=0)

    return dimp


def boruta_feature_selection(music_dataset):
    df, predictors, targets, sp_targets = load_data(music_dataset)
    Xdf = df[predictors]

    for target, sp_target in zip(targets, sp_targets):
        print(target)
        y = df[[sp_target]].values.ravel()
        dimp = fit_boruta_shap(Xdf, y)
        dump(dimp, f'.\\LGBMmodel\\files\\borutaFeatureImportances{music_dataset.capitalize()}'
                   f'{target.capitalize()}.sav')


music_dataset = 'all'  # 'my', 'random', 'complete' or 'all'

if music_dataset == 'all':
    for md in ['my', 'random', 'complete']:
        print(md)
        boruta_feature_selection(md)
else:
    boruta_feature_selection(music_dataset)

