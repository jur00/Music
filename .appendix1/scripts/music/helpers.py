import os
import re
import time
from datetime import datetime
from contextlib import contextmanager
from sklearn.linear_model import LassoCV

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from joblib import load

@contextmanager
def set_dir(path):
    origin = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def get_latest_version(file_kind='music', music_dataset='my', dataset_type='model', latest=True):
    idx = latest - 2
    if file_kind == 'music':
        file_str = f'music_{music_dataset}_{dataset_type}_'
    else:
        file_str = file_kind + '_'

    version_number = int(re.sub(r'[^0-9]', '', [file for file in
                                                os.listdir()
                                                if
                                                file.startswith(file_str)][idx]))

    return version_number

def jaccard_similarity(test, real):
    intersection = set(test).intersection(set(real))
    union = set(test).union(set(real))
    return len(intersection) / len(union)

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def replace_keys(value, kind='tonal_to_camelot'):
    # kind in ['tonal_to_camelot', 'camelot_to_pitch_mode']
    tonal_camelot = {'A': '11B',
                     'Am': '8A',
                     'A#': '6B',
                     'Bb': '6B',
                     'A#m': '3A',
                     'Bbm': '3A',
                     'B': '1B',
                     'Bm': '10A',
                     'C': '8B',
                     'Cm': '5A',
                     'C#': '3B',
                     'Db': '3B',
                     'C#m': '12A',
                     'Dbm': '12A',
                     'D': '10B',
                     'Dm': '7A',
                     'D#': '5B',
                     'Eb': '5B',
                     'D#m': '2A',
                     'Ebm': '2A',
                     'E': '12B',
                     'Em': '9A',
                     'F': '7B',
                     'Fm': '4A',
                     'F#': '2B',
                     'Gb': '2B',
                     'F#m': '11A',
                     'G': '9B',
                     'Gm': '6A',
                     'G#': '4B',
                     'Ab': '4B',
                     'G#m': '1A',
                     'Abm': '1A'}
    tonal_open = {'A': '4d',
                  'Am': '1m',
                  'A#': '11d',
                  'Bb': '11d',
                  'A#m': '8m',
                  'Bbm': '8m',
                  'B': '6d',
                  'Bm': '3m',
                  'C': '1d',
                  'Cm': '10m',
                  'C#': '8d',
                  'Db': '8d',
                  'C#m': '5m',
                  'Dbm': '5m',
                  'D': '3d',
                  'Dm': '12m',
                  'D#': '10d',
                  'Eb': '10d',
                  'D#m': '7m',
                  'Ebm': '7m',
                  'E': '5d',
                  'Em': '2m',
                  'F': '12d',
                  'Fm': '9m',
                  'F#': '7d',
                  'Gb': '7d',
                  'F#m': '4m',
                  'G': '2d',
                  'Gm': '11m',
                  'G#': '9d',
                  'Ab': '9d',
                  'G#m': '6m',
                  'Abm': '6m'}
    pitch_mode_camelot = {(0, 1): '8B',
                          (1, 1): '3B',
                          (2, 1): '10B',
                          (3, 1): '5B',
                          (4, 1): '12B',
                          (5, 1): '7B',
                          (6, 1): '2B',
                          (7, 1): '9B',
                          (8, 1): '4B',
                          (9, 1): '11B',
                          (10, 1): '6B',
                          (11, 1): '1B',
                          (0, 0): '5A',
                          (1, 0): '12A',
                          (2, 0): '7A',
                          (3, 0): '2A',
                          (4, 0): '9A',
                          (5, 0): '4A',
                          (6, 0): '11A',
                          (7, 0): '6A',
                          (8, 0): '1A',
                          (9, 0): '8A',
                          (10, 0): '3A',
                          (11, 0): '10A'}

    if kind == 'tonal_to_camelot':
        if value in list(tonal_camelot.keys()):
            value = [v for k, v in tonal_camelot.items() if value == k][0]

        return value
    elif kind == 'camelot_to_pitch_mode':
        pitch, mode = [k for k, v in pitch_mode_camelot.items() if value == v][0]
        return pitch, mode

def prepare_for_modeling(data):
    df = pd.DataFrame(data)
    id = 'File Name'
    targets = ['danceability', 'energy', 'valence']
    sp_targets = ['sp_' + t for t in targets]
    predictors = [f for f in df.columns if f not in sp_targets + [id]]
    X = df[predictors]

    return df, id, targets, sp_targets, predictors, X

def create_df_importances(columns, importances):
    dimp = pd.DataFrame(np.array([columns, importances]).T, columns=['feature', 'importance'])
    dimp = dimp.sort_values(by='importance', ascending=False).reset_index(drop=True)

    if any(dimp['feature'].str.contains('key')):
        add_keys = ['key_' + str(i) for i in range(11) if 'key_' + str(i) not in dimp['feature'].to_list()]
        add_importances = [min(dimp['importance']) - np.abs(min(dimp['importance']) / 10) for _ in range(len(add_keys))]
        df_add = pd.DataFrame({'feature': add_keys, 'importance': add_importances})
        dimp = pd.concat([dimp, df_add], axis=0)

    return dimp

def create_test_set(test_set_type='my', k=10, random_state=8):
    # test_set_type = 'my', 'random', 'full'

    version = get_latest_version(music_dataset=test_set_type)
    test_tracks = [t['File Name'] for t in load(f'music_{test_set_type}_model_{version}.sav')]
    np.random.seed(random_state)
    np.random.shuffle(test_tracks)

    return test_tracks[:int((len(test_tracks) / k))]


class Progress:

    def __init__(self):
        self.start = time.time()

    def show(self, loop_space, current_loop):
        n = len(loop_space)
        counter = loop_space.index(current_loop) + 1
        fraction_done = counter / n

        progress_precentage = str(round(fraction_done * 100, 2)) + "%"
        now = time.time()
        eta = self.start + ((now - self.start) / fraction_done)
        time_completed = datetime.fromtimestamp(eta).strftime("%Y-%m-%d %H:%M:%S")

        average_compute_time = (now - self.start) / counter
        minutes = int(average_compute_time / 60)
        seconds = int(average_compute_time % 60)
        print('Done: {} / {} ({}), ETA: {}, Average time per loop: {} minutes and {} seconds'.
              format(counter, n, progress_precentage, time_completed, minutes, seconds),
              end='\r')

class LogitLassoCV(LassoCV):

    def fit(self, x, p):
        p = np.asarray(p)
        p[p <= 0] = 1e-7
        p[p >= 1] = 1 - 1e-7
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)


class LogitLinearSklearn(LinearRegression):

    def fit(self, x, p):
        p = np.asarray(p)
        p[p == 0] = 1e-7
        p[p >= 1] = 1 - 1e-7
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)

