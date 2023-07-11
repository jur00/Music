from itertools import chain
import pandas as pd
from joblib import load, dump
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pytube import extract
from unidecode import unidecode
from datetime import datetime
import re
import time
import requests
import librosa
from spafe.features.gfcc import gfcc
from entropy_estimators import continuous
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis, skew, beta
import subprocess
import shutil
import warnings
import os
from contextlib import contextmanager
import stat
import json

from update_dataset.helpers import (jaccard_similarity, neutralize, overrule_connection_errors,
                                    create_name_string, check_original, add_or_remove_original,
                                    set_dir, levenshtein_distance, find)


class Credentials:

    def __init__(self, directory, filename, api):
        self.file_location = f'{directory}\\{filename}' if directory != '' else filename
        self.api = api

        self.credentials = None

    def get(self):
        with open(self.file_location, 'rb') as f:
            self.credentials = json.load(f)[self.api]


class MyMusic:

    def __init__(self, directory, filename):
        self.file_location = f'{directory}\\{filename}' if directory != '' else filename

        self.data = None

    def get(self):
        self._load()

    def _load(self):
        self.data = load(self.file_location)


class RekordboxMusic:

    def __init__(self,
                 directory,
                 filename):

        self.file_location = f'{directory}\\{filename}' if directory != '' else filename

        self.sp = None
        self.driver = None
        self.youtube = None
        self.df = None
        self.data = None

    def get(self):
        self._load_txt()
        self._remove_duplicated_filenames()
        self._change_column_name()
        self._fill_missing()
        self._add_genre_bool_columns()
        self._quantize_rating()
        self._quantize_duration()
        self._quantize_bpm()
        self._categorize_origin_type()
        self._drop_rownumber_column()
        self._to_records()

    def _load_txt(self):
        self.df = pd.read_csv(self.file_location,
                              sep=None, header=0,
                              encoding='utf-16',
                              engine='python')

    def _remove_duplicated_filenames(self):
        self.df = self.df.loc[~self.df.duplicated(subset='File Name'), :]
        self.df = self.df.reset_index(drop=True)

    def _change_column_name(self):
        self.df = self.df.rename(columns={'#': 'row'})

    def _fill_missing(self):
        self.df['row'] = list(range(1, (self.df.shape[0] + 1)))
        self.df['Composer'] = self.df['Composer'].fillna('')
        self.df['Album'] = self.df['Album'].fillna('')
        self.df['Label'] = self.df['Label'].fillna('')

    def _add_genre_bool_columns(self):
        genres = ['Afro Disco', 'Balearic', 'Cosmic', 'Disco', 'Italo Disco', 'Nu Disco',
                  'Acid House', 'Deep House', 'House', 'Indie', 'Techno', 'Nostalgia', 'Old Deep House',
                  'Offbeat']
        self.df['Comments'] = self.df['Comments']. \
            str.lstrip(' /* '). \
            str.rstrip(' */'). \
            str.split(' / ')
        for g in genres:
            m = []
            for i in range(len(self.df['Comments'])):
                m.append(g in self.df['Comments'][i])

            self.df[g.replace(' ', '_')] = m

        del self.df['Comments']

    def _quantize_rating(self):
        self.df['Rating'] = self.df['Rating'].str.rstrip().str.len()

    def _quantize_duration(self):
        minutes = self.df['Time'].str.rpartition(":")[0].astype(int) * 60
        seconds = self.df['Time'].str.rpartition(":")[2].astype(int)
        self.df['rb_duration'] = (minutes + seconds) * 1000

    def _quantize_bpm(self):
        if self.df.dtypes['BPM'] == str:
            self.df['BPM'] = self.df['BPM'].str.replace(',', '.').astype(float)

    def _categorize_origin_type(self):
        self.df['track_kind'] = 'original'
        self.df.loc[~self.df['Label'].isin(['', ' ']), 'track_kind'] = 'remix'
        self.df.loc[~self.df['Album'].str.lower().isin(
            ['', ' ', 'original', 'original mix']
        ), 'track_kind'] = 'version'

    def _drop_rownumber_column(self):
        self.df.drop('row', axis=1, inplace=True)

    def _to_records(self):
        self.data = self.df.to_dict('records')


class Disjoint:

    def __init__(self, data1,
                 data2, datatype1='dict', datatype2='dict',
                 feature='File Name'):
        self.data1 = data1
        self.data2 = data2
        self.datatype1 = datatype1
        self.datatype2 = datatype2
        self.feature = feature

    def not_in_data1(self):
        return list(self._check_filenames(self.data2, self.datatype2) -
                    self._check_filenames(self.data1, self.datatype1))

    def not_in_data2(self):
        return list(self._check_filenames(self.data1, self.datatype1) -
                    self._check_filenames(self.data2, self.datatype2))

    def get_indexes(self, type='not_in_data2'):
        if type == 'not_in_data2':
            names = self.not_in_data2()
            return list([find(self.data1, 'File Name', fn) for fn in names])
        else:
            names = self.not_in_data1()
            return list([find(self.data2, 'File Name', fn) for fn in names])

    def _check_filenames(self, data, datatype):
        if datatype == 'dict':
            filenames_data = set([d[self.feature] for d in data])
        elif datatype == 'list':
            filenames_data = set(data)

        return filenames_data


class SpotifyConnect:

    def __init__(self, credentials):
        self.sp = self.__make_spotify(credentials)

    @staticmethod
    def __make_spotify(credentials):
        cid = credentials['cid']
        secret = credentials['secret']
        ccm = SpotifyClientCredentials(client_id=cid,
                                       client_secret=secret)
        sp = spotipy.Spotify(client_credentials_manager=ccm)

        return sp


class SpotifyFeatures(SpotifyConnect):

    def __init__(self, rb_data, credentials):

        super().__init__(credentials)

        self.rb_data = rb_data

        self.results = None
        self.i = None
        self.spotify_features = {}
        self.track_main_features = None
        self.track_audio_features = None

        self._is_original = None
        self._best = None

        self.__rb_name_set = None

    def get(self, i):
        self.get_track_main_features(i)
        self.spotify_features.update(self.track_main_features)
        self.get_track_audio_features(self.track_main_features['sp_id'])
        self.spotify_features.update(self.track_audio_features)

    def get_track_main_features(self, i):
        self.i = i

        self._search()

        if len(self.results) > 0:
            track_name_parts = ['Artist', 'Mix Name', 'Composer', 'Album', 'Label']
            compare_name = create_name_string(self.rb_data, self.i, track_name_parts)
            similarities = self.__get_similarity_scores(compare_name)
            self._is_original = check_original(self.rb_data, self.i)
            self._identify_most_similar_spotify_name(compare_name, similarities)
            self.track_main_features = self._main_features_best_result()
        else:
            self.track_main_features = self._main_features_empty_result()

    def get_track_audio_features(self, sp_id):
        features_list = ['id', 'danceability', 'energy', 'valence', 'instrumentalness',
                         'speechiness', 'acousticness', 'loudness', 'key', 'mode']
        if len(self.track_main_features['sp_id']) > 0:
            audio_features = overrule_connection_errors(self.sp.audio_features(sp_id))
            self.track_audio_features = {f'sp_{feature}': audio_features[0][f'{feature}'] for feature in features_list}
        else:
            self.track_audio_features = {f'sp_{feature}': 0 for feature in features_list}

    def _search(self):
        query_name_parts = ['Artist', 'Mix Name']
        artist_track = create_name_string(self.rb_data, self.i, query_name_parts)
        self.results = overrule_connection_errors(self.sp.search(q=artist_track, type="track", limit=50))['tracks'][
            'items']

    def _identify_most_similar_spotify_name(self, compare_name, similarities):
        if self._is_original:
            self.__compare_names_original_mix(compare_name, similarities)
        else:
            self.__compare_name_remix_like(compare_name, similarities)

    def _main_features_best_result(self):
        sp_results = self.results[self._best]
        sp_name_set = neutralize(f"{sp_results['name']} {sp_results['artists'][0]['name']}").split()

        main_features = {'sp_id': sp_results['id'],
                         'sp_artist': sp_results['artists'][0]['name'],
                         'sp_trackname': sp_results['name'],
                         'sp_duration': sp_results['duration_ms'],
                         'sp_popularity': sp_results['popularity'],
                         'sp_rb_name_dif': list(set(sp_name_set) - set(self.__rb_name_set)),
                         'rb_sp_name_dif': list(set(self.__rb_name_set) - set(sp_name_set))}

        return main_features

    @staticmethod
    def _main_features_empty_result():
        return {'sp_id': '',
                'sp_artist': '',
                'sp_trackname': '',
                'sp_duration': 0,
                'sp_popularity': 0,
                'sp_rb_name_dif': [],
                'rb_sp_name_dif': []}

    def __compare_names_original_mix(self, compare_name, similarities):
        compare_name_original = add_or_remove_original(compare_name)
        similarities_original = self.__get_similarity_scores(compare_name_original)
        similarities_both = np.array([similarities, similarities_original])
        best_idxs = np.unravel_index(similarities_both.argmax(), similarities_both.shape)

        self._best = best_idxs[1]
        self.__rb_name_set = [compare_name, compare_name_original][best_idxs[0]].split()

    def __compare_name_remix_like(self, compare_name, similarities):
        self._best = np.argmax(similarities)
        self.__rb_name_set = compare_name.split()

    def __get_similarity_scores(self, comp_name):
        similarities = []
        for t in self.results:
            sp_name = neutralize(f"{t['name']} {t['artists'][0]['name']}")
            similarities.append(jaccard_similarity(comp_name, sp_name))

        return similarities


class YoutubeConnect:

    def __init__(self, credentials):
        self.youtube, self.driver = self.__make_youtube(credentials)

    @staticmethod
    def __make_youtube(credentials):
        youtube = build('youtube', 'v3', developerKey=credentials['api_key'])
        browser_option = webdriver.ChromeOptions()
        browser_option.add_argument('--no-sandbox')
        browser_option.add_argument('--headless')
        browser_option.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_option)

        return youtube, driver


class YoutubeFeatures(YoutubeConnect):

    def __init__(self, rb_data, credentials):
        super().__init__(credentials)

        self.rb_data = rb_data

        self.i = None
        self.youtube_features = None

    def _create_rb_name_set(self):
        full_name = create_name_string(self.rb_data, self.i, ['Artist', 'Mix Name', 'Composer', 'Album', 'Label'])
        return full_name.split()

    def _create_yt_search_link(self):
        search_query = self.rb_data[self.i]['Track Title'].replace(' ', '+').replace('&', '%26')
        return 'https://www.youtube.com/results?search_query=' + search_query

    def _input_link(self, link):
        overrule_connection_errors(self.driver.get(link))

    def _search(self):
        return overrule_connection_errors(self.driver.find_elements('xpath', '//*[@id="video-title"]'))

    @staticmethod
    def _extract_youtube_id(user_data):
        youtube_url = user_data[0].get_attribute('href')
        return extract.video_id(youtube_url)

    def _pull_features(self, yt_id):
        return overrule_connection_errors(self.youtube.videos().list(part='snippet,statistics,contentDetails',
                                                                     id=yt_id).execute())

    @staticmethod
    def _set_features(yt_result):
        yt_name = yt_result['items'][0]['snippet']['title']
        yt_category = int(yt_result['items'][0]['snippet']['categoryId'])
        yt_views = int(yt_result['items'][0]['statistics']['viewCount'])
        yt_duration_str = yt_result['items'][0]['contentDetails']['duration']
        yt_publish_date = yt_result['items'][0]['snippet']['publishedAt'].split('T')[0]

        return yt_name, yt_category, yt_views, yt_duration_str, yt_publish_date

    @staticmethod
    def _transform_yt_duration(yt_duration_str):
        if 'H' in yt_duration_str:
            yt_duration = 0
        else:
            minutes = yt_duration_str.rpartition('PT')[2].rpartition('M')[0]
            seconds = yt_duration_str.rpartition('PT')[2].rpartition('M')[2].rpartition('S')[0]
            if minutes == '':
                minutes = 0
            if seconds == '':
                seconds = 0
            yt_duration = ((int(minutes) * 60) + int(seconds)) * 1000

        return yt_duration

    @staticmethod
    def _create_yt_name_set(yt_name):
        return neutralize(yt_name).split()

    def get(self, i):
        self.i = i

        yt_name = ''
        yt_views = 0
        yt_publish_date = ''
        yt_duration = 0
        yt_category = 0
        yt_name_set = []

        rb_name_set = self._create_rb_name_set()
        link = self._create_yt_search_link()
        self._input_link(link)
        user_data = self._search()

        if len(user_data) > 0:
            if not user_data[0].get_attribute('href') is None:
                yt_id = self._extract_youtube_id(user_data)
                yt_result = self._pull_features(yt_id)
                yt_name, yt_category, yt_views, yt_duration_str, yt_publish_date = self._set_features(yt_result)
                yt_duration = self._transform_yt_duration(yt_duration_str)
                yt_name_set = self._create_yt_name_set(yt_name)

        self.youtube_features = {
            'yt_name': yt_name,
            'yt_views': yt_views,
            'yt_publish_date': yt_publish_date,
            'yt_duration': yt_duration,
            'yt_category': yt_category,
            'yt_rb_name_dif': list(set(yt_name_set) - set(rb_name_set)),
            'rb_yt_name_dif': list(set(rb_name_set) - set(yt_name_set))
        }


class WaveFeatures:

    def __init__(self, tracks_dir, rb_data):

        self.tracks_dir = tracks_dir
        self.rb_data = rb_data

        self.wave_features = None

    def _load_waveform(self, i):
        filename = self.rb_data[i]['File Name']
        track_path = self.tracks_dir + filename
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            y, sr = librosa.load(track_path, sr=44100)

        bpm = self.rb_data[i]['BPM']

        return filename, y, sr, bpm

    @staticmethod
    def _get_librosa_features(y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_lib, beats = librosa.beat.beat_track(y=y, sr=sr)
        lib_duration = librosa.get_duration(y=y, sr=sr)
        beat_strength_mean = np.mean(onset_env[beats])
        offbeat_strength = np.mean(onset_env[-beats])

        S = np.abs(librosa.stft(y))
        melbands = librosa.feature.melspectrogram(S=S, n_mels=13)
        half = round(melbands.shape[0] / 2)

        features_num = {'sample_rate': sr,
                        'beat_strength': np.max(onset_env[beats]),
                        'beat_dif': beat_strength_mean - offbeat_strength,
                        'max_loudness': np.max(y),
                        'onset_rate': len(beats) / lib_duration,
                        'hfc': np.sum(melbands[:half]) / np.sum(melbands)}

        features_1d = {'rms': librosa.feature.rms(y=y)[0],
                       'spectral_bandwith': librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
                       'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr)[0],
                       'spectral_flatness': librosa.feature.spectral_flatness(y=y)[0],
                       'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
                       'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y)[0]}

        features_2d = {'chroma': librosa.feature.chroma_stft(S=S, sr=sr),
                       'cqt': np.abs(librosa.cqt(y, sr=sr, n_bins=6)),
                       'gfcc': gfcc(sig=y).transpose(),
                       'melbands': melbands,
                       'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
                       'spectral_contrast': librosa.feature.spectral_contrast(S=S, sr=sr)}

        entropies = {i + '_entropy': continuous.get_h(j, k=8) for i, j in features_2d.items()}

        for i in features_2d.keys():
            features_no_zero = features_2d[i].copy()
            for j in range(len(features_2d[i])):
                unique_vals = np.unique(np.abs(features_2d[i][j]))
                for k in range(len(features_2d[i][j])):
                    if features_2d[i][j][k] == 0:
                        if len(unique_vals) > 1:
                            second_min = np.sort(unique_vals)[1]
                        else:
                            second_min = 1e-6
                        if second_min > .01:
                            replacement = .001
                        else:
                            replacement = second_min / 10

                        features_no_zero[j][k] = replacement

            d = {'means': [np.mean(x) for x in features_2d[i]],
                 'maxs': [np.max(x) for x in features_2d[i]],
                 'sds': [np.std(x) for x in features_2d[i]],
                 'kurts': [kurtosis(x) for x in features_2d[i]],
                 'skews': [skew(x) for x in features_2d[i]],
                 'crests': [np.max(x) / np.sqrt(np.mean(x ** 2)) for x in features_no_zero],
                 'flats': [gmean(np.abs(x)) / np.mean(np.abs(x)) for x in features_no_zero],
                 'gmeans': [gmean(np.abs(x)) for x in features_no_zero]}

            features_2d[i] = d

        features_2d_values = {}
        for i in features_2d.keys():
            for j in features_2d[i].keys():
                for k in range(len(features_2d[i][j])):
                    feat = i + '_' + str(j)[:-1] + '_' + str(k)
                    features_2d_values[feat] = features_2d[i][j][k]

        features_2d = pd.json_normalize(features_2d, sep='_').to_dict('records')[0]
        features_1d.update(features_2d)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            for i in features_1d.keys():
                features_no_zero = features_1d[i].copy()
                unique_vals = np.unique(np.abs(features_1d[i]))
                for j in range(len(features_1d[i])):
                    if features_1d[i][j] == 0:
                        if len(unique_vals) > 1:
                            second_min = np.sort(unique_vals)[1]
                        else:
                            second_min = 1e-6
                        if second_min > .01:
                            replacement = .001
                        else:
                            replacement = second_min / 10

                        features_no_zero[j] = replacement
                ar = np.array(features_1d[i])
                ar_no_zero = np.array(features_no_zero)
                d = {'mean': np.mean(ar),
                     'max': np.max(ar),
                     'sd': np.std(ar),
                     'kurt': kurtosis(ar),
                     'skew': skew(ar),
                     'crest': np.max(ar_no_zero) / np.sqrt(np.mean(ar_no_zero ** 2)),
                     'flat': gmean(np.abs(ar_no_zero)) / np.mean(np.abs(ar_no_zero)),
                     'gmean': gmean(np.abs(ar_no_zero))}

                features_1d[i] = d

        features_1d = pd.json_normalize(features_1d, sep='_').to_dict('records')[0]

        librosa_features = {}
        librosa_features.update(features_num)
        librosa_features.update(features_2d_values)
        librosa_features.update(features_1d)
        librosa_features.update(entropies)
        pop_features = ['chroma_max_0', 'chroma_max_2', 'chroma_max_7', 'chroma_max_9',
                        'chroma_maxs_max', 'cqt_entropy', 'spectral_contrast_entropy']
        for pop_f in pop_features:
            librosa_features.pop(pop_f)

        return librosa_features

    @staticmethod
    def _get_chord_features(y, sr, bpm):
        templates = [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                     [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                     [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]]

        chords = ['N', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'Gm',
                  'G#m', 'Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m']

        chord_feats = ['var', 'same_chord2'] + \
                      ['bar_variety' + str(i) for i in [4, 16]]
        chroma_kinds = ['stft', 'cqt', 'cens']
        d = {ck: {cf: [] for cf in chord_feats} for ck in chroma_kinds}
        k = np.array([2 ** j for j in range(7, 15)])

        minutes = librosa.get_duration(y, sr) / 60
        n_beats = bpm * minutes
        nFrames = int(n_beats * 4)
        hop_length = int(len(y) / nFrames)
        for ch in chroma_kinds:
            if ch == 'stft':
                nfft = hop_length * 2
            else:
                kdif = hop_length - k
                hop_length = k[list(kdif).index(min([k for k in kdif if k >= 0]))]
                y_l = hop_length * nFrames
                start_y = int((round((len(y) - y_l) / 2)))
                y = y[start_y:start_y + y_l]
            if ch == 'stft':
                chroma = librosa.feature.chroma_stft(y, sr, hop_length=hop_length, n_fft=nfft)
            elif ch == 'cqt':
                chroma = librosa.feature.chroma_cqt(y, sr, hop_length=hop_length)
            else:
                chroma = librosa.feature.chroma_cens(y, sr, hop_length=hop_length)
            id_chord = np.zeros(nFrames, dtype='int32')
            max_cor = np.zeros(nFrames)
            for n in range(nFrames):
                cor_vec = np.zeros(24)
                for ni in range(24):
                    cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))
                max_cor[n] = np.max(cor_vec)
                id_chord[n] = np.argmax(cor_vec) + 1

            id_chord[np.where(max_cor < .5 * np.max(max_cor))] = 0

            chord_list = [chords[id_chord[n]] for n in range(nFrames)]
            perc_chords = [chord_list.count(j) / len(chord_list) for j in chords]

            d[ch]['var'] = np.var(perc_chords)
            d[ch]['same_chord2'] = np.mean(np.array([chord_list[j] == chord_list[j - 2]
                                                     for j in range(2, len(chord_list) - 2)]))
            for z in [4, 16]:
                d[ch]['bar_variety' + str(z)] = np.mean([len(np.unique(chord_list[j:(j + z)]))
                                                         for j in range(len(chord_list) - z)])

            features_chord = pd.json_normalize(d, sep='_').to_dict('records')[0]

        return features_chord

    def _get_vocalness_feature(self, filename, y_o):
        with set_dir(self.tracks_dir):
            mapname, extension = os.path.splitext(filename)
            mapname = re.sub(r'[^0-9a-zA-Z]+', '', mapname)
            mapname_ext = mapname + extension
            time.sleep(1)
            os.rename(filename, mapname_ext)
            command = "spleeter separate -p spleeter:2stems -o output " + mapname_ext
            subprocess.run(["powershell", "-Command", command])
            rename_counter = 0
            map_rename_error = True
            while (rename_counter < 30) & map_rename_error:
                try:
                    os.rename(mapname_ext, filename)
                    map_rename_error = False
                except PermissionError:
                    time.sleep(1)
                    rename_counter += 1

        with set_dir(f'{self.tracks_dir}output\\{mapname}'):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                y_v, sr_v = librosa.load('vocals.wav', sr=44100)

        with set_dir(f'{self.tracks_dir}\\output\\'):
            shutil.rmtree(mapname, onerror=self._on_rm_error)

        overall_max = np.max(np.abs(y_o))
        y_v = np.abs(y_v) / overall_max
        threshold = .2
        sq = 0.11965
        vocalness = np.mean(y_v > threshold) ** sq
        pA = pB = 8
        xVals = np.linspace(0, 1, num=101)
        yVals = beta.pdf(x=xVals, a=pA, b=pB)
        vals = {'x': xVals,
                'y': yVals / max(yVals)}
        uncertainty = vals['y'][np.where(round(vocalness, 2) == np.round(np.array(xVals), 2))[0][0]]
        if uncertainty == 0:
            uncertainty = .0001
        if vocalness > .5:
            vocalness **= uncertainty
        else:
            vocalness **= (1 / uncertainty)

        vocalness_feature = {'vocalness': round(vocalness, 2),
                             'file_type': extension,
                             'datetime_analyzed': datetime.now()}

        return vocalness_feature

    def get(self, i):
        filename, y, sr, bpm = self._load_waveform(i)
        self.wave_features = self._get_librosa_features(y, sr)
        self.wave_features.update(self._get_chord_features(y, sr, bpm))
        self.wave_features.update(self._get_vocalness_feature(filename, y))

    @staticmethod
    def _on_rm_error(func, path, exc_info):
        # path contains the path of the file that couldn't be removed
        # let's just assume that it's read-only and unlink it.
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)


class FeaturesImprovement:

    def __init__(self, af):

        self.af = af

    def _remove_accented_characters(self):
        unaccent_cols = ['Track Title', 'Mix Name', 'Artist', 'Composer', 'Label',
                         'Album', 'sp_artist', 'sp_trackname', 'yt_name']

        for uc in unaccent_cols:
            self.af[uc] = unidecode(self.af[uc])

    def _remove_sp_artist_from_name_dif(self):
        s_name_dif = self.af['rb_sp_name_dif']
        s_artist = re.sub(r'[^A-Za-z0-9 À-ú]', '', self.af['Artist']).lower().split(' ')
        if not all([a in s_name_dif for a in s_artist]):
            self.af['rb_sp_name_dif'] = [s for s in s_name_dif if s not in s_artist]

    def _remove_typical_sp_yt_words_from_name_dif(self):
        yt_words = ['official', 'audio', 'music', 'video', 'full',
                    'enhanced', 'track', 'feat', 'ft', 'featuring',
                    'hq', 'premiere', 'records', 'hd', 'and', 'the']
        years = [str(i) for i in range(1955, datetime.now().year)]
        all_yt_words = yt_words + years
        all_sp_words = ['and', 'feat']

        self.af['yt_rb_name_dif'] = [w for w in self.af['yt_rb_name_dif'] if w not in all_yt_words]
        self.af['sp_rb_name_dif'] = [w for w in self.af['sp_rb_name_dif'] if w not in all_sp_words]

    def _remove_yt_artist_from_name_dif(self):
        r_artist = unidecode(re.sub(r'[^a-zA-Z0-9 À-ú]', '', self.af['Artist'])).lower().split(' ')
        self.af['yt_rb_name_dif'] = [w for w in self.af['yt_rb_name_dif'] if w not in r_artist]

    def _classify_sp_yt_dif_types(self):
        sp_yt = ['sp', 'yt']
        for sy in sp_yt:
            s0 = self.af[f'rb_{sy}_name_dif']
            s1 = self.af[f'{sy}_rb_name_dif']
            if (len(s0) > 0) & (len(s1) > 0):
                for s in s0:
                    sd = [levenshtein_distance(s, ss) for ss in s1]
                    if 1 in sd:
                        s0 = [x for x in s0 if x != s]
                        s1 = [x for x in s1 if x != s1[sd.index(1)]]

            ss = [s0, s1]
            name_dif_strings = [f'rb_{sy}_name_dif', f'{sy}_rb_name_dif']
            for s, nd in zip(ss, name_dif_strings):
                if ('original' in s) and ('mix' in s):
                    self.af[nd] = [om for om in s if om not in ['original', 'mix']]
                elif ('original' in s) and ('mix' not in s):
                    self.af[nd] = [o for o in s if o != 'original']
                elif s == ['']:
                    self.af[nd] = []
                else:
                    self.af[nd] = s

            if ((self.af[f'{sy}_duration'] * 1.05 > self.af['rb_duration']) &
                    (self.af[f'{sy}_duration'] * .95 < self.af['rb_duration'])):
                self.af[f'{sy}_same_duration'] = True
            else:
                self.af[f'{sy}_same_duration'] = False

            if (len(self.af[f'rb_{sy}_name_dif']) > 1) | (len(self.af[f'{sy}_rb_name_dif']) > 1):
                self.af[f'{sy}_same_name'] = False
            else:
                self.af[f'{sy}_same_name'] = True

            sy_id = 'sp_id' if sy == 'sp' else 'yt_name'
            if self.af[sy_id] == '':
                self.af[f'{sy}_dif_type'] = 'no song'
            elif self.af[f'{sy}_same_name'] & self.af[f'{sy}_same_duration']:
                self.af[f'{sy}_dif_type'] = 'same'
            elif self.af[f'{sy}_same_name'] & ~self.af[f'{sy}_same_duration']:
                self.af[f'{sy}_dif_type'] = 'other version'
            else:
                self.af[f'{sy}_dif_type'] = 'other song'

    def improve(self):
        self._remove_accented_characters()
        self._remove_sp_artist_from_name_dif()
        self._remove_yt_artist_from_name_dif()
        self._remove_typical_sp_yt_words_from_name_dif()
        self._classify_sp_yt_dif_types()


class TestFeatures:

    def __init__(self, data):
        self.data = data

    def test(self):
        for i in range(len(self.data)):
            self._test_features_names(self.data[i].keys())
            self._test_types(self.data[i])

    @staticmethod
    def _test_features_names(all_features):
        all_features_names = {'Track Title', 'Mix Name', 'Artist', 'Composer', 'Label', 'Album', 'Rating', 'File Name',
                              'Date Added', 'Bitrate', 'BPM', 'Time', 'Key', 'Afro_Disco', 'Balearic', 'Cosmic',
                              'Disco',
                              'Italo_Disco', 'Nu_Disco', 'Acid_House', 'Deep_House', 'House', 'Indie', 'Techno',
                              'Nostalgia', 'Old_Deep_House', 'Offbeat', 'rb_duration', 'track_kind', 'sp_id',
                              'sp_artist',
                              'sp_trackname', 'sp_duration', 'sp_popularity', 'sp_rb_name_dif', 'rb_sp_name_dif',
                              'sp_danceability', 'sp_energy', 'sp_valence', 'sp_instrumentalness', 'sp_speechiness',
                              'sp_acousticness', 'sp_loudness', 'sp_key', 'sp_mode', 'yt_name', 'yt_views',
                              'yt_publish_date', 'yt_duration', 'yt_category', 'yt_rb_name_dif', 'rb_yt_name_dif',
                              'sample_rate', 'beat_strength', 'beat_dif', 'max_loudness', 'onset_rate', 'hfc',
                              'chroma_mean_0', 'chroma_mean_1', 'chroma_mean_2', 'chroma_mean_3', 'chroma_mean_4',
                              'chroma_mean_5', 'chroma_mean_6', 'chroma_mean_7', 'chroma_mean_8', 'chroma_mean_9',
                              'chroma_mean_10', 'chroma_mean_11', 'chroma_max_1', 'chroma_max_3', 'chroma_max_4',
                              'chroma_max_5', 'chroma_max_6', 'chroma_max_8', 'chroma_max_10', 'chroma_max_11',
                              'chroma_sd_0', 'chroma_sd_1', 'chroma_sd_2', 'chroma_sd_3', 'chroma_sd_4', 'chroma_sd_5',
                              'chroma_sd_6', 'chroma_sd_7', 'chroma_sd_8', 'chroma_sd_9', 'chroma_sd_10',
                              'chroma_sd_11',
                              'chroma_kurt_0', 'chroma_kurt_1', 'chroma_kurt_2', 'chroma_kurt_3', 'chroma_kurt_4',
                              'chroma_kurt_5', 'chroma_kurt_6', 'chroma_kurt_7', 'chroma_kurt_8', 'chroma_kurt_9',
                              'chroma_kurt_10', 'chroma_kurt_11', 'chroma_skew_0', 'chroma_skew_1', 'chroma_skew_2',
                              'chroma_skew_3', 'chroma_skew_4', 'chroma_skew_5', 'chroma_skew_6', 'chroma_skew_7',
                              'chroma_skew_8', 'chroma_skew_9', 'chroma_skew_10', 'chroma_skew_11', 'chroma_crest_0',
                              'chroma_crest_1', 'chroma_crest_2', 'chroma_crest_3', 'chroma_crest_4', 'chroma_crest_5',
                              'chroma_crest_6', 'chroma_crest_7', 'chroma_crest_8', 'chroma_crest_9', 'chroma_crest_10',
                              'chroma_crest_11', 'chroma_flat_0', 'chroma_flat_1', 'chroma_flat_2', 'chroma_flat_3',
                              'chroma_flat_4', 'chroma_flat_5', 'chroma_flat_6', 'chroma_flat_7', 'chroma_flat_8',
                              'chroma_flat_9', 'chroma_flat_10', 'chroma_flat_11', 'chroma_gmean_0', 'chroma_gmean_1',
                              'chroma_gmean_2', 'chroma_gmean_3', 'chroma_gmean_4', 'chroma_gmean_5', 'chroma_gmean_6',
                              'chroma_gmean_7', 'chroma_gmean_8', 'chroma_gmean_9', 'chroma_gmean_10',
                              'chroma_gmean_11',
                              'cqt_mean_0', 'cqt_mean_1', 'cqt_mean_2', 'cqt_mean_3', 'cqt_mean_4', 'cqt_mean_5',
                              'cqt_max_0', 'cqt_max_1', 'cqt_max_2', 'cqt_max_3', 'cqt_max_4', 'cqt_max_5', 'cqt_sd_0',
                              'cqt_sd_1', 'cqt_sd_2', 'cqt_sd_3', 'cqt_sd_4', 'cqt_sd_5', 'cqt_kurt_0', 'cqt_kurt_1',
                              'cqt_kurt_2', 'cqt_kurt_3', 'cqt_kurt_4', 'cqt_kurt_5', 'cqt_skew_0', 'cqt_skew_1',
                              'cqt_skew_2', 'cqt_skew_3', 'cqt_skew_4', 'cqt_skew_5', 'cqt_crest_0', 'cqt_crest_1',
                              'cqt_crest_2', 'cqt_crest_3', 'cqt_crest_4', 'cqt_crest_5', 'cqt_flat_0', 'cqt_flat_1',
                              'cqt_flat_2', 'cqt_flat_3', 'cqt_flat_4', 'cqt_flat_5', 'cqt_gmean_0', 'cqt_gmean_1',
                              'cqt_gmean_2', 'cqt_gmean_3', 'cqt_gmean_4', 'cqt_gmean_5', 'gfcc_mean_0', 'gfcc_mean_1',
                              'gfcc_mean_2', 'gfcc_mean_3', 'gfcc_mean_4', 'gfcc_mean_5', 'gfcc_mean_6', 'gfcc_mean_7',
                              'gfcc_mean_8', 'gfcc_mean_9', 'gfcc_mean_10', 'gfcc_mean_11', 'gfcc_mean_12',
                              'gfcc_max_0',
                              'gfcc_max_1', 'gfcc_max_2', 'gfcc_max_3', 'gfcc_max_4', 'gfcc_max_5', 'gfcc_max_6',
                              'gfcc_max_7', 'gfcc_max_8', 'gfcc_max_9', 'gfcc_max_10', 'gfcc_max_11', 'gfcc_max_12',
                              'gfcc_sd_0', 'gfcc_sd_1', 'gfcc_sd_2', 'gfcc_sd_3', 'gfcc_sd_4', 'gfcc_sd_5', 'gfcc_sd_6',
                              'gfcc_sd_7', 'gfcc_sd_8', 'gfcc_sd_9', 'gfcc_sd_10', 'gfcc_sd_11', 'gfcc_sd_12',
                              'gfcc_kurt_0', 'gfcc_kurt_1', 'gfcc_kurt_2', 'gfcc_kurt_3', 'gfcc_kurt_4', 'gfcc_kurt_5',
                              'gfcc_kurt_6', 'gfcc_kurt_7', 'gfcc_kurt_8', 'gfcc_kurt_9', 'gfcc_kurt_10',
                              'gfcc_kurt_11',
                              'gfcc_kurt_12', 'gfcc_skew_0', 'gfcc_skew_1', 'gfcc_skew_2', 'gfcc_skew_3', 'gfcc_skew_4',
                              'gfcc_skew_5', 'gfcc_skew_6', 'gfcc_skew_7', 'gfcc_skew_8', 'gfcc_skew_9', 'gfcc_skew_10',
                              'gfcc_skew_11', 'gfcc_skew_12', 'gfcc_crest_0', 'gfcc_crest_1', 'gfcc_crest_2',
                              'gfcc_crest_3', 'gfcc_crest_4', 'gfcc_crest_5', 'gfcc_crest_6', 'gfcc_crest_7',
                              'gfcc_crest_8', 'gfcc_crest_9', 'gfcc_crest_10', 'gfcc_crest_11', 'gfcc_crest_12',
                              'gfcc_flat_0', 'gfcc_flat_1', 'gfcc_flat_2', 'gfcc_flat_3', 'gfcc_flat_4', 'gfcc_flat_5',
                              'gfcc_flat_6', 'gfcc_flat_7', 'gfcc_flat_8', 'gfcc_flat_9', 'gfcc_flat_10',
                              'gfcc_flat_11',
                              'gfcc_flat_12', 'gfcc_gmean_0', 'gfcc_gmean_1', 'gfcc_gmean_2', 'gfcc_gmean_3',
                              'gfcc_gmean_4', 'gfcc_gmean_5', 'gfcc_gmean_6', 'gfcc_gmean_7', 'gfcc_gmean_8',
                              'gfcc_gmean_9', 'gfcc_gmean_10', 'gfcc_gmean_11', 'gfcc_gmean_12', 'melbands_mean_0',
                              'melbands_mean_1', 'melbands_mean_2', 'melbands_mean_3', 'melbands_mean_4',
                              'melbands_mean_5',
                              'melbands_mean_6', 'melbands_mean_7', 'melbands_mean_8', 'melbands_mean_9',
                              'melbands_mean_10', 'melbands_mean_11', 'melbands_mean_12', 'melbands_max_0',
                              'melbands_max_1', 'melbands_max_2', 'melbands_max_3', 'melbands_max_4', 'melbands_max_5',
                              'melbands_max_6', 'melbands_max_7', 'melbands_max_8', 'melbands_max_9', 'melbands_max_10',
                              'melbands_max_11', 'melbands_max_12', 'melbands_sd_0', 'melbands_sd_1', 'melbands_sd_2',
                              'melbands_sd_3', 'melbands_sd_4', 'melbands_sd_5', 'melbands_sd_6', 'melbands_sd_7',
                              'melbands_sd_8', 'melbands_sd_9', 'melbands_sd_10', 'melbands_sd_11', 'melbands_sd_12',
                              'melbands_kurt_0', 'melbands_kurt_1', 'melbands_kurt_2', 'melbands_kurt_3',
                              'melbands_kurt_4',
                              'melbands_kurt_5', 'melbands_kurt_6', 'melbands_kurt_7', 'melbands_kurt_8',
                              'melbands_kurt_9',
                              'melbands_kurt_10', 'melbands_kurt_11', 'melbands_kurt_12', 'melbands_skew_0',
                              'melbands_skew_1', 'melbands_skew_2', 'melbands_skew_3', 'melbands_skew_4',
                              'melbands_skew_5',
                              'melbands_skew_6', 'melbands_skew_7', 'melbands_skew_8', 'melbands_skew_9',
                              'melbands_skew_10', 'melbands_skew_11', 'melbands_skew_12', 'melbands_crest_0',
                              'melbands_crest_1', 'melbands_crest_2', 'melbands_crest_3', 'melbands_crest_4',
                              'melbands_crest_5', 'melbands_crest_6', 'melbands_crest_7', 'melbands_crest_8',
                              'melbands_crest_9', 'melbands_crest_10', 'melbands_crest_11', 'melbands_crest_12',
                              'melbands_flat_0', 'melbands_flat_1', 'melbands_flat_2', 'melbands_flat_3',
                              'melbands_flat_4',
                              'melbands_flat_5', 'melbands_flat_6', 'melbands_flat_7', 'melbands_flat_8',
                              'melbands_flat_9',
                              'melbands_flat_10', 'melbands_flat_11', 'melbands_flat_12', 'melbands_gmean_0',
                              'melbands_gmean_1', 'melbands_gmean_2', 'melbands_gmean_3', 'melbands_gmean_4',
                              'melbands_gmean_5', 'melbands_gmean_6', 'melbands_gmean_7', 'melbands_gmean_8',
                              'melbands_gmean_9', 'melbands_gmean_10', 'melbands_gmean_11', 'melbands_gmean_12',
                              'mfcc_mean_0', 'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
                              'mfcc_mean_6', 'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10',
                              'mfcc_mean_11',
                              'mfcc_mean_12', 'mfcc_max_0', 'mfcc_max_1', 'mfcc_max_2', 'mfcc_max_3', 'mfcc_max_4',
                              'mfcc_max_5', 'mfcc_max_6', 'mfcc_max_7', 'mfcc_max_8', 'mfcc_max_9', 'mfcc_max_10',
                              'mfcc_max_11', 'mfcc_max_12', 'mfcc_sd_0', 'mfcc_sd_1', 'mfcc_sd_2', 'mfcc_sd_3',
                              'mfcc_sd_4',
                              'mfcc_sd_5', 'mfcc_sd_6', 'mfcc_sd_7', 'mfcc_sd_8', 'mfcc_sd_9', 'mfcc_sd_10',
                              'mfcc_sd_11',
                              'mfcc_sd_12', 'mfcc_kurt_0', 'mfcc_kurt_1', 'mfcc_kurt_2', 'mfcc_kurt_3', 'mfcc_kurt_4',
                              'mfcc_kurt_5', 'mfcc_kurt_6', 'mfcc_kurt_7', 'mfcc_kurt_8', 'mfcc_kurt_9', 'mfcc_kurt_10',
                              'mfcc_kurt_11', 'mfcc_kurt_12', 'mfcc_skew_0', 'mfcc_skew_1', 'mfcc_skew_2',
                              'mfcc_skew_3',
                              'mfcc_skew_4', 'mfcc_skew_5', 'mfcc_skew_6', 'mfcc_skew_7', 'mfcc_skew_8', 'mfcc_skew_9',
                              'mfcc_skew_10', 'mfcc_skew_11', 'mfcc_skew_12', 'mfcc_crest_0', 'mfcc_crest_1',
                              'mfcc_crest_2', 'mfcc_crest_3', 'mfcc_crest_4', 'mfcc_crest_5', 'mfcc_crest_6',
                              'mfcc_crest_7', 'mfcc_crest_8', 'mfcc_crest_9', 'mfcc_crest_10', 'mfcc_crest_11',
                              'mfcc_crest_12', 'mfcc_flat_0', 'mfcc_flat_1', 'mfcc_flat_2', 'mfcc_flat_3',
                              'mfcc_flat_4',
                              'mfcc_flat_5', 'mfcc_flat_6', 'mfcc_flat_7', 'mfcc_flat_8', 'mfcc_flat_9', 'mfcc_flat_10',
                              'mfcc_flat_11', 'mfcc_flat_12', 'mfcc_gmean_0', 'mfcc_gmean_1', 'mfcc_gmean_2',
                              'mfcc_gmean_3', 'mfcc_gmean_4', 'mfcc_gmean_5', 'mfcc_gmean_6', 'mfcc_gmean_7',
                              'mfcc_gmean_8', 'mfcc_gmean_9', 'mfcc_gmean_10', 'mfcc_gmean_11', 'mfcc_gmean_12',
                              'spectral_contrast_mean_0', 'spectral_contrast_mean_1', 'spectral_contrast_mean_2',
                              'spectral_contrast_mean_3', 'spectral_contrast_mean_4', 'spectral_contrast_mean_5',
                              'spectral_contrast_mean_6', 'spectral_contrast_max_0', 'spectral_contrast_max_1',
                              'spectral_contrast_max_2', 'spectral_contrast_max_3', 'spectral_contrast_max_4',
                              'spectral_contrast_max_5', 'spectral_contrast_max_6', 'spectral_contrast_sd_0',
                              'spectral_contrast_sd_1', 'spectral_contrast_sd_2', 'spectral_contrast_sd_3',
                              'spectral_contrast_sd_4', 'spectral_contrast_sd_5', 'spectral_contrast_sd_6',
                              'spectral_contrast_kurt_0', 'spectral_contrast_kurt_1', 'spectral_contrast_kurt_2',
                              'spectral_contrast_kurt_3', 'spectral_contrast_kurt_4', 'spectral_contrast_kurt_5',
                              'spectral_contrast_kurt_6', 'spectral_contrast_skew_0', 'spectral_contrast_skew_1',
                              'spectral_contrast_skew_2', 'spectral_contrast_skew_3', 'spectral_contrast_skew_4',
                              'spectral_contrast_skew_5', 'spectral_contrast_skew_6', 'spectral_contrast_crest_0',
                              'spectral_contrast_crest_1', 'spectral_contrast_crest_2', 'spectral_contrast_crest_3',
                              'spectral_contrast_crest_4', 'spectral_contrast_crest_5', 'spectral_contrast_crest_6',
                              'spectral_contrast_flat_0', 'spectral_contrast_flat_1', 'spectral_contrast_flat_2',
                              'spectral_contrast_flat_3', 'spectral_contrast_flat_4', 'spectral_contrast_flat_5',
                              'spectral_contrast_flat_6', 'spectral_contrast_gmean_0', 'spectral_contrast_gmean_1',
                              'spectral_contrast_gmean_2', 'spectral_contrast_gmean_3', 'spectral_contrast_gmean_4',
                              'spectral_contrast_gmean_5', 'spectral_contrast_gmean_6', 'rms_mean', 'rms_max', 'rms_sd',
                              'rms_kurt', 'rms_skew', 'rms_crest', 'rms_flat', 'rms_gmean', 'spectral_bandwith_mean',
                              'spectral_bandwith_max', 'spectral_bandwith_sd', 'spectral_bandwith_kurt',
                              'spectral_bandwith_skew', 'spectral_bandwith_crest', 'spectral_bandwith_flat',
                              'spectral_bandwith_gmean', 'spectral_centroid_mean', 'spectral_centroid_max',
                              'spectral_centroid_sd', 'spectral_centroid_kurt', 'spectral_centroid_skew',
                              'spectral_centroid_crest', 'spectral_centroid_flat', 'spectral_centroid_gmean',
                              'spectral_flatness_mean', 'spectral_flatness_max', 'spectral_flatness_sd',
                              'spectral_flatness_kurt', 'spectral_flatness_skew', 'spectral_flatness_crest',
                              'spectral_flatness_flat', 'spectral_flatness_gmean', 'spectral_rolloff_mean',
                              'spectral_rolloff_max', 'spectral_rolloff_sd', 'spectral_rolloff_kurt',
                              'spectral_rolloff_skew', 'spectral_rolloff_crest', 'spectral_rolloff_flat',
                              'spectral_rolloff_gmean', 'zero_crossing_rate_mean', 'zero_crossing_rate_max',
                              'zero_crossing_rate_sd', 'zero_crossing_rate_kurt', 'zero_crossing_rate_skew',
                              'zero_crossing_rate_crest', 'zero_crossing_rate_flat', 'zero_crossing_rate_gmean',
                              'chroma_means_mean', 'chroma_means_max', 'chroma_means_sd', 'chroma_means_kurt',
                              'chroma_means_skew', 'chroma_means_crest', 'chroma_means_flat', 'chroma_means_gmean',
                              'chroma_maxs_mean', 'chroma_maxs_sd', 'chroma_maxs_kurt', 'chroma_maxs_skew',
                              'chroma_maxs_crest', 'chroma_maxs_flat', 'chroma_maxs_gmean', 'chroma_sds_mean',
                              'chroma_sds_max', 'chroma_sds_sd', 'chroma_sds_kurt', 'chroma_sds_skew',
                              'chroma_sds_crest',
                              'chroma_sds_flat', 'chroma_sds_gmean', 'chroma_kurts_mean', 'chroma_kurts_max',
                              'chroma_kurts_sd', 'chroma_kurts_kurt', 'chroma_kurts_skew', 'chroma_kurts_crest',
                              'chroma_kurts_flat', 'chroma_kurts_gmean', 'chroma_skews_mean', 'chroma_skews_max',
                              'chroma_skews_sd', 'chroma_skews_kurt', 'chroma_skews_skew', 'chroma_skews_crest',
                              'chroma_skews_flat', 'chroma_skews_gmean', 'chroma_crests_mean', 'chroma_crests_max',
                              'chroma_crests_sd', 'chroma_crests_kurt', 'chroma_crests_skew', 'chroma_crests_crest',
                              'chroma_crests_flat', 'chroma_crests_gmean', 'chroma_flats_mean', 'chroma_flats_max',
                              'chroma_flats_sd', 'chroma_flats_kurt', 'chroma_flats_skew', 'chroma_flats_crest',
                              'chroma_flats_flat', 'chroma_flats_gmean', 'chroma_gmeans_mean', 'chroma_gmeans_max',
                              'chroma_gmeans_sd', 'chroma_gmeans_kurt', 'chroma_gmeans_skew', 'chroma_gmeans_crest',
                              'chroma_gmeans_flat', 'chroma_gmeans_gmean', 'cqt_means_mean', 'cqt_means_max',
                              'cqt_means_sd', 'cqt_means_kurt', 'cqt_means_skew', 'cqt_means_crest', 'cqt_means_flat',
                              'cqt_means_gmean', 'cqt_maxs_mean', 'cqt_maxs_max', 'cqt_maxs_sd', 'cqt_maxs_kurt',
                              'cqt_maxs_skew', 'cqt_maxs_crest', 'cqt_maxs_flat', 'cqt_maxs_gmean', 'cqt_sds_mean',
                              'cqt_sds_max', 'cqt_sds_sd', 'cqt_sds_kurt', 'cqt_sds_skew', 'cqt_sds_crest',
                              'cqt_sds_flat',
                              'cqt_sds_gmean', 'cqt_kurts_mean', 'cqt_kurts_max', 'cqt_kurts_sd', 'cqt_kurts_kurt',
                              'cqt_kurts_skew', 'cqt_kurts_crest', 'cqt_kurts_flat', 'cqt_kurts_gmean',
                              'cqt_skews_mean',
                              'cqt_skews_max', 'cqt_skews_sd', 'cqt_skews_kurt', 'cqt_skews_skew', 'cqt_skews_crest',
                              'cqt_skews_flat', 'cqt_skews_gmean', 'cqt_crests_mean', 'cqt_crests_max', 'cqt_crests_sd',
                              'cqt_crests_kurt', 'cqt_crests_skew', 'cqt_crests_crest', 'cqt_crests_flat',
                              'cqt_crests_gmean', 'cqt_flats_mean', 'cqt_flats_max', 'cqt_flats_sd', 'cqt_flats_kurt',
                              'cqt_flats_skew', 'cqt_flats_crest', 'cqt_flats_flat', 'cqt_flats_gmean',
                              'cqt_gmeans_mean',
                              'cqt_gmeans_max', 'cqt_gmeans_sd', 'cqt_gmeans_kurt', 'cqt_gmeans_skew',
                              'cqt_gmeans_crest',
                              'cqt_gmeans_flat', 'cqt_gmeans_gmean', 'gfcc_means_mean', 'gfcc_means_max',
                              'gfcc_means_sd',
                              'gfcc_means_kurt', 'gfcc_means_skew', 'gfcc_means_crest', 'gfcc_means_flat',
                              'gfcc_means_gmean', 'gfcc_maxs_mean', 'gfcc_maxs_max', 'gfcc_maxs_sd', 'gfcc_maxs_kurt',
                              'gfcc_maxs_skew', 'gfcc_maxs_crest', 'gfcc_maxs_flat', 'gfcc_maxs_gmean', 'gfcc_sds_mean',
                              'gfcc_sds_max', 'gfcc_sds_sd', 'gfcc_sds_kurt', 'gfcc_sds_skew', 'gfcc_sds_crest',
                              'gfcc_sds_flat', 'gfcc_sds_gmean', 'gfcc_kurts_mean', 'gfcc_kurts_max', 'gfcc_kurts_sd',
                              'gfcc_kurts_kurt', 'gfcc_kurts_skew', 'gfcc_kurts_crest', 'gfcc_kurts_flat',
                              'gfcc_kurts_gmean', 'gfcc_skews_mean', 'gfcc_skews_max', 'gfcc_skews_sd',
                              'gfcc_skews_kurt',
                              'gfcc_skews_skew', 'gfcc_skews_crest', 'gfcc_skews_flat', 'gfcc_skews_gmean',
                              'gfcc_crests_mean', 'gfcc_crests_max', 'gfcc_crests_sd', 'gfcc_crests_kurt',
                              'gfcc_crests_skew', 'gfcc_crests_crest', 'gfcc_crests_flat', 'gfcc_crests_gmean',
                              'gfcc_flats_mean', 'gfcc_flats_max', 'gfcc_flats_sd', 'gfcc_flats_kurt',
                              'gfcc_flats_skew',
                              'gfcc_flats_crest', 'gfcc_flats_flat', 'gfcc_flats_gmean', 'gfcc_gmeans_mean',
                              'gfcc_gmeans_max', 'gfcc_gmeans_sd', 'gfcc_gmeans_kurt', 'gfcc_gmeans_skew',
                              'gfcc_gmeans_crest', 'gfcc_gmeans_flat', 'gfcc_gmeans_gmean', 'melbands_means_mean',
                              'melbands_means_max', 'melbands_means_sd', 'melbands_means_kurt', 'melbands_means_skew',
                              'melbands_means_crest', 'melbands_means_flat', 'melbands_means_gmean',
                              'melbands_maxs_mean',
                              'melbands_maxs_max', 'melbands_maxs_sd', 'melbands_maxs_kurt', 'melbands_maxs_skew',
                              'melbands_maxs_crest', 'melbands_maxs_flat', 'melbands_maxs_gmean', 'melbands_sds_mean',
                              'melbands_sds_max', 'melbands_sds_sd', 'melbands_sds_kurt', 'melbands_sds_skew',
                              'melbands_sds_crest', 'melbands_sds_flat', 'melbands_sds_gmean', 'melbands_kurts_mean',
                              'melbands_kurts_max', 'melbands_kurts_sd', 'melbands_kurts_kurt', 'melbands_kurts_skew',
                              'melbands_kurts_crest', 'melbands_kurts_flat', 'melbands_kurts_gmean',
                              'melbands_skews_mean',
                              'melbands_skews_max', 'melbands_skews_sd', 'melbands_skews_kurt', 'melbands_skews_skew',
                              'melbands_skews_crest', 'melbands_skews_flat', 'melbands_skews_gmean',
                              'melbands_crests_mean',
                              'melbands_crests_max', 'melbands_crests_sd', 'melbands_crests_kurt',
                              'melbands_crests_skew',
                              'melbands_crests_crest', 'melbands_crests_flat', 'melbands_crests_gmean',
                              'melbands_flats_mean', 'melbands_flats_max', 'melbands_flats_sd', 'melbands_flats_kurt',
                              'melbands_flats_skew', 'melbands_flats_crest', 'melbands_flats_flat',
                              'melbands_flats_gmean',
                              'melbands_gmeans_mean', 'melbands_gmeans_max', 'melbands_gmeans_sd',
                              'melbands_gmeans_kurt',
                              'melbands_gmeans_skew', 'melbands_gmeans_crest', 'melbands_gmeans_flat',
                              'melbands_gmeans_gmean', 'mfcc_means_mean', 'mfcc_means_max', 'mfcc_means_sd',
                              'mfcc_means_kurt', 'mfcc_means_skew', 'mfcc_means_crest', 'mfcc_means_flat',
                              'mfcc_means_gmean', 'mfcc_maxs_mean', 'mfcc_maxs_max', 'mfcc_maxs_sd', 'mfcc_maxs_kurt',
                              'mfcc_maxs_skew', 'mfcc_maxs_crest', 'mfcc_maxs_flat', 'mfcc_maxs_gmean', 'mfcc_sds_mean',
                              'mfcc_sds_max', 'mfcc_sds_sd', 'mfcc_sds_kurt', 'mfcc_sds_skew', 'mfcc_sds_crest',
                              'mfcc_sds_flat', 'mfcc_sds_gmean', 'mfcc_kurts_mean', 'mfcc_kurts_max', 'mfcc_kurts_sd',
                              'mfcc_kurts_kurt', 'mfcc_kurts_skew', 'mfcc_kurts_crest', 'mfcc_kurts_flat',
                              'mfcc_kurts_gmean', 'mfcc_skews_mean', 'mfcc_skews_max', 'mfcc_skews_sd',
                              'mfcc_skews_kurt',
                              'mfcc_skews_skew', 'mfcc_skews_crest', 'mfcc_skews_flat', 'mfcc_skews_gmean',
                              'mfcc_crests_mean', 'mfcc_crests_max', 'mfcc_crests_sd', 'mfcc_crests_kurt',
                              'mfcc_crests_skew', 'mfcc_crests_crest', 'mfcc_crests_flat', 'mfcc_crests_gmean',
                              'mfcc_flats_mean', 'mfcc_flats_max', 'mfcc_flats_sd', 'mfcc_flats_kurt',
                              'mfcc_flats_skew',
                              'mfcc_flats_crest', 'mfcc_flats_flat', 'mfcc_flats_gmean', 'mfcc_gmeans_mean',
                              'mfcc_gmeans_max', 'mfcc_gmeans_sd', 'mfcc_gmeans_kurt', 'mfcc_gmeans_skew',
                              'mfcc_gmeans_crest', 'mfcc_gmeans_flat', 'mfcc_gmeans_gmean',
                              'spectral_contrast_means_mean',
                              'spectral_contrast_means_max', 'spectral_contrast_means_sd',
                              'spectral_contrast_means_kurt',
                              'spectral_contrast_means_skew', 'spectral_contrast_means_crest',
                              'spectral_contrast_means_flat', 'spectral_contrast_means_gmean',
                              'spectral_contrast_maxs_mean', 'spectral_contrast_maxs_max', 'spectral_contrast_maxs_sd',
                              'spectral_contrast_maxs_kurt', 'spectral_contrast_maxs_skew',
                              'spectral_contrast_maxs_crest',
                              'spectral_contrast_maxs_flat', 'spectral_contrast_maxs_gmean',
                              'spectral_contrast_sds_mean',
                              'spectral_contrast_sds_max', 'spectral_contrast_sds_sd', 'spectral_contrast_sds_kurt',
                              'spectral_contrast_sds_skew', 'spectral_contrast_sds_crest', 'spectral_contrast_sds_flat',
                              'spectral_contrast_sds_gmean', 'spectral_contrast_kurts_mean',
                              'spectral_contrast_kurts_max',
                              'spectral_contrast_kurts_sd', 'spectral_contrast_kurts_kurt',
                              'spectral_contrast_kurts_skew',
                              'spectral_contrast_kurts_crest', 'spectral_contrast_kurts_flat',
                              'spectral_contrast_kurts_gmean', 'spectral_contrast_skews_mean',
                              'spectral_contrast_skews_max', 'spectral_contrast_skews_sd',
                              'spectral_contrast_skews_kurt',
                              'spectral_contrast_skews_skew', 'spectral_contrast_skews_crest',
                              'spectral_contrast_skews_flat', 'spectral_contrast_skews_gmean',
                              'spectral_contrast_crests_mean', 'spectral_contrast_crests_max',
                              'spectral_contrast_crests_sd', 'spectral_contrast_crests_kurt',
                              'spectral_contrast_crests_skew', 'spectral_contrast_crests_crest',
                              'spectral_contrast_crests_flat', 'spectral_contrast_crests_gmean',
                              'spectral_contrast_flats_mean', 'spectral_contrast_flats_max',
                              'spectral_contrast_flats_sd',
                              'spectral_contrast_flats_kurt', 'spectral_contrast_flats_skew',
                              'spectral_contrast_flats_crest', 'spectral_contrast_flats_flat',
                              'spectral_contrast_flats_gmean', 'spectral_contrast_gmeans_mean',
                              'spectral_contrast_gmeans_max', 'spectral_contrast_gmeans_sd',
                              'spectral_contrast_gmeans_kurt', 'spectral_contrast_gmeans_skew',
                              'spectral_contrast_gmeans_crest', 'spectral_contrast_gmeans_flat',
                              'spectral_contrast_gmeans_gmean', 'chroma_entropy', 'gfcc_entropy', 'melbands_entropy',
                              'mfcc_entropy', 'stft_var', 'stft_same_chord2', 'stft_bar_variety4', 'stft_bar_variety16',
                              'cqt_var', 'cqt_same_chord2', 'cqt_bar_variety4', 'cqt_bar_variety16', 'cens_var',
                              'cens_same_chord2', 'cens_bar_variety4', 'cens_bar_variety16', 'vocalness', 'file_type',
                              'datetime_analyzed', 'sp_same_duration', 'sp_same_name', 'sp_dif_type',
                              'yt_same_duration',
                              'yt_same_name', 'yt_dif_type'}
        diff = all_features_names - set(all_features)

        assert len(diff) == 0, f'Incomplete set of features, missing: {diff}'

    @staticmethod
    def _test_types(all_features):
        all_types = [type(f).__name__ for f in all_features.values()]
        check_nan = [np.isnan(f) for f in all_features.values() if type(f).__name__.__contains__('float')]
        check_none = [t == 'NoneType' for t in all_types]

        assert sum(check_nan) == 0, 'NaNs discovered in data'
        assert sum(check_none) == 0, 'NoneTypes discovered in data'


class Popularity:

    def __init__(self, data):
        self.data = data

        self.complete = None

    def get(self):
        popularities_in_data = ['popularity' in list(self.data[i].keys()) for i in range(len(self.data))]
        n_popularities = np.sum(popularities_in_data)
        self.complete = n_popularities < len(self.data)
        if self.complete:
            self._calculate_popularity_score()

    def _calculate_popularity_score(self):
        rl = range(len(self.data))
        for i in rl:
            if self.data[i]['yt_publish_date'] != '':
                self.data[i]['yt_days_since_publish'] = (datetime.now() -
                                                         datetime.strptime(self.data[i]['yt_publish_date'],
                                                                           '%Y-%m-%d')).days
                self.data[i]['yt_views_per_day'] = round(
                    self.data[i]['yt_views'] / self.data[i]['yt_days_since_publish'], 2)
            else:
                self.data[i]['yt_days_since_publish'] = 0
                self.data[i]['yt_views_per_day'] = 0
        sp_pop_dist = []
        yt_pop_dist = []
        for i in rl:
            if self.data[i]['sp_dif_type'] in ['same', 'other_version']:
                sp_pop_dist.append(self.data[i]['sp_popularity'])
            else:
                sp_pop_dist.append(0)
            if self.data[i]['yt_dif_type'] in ['same', 'other_version']:
                yt_pop_dist.append(self.data[i]['yt_views_per_day'])
            else:
                yt_pop_dist.append(0)
        yt_pop_dist = np.array(yt_pop_dist)
        yt_pop_dist[yt_pop_dist <= 0] = .001
        yt_pop_dist = np.log(yt_pop_dist)
        yt_pop_dist[yt_pop_dist < 0] = 0
        yt_pop_dist *= (max(sp_pop_dist) / max(yt_pop_dist))

        for i in rl:
            self.data[i]['sp_popularity'] = sp_pop_dist[i]
            self.data[i]['yt_popularity'] = yt_pop_dist[i]
            self.data[i]['popularity'] = (self.data[i]['sp_popularity'] + self.data[i]['yt_popularity']) / 2
