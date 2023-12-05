import subprocess
import warnings
import os
import shutil
from unidecode import unidecode
from datetime import datetime
import re
import time

import pandas as pd
from joblib import load, dump
import numpy as np
from pytube import extract
import librosa
from spafe.features.gfcc import gfcc
from entropy_estimators import continuous
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis, skew, beta

from base.spotify_youtube import (get_spotify_audio_features, search_spotify_tracks, get_youtube_link,
                                  find_youtube_elements, get_youtube_video_properties)
from base.connect import (SpotifyConnect, YoutubeConnect)
from base.helpers import levenshtein_distance
from update_dataset.helpers import (jaccard_similarity, neutralize,
                                    create_name_string, check_original, add_or_remove_original,
                                    set_dir, find, on_rm_error)


class RekordboxMusic:

    def __init__(self,
                 path):

        self.file_location = path

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

        return self.data

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


class ExplorerInterruption:

    def __init__(self, data_rm, tracks_dir):
        self.data_rm = data_rm
        self.tracks_dir = tracks_dir

    def change_shortened_filenames(self):
        data_rm_filenames_original = [d['File Name'] for d in self.data_rm]
        data_rm_filenames_short = [
            re.sub(r'[^0-9a-zA-Z]+', '', os.path.splitext(filename)[0]) + os.path.splitext(filename)[1]
            for filename in data_rm_filenames_original]

        with set_dir(self.tracks_dir):
            dir_filenames = os.listdir()
            for fn_original, fn_short in zip(data_rm_filenames_original, data_rm_filenames_short):
                if (fn_short in dir_filenames) & (fn_short != fn_original):
                    os.rename(fn_short, fn_original)

    def empty_output_map(self):
        with set_dir(f'{self.tracks_dir}\\output'):
            temp_maps = os.listdir()
            if len(temp_maps) > 0:
                for mapname in temp_maps:
                    shutil.rmtree(mapname, onerror=on_rm_error)


class Disjoint:

    def __init__(self, data_rm, data_mm, tracks_dir, quick_test):
        self.data_rm = data_mm
        self.data_mm = data_mm
        self.tracks_dir = tracks_dir
        self.__quick_test = quick_test

        self.filenames_rm = set([d['File Name'] for d in data_rm])
        self.filenames_mm = set([d['File Name'] for d in data_mm])
        self.filenames_tracks_dir = set(os.listdir(tracks_dir))

        self.filenames_added = None
        self.filenames_removed = None
        self.filenames_wave = None
        self.n_changes = None

    def check_missing_filenames_in_tracks_dir(self):
        filenames_missing_in_tracks_dir = list(self.filenames_rm - self.filenames_tracks_dir)
        n_missing_tracks = len(filenames_missing_in_tracks_dir)
        if n_missing_tracks > 0:
            raise FileNotFoundError(f'{n_missing_tracks} tracks need to be copied to tracks_dir: {filenames_missing_in_tracks_dir}')

    def get_added_tracks(self):
        self.filenames_added = list(self.filenames_rm - self.filenames_mm)
        return self.filenames_added

    def get_removed_tracks(self):
        self.filenames_removed = list(self.filenames_mm - self.filenames_rm)
        return self.filenames_removed

    def get_tracks_for_wave_analysis(self):
        check_col = 'wave_col' if self.__quick_test else 'sample_rate'
        self.filenames_wave = self.filenames_added + [d['File Name'] for d in self.data_mm if check_col not in d.keys()]
        self.n_changes = len(self.filenames_wave)
        return self.filenames_wave


class SpotifyFeatures(SpotifyConnect):

    def __init__(self, rb_data):

        super().__init__()

        self.rb_data = rb_data

        self.results = None
        self.i = None
        self.spotify_features = {}
        self.track_main_features = None
        self.track_audio_features = None
        self.conn_error_main = False
        self.conn_error_audio = False

        self._is_original = None
        self._best = None

        self.__rb_name_set = None

    def get(self, i):
        self.get_track_main_features(i)
        self.spotify_features.update(self.track_main_features)
        self.get_track_audio_features(self.track_main_features['sp_id'])
        self.spotify_features.update(self.track_audio_features)
        self.spotify_features.update({'sp_main_conn_error': self.conn_error_main,
                                      'sp_audio_conn_error': self.conn_error_audio,
                                      'datetime_analyzed_sp_yt': datetime.now()})

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
            audio_features, self.conn_error_audio = get_spotify_audio_features(self.sp, sp_id)
            self.track_audio_features = {f'sp_{feature}': audio_features[0][f'{feature}'] for feature in features_list}
        else:
            self.track_audio_features = {f'sp_{feature}': 0 for feature in features_list}

    def _search(self):
        query_name_parts = ['Artist', 'Mix Name']
        artist_track = create_name_string(self.rb_data, self.i, query_name_parts)
        self.results, self.conn_error_main = search_spotify_tracks(self.sp, artist_track)
        self.results = self.results['tracks']['items']

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


class YoutubeFeatures(YoutubeConnect):

    def __init__(self, rb_data):
        super().__init__()

        self.rb_data = rb_data

        self.i = None
        self.youtube_features = None
        self.conn_error_search = None
        self.conn_error_video = None

    def _create_rb_name_set(self):
        full_name = create_name_string(self.rb_data, self.i, ['Artist', 'Mix Name', 'Composer', 'Album', 'Label'])
        return full_name.split()

    def _create_yt_search_link(self):
        search_query = self.rb_data[self.i]['Track Title'].replace(' ', '+').replace('&', '%26')
        return 'https://www.youtube.com/results?search_query=' + search_query

    def _input_link(self, link):
        get_youtube_link(self.driver, link)

    def _search(self):
        return find_youtube_elements(self.driver)

    @staticmethod
    def _extract_youtube_id(user_data):
        youtube_url = user_data[0].get_attribute('href')
        return extract.video_id(youtube_url)

    def _pull_features(self, yt_id):
        return get_youtube_video_properties(self.youtube, yt_id)

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
        user_data, self.conn_error_search = self._search()

        if len(user_data) > 0:
            if not user_data[0].get_attribute('href') is None:
                yt_id = self._extract_youtube_id(user_data)
                yt_result, self.conn_error_video = self._pull_features(yt_id)
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
            'rb_yt_name_dif': list(set(rb_name_set) - set(yt_name_set)),
            'yt_search_conn_error': self.conn_error_search,
            'yt_video_conn_error': self.conn_error_video
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
            shutil.rmtree(mapname, onerror=on_rm_error)

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
                             'datetime_analyzed_wave': datetime.now()}

        return vocalness_feature

    def get(self, i):
        filename, y, sr, bpm = self._load_waveform(i)
        self.wave_features = self._get_librosa_features(y, sr)
        self.wave_features.update(self._get_chord_features(y, sr, bpm))
        self.wave_features.update(self._get_vocalness_feature(filename, y))


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


class Popularity:

    def __init__(self, data, my_music_path):
        self.data = [d for d in data if 'sp_id' in d.keys()]
        self.my_music_path = my_music_path

        self.complete = None

    def get(self):
        popularities_in_data = ['popularity' in list(self.data[i].keys()) for i in range(len(self.data))]
        n_popularities = np.sum(popularities_in_data)
        self.complete = n_popularities == len(self.data)
        if self.complete:
            print('Spotify and Youtube features up to date')
        else:
            self._calculate_popularity_score()
            dump(self.data, self.my_music_path)
            print('Popularity added')

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


class Versioning:

    def __init__(self,
                 data_rm,
                 rekordbox_music_path,
                 rekordbox_music_version_check_path,
                 data_mm,
                 my_music_path,
                 added,
                 removed):

        self.data_rm = data_rm
        self.rekordbox_music_path = rekordbox_music_path
        self.rekordbox_music_version_check_path = rekordbox_music_version_check_path
        self.data_mm = data_mm
        self.my_music_path = my_music_path
        self.added = added
        self.removed = removed

        self.__create_version_check_file_if_not_exist()
        rm_vc = RekordboxMusic(rekordbox_music_version_check_path)
        self.data_rm_vc = rm_vc.get()

        self.new_version = None
        self.version = None

    def __create_version_check_file_if_not_exist(self):
        if not os.path.exists(self.rekordbox_music_version_check_path):
            shutil.copy(self.rekordbox_music_path, self.rekordbox_music_version_check_path)
            while not os.path.exists(self.rekordbox_music_version_check_path):
                time.sleep(1)

    def _replace_rekordbox_file(self):
        shutil.copy(self.rekordbox_music_path, self.rekordbox_music_version_check_path)

    def check_new_rekordbox_file(self):
        df_rm = pd.DataFrame(self.data_rm)
        df_rm_vc = pd.DataFrame(self.data_rm_vc)
        if df_rm.equals(df_rm_vc):
            self.new_version = False
        else:
            self.new_version = True
        self._replace_rekordbox_file()

    def get_version(self):
        if len(self.data_mm) == 0:
            self.version = 1
        else:
            version_cols = [col for col in pd.DataFrame(self.data_mm).columns if col.startswith('version_')]
            self.version = len(version_cols) + int(self.new_version)

    def set_version_column(self):
        if self.version > 1:
            d = {f'version_{i}': 0 for i in range(1, self.version)}
            d.update({f'version_{self.version}': 1})
            return d
        else:
            return {f'version_{self.version}': 1}

    def expand_versions_of_existing_tracks(self):
        for i in range(len(self.data_mm)):
            if self.data_mm[i]['File Name'] not in self.added:
                in_previous_version = self.data_mm[i][f'version_{self.version - 1}']
                self.data_mm[i].update({f'version_{self.version}': in_previous_version})
            if self.data_mm[i]['File Name'] in self.removed:
                self.data_mm[i].update({f'version_{self.version}': 0})

        dump(self.data_mm, self.my_music_path)


class ConnectionErrors:

    def __init__(self, all_features, data_mm, data_rm, sf, yf):

        self.all_features = all_features
        self.data_mm = data_mm.copy()
        self.data_rm = data_rm.copy()
        self.sf = sf
        self.yf = yf

        self.conn_errors = {'sp': ['sp_main_conn_error', 'sp_audio_conn_error'],
                            'yt': ['yt_search_conn_error', 'yt_video_conn_error']}

    def retry(self, d, sp_yt):
        i = find(self.data_rm, 'File Name', d['File Name'])
        if sp_yt == 'sp':
            self.sf.get(i)
            return self.sf.spotify_features
        if sp_yt == 'yt':
            self.yf.get(i)
            return self.yf.youtube_features

    def handle(self):
        any_errors = any([self.all_features[err] for err in self.conn_errors['sp'] + self.conn_errors['yt']])
        if not any_errors:
            for d in self.data_mm:
                rm_filenames = [drm['File Name'] for drm in self.data_rm]
                if d['File Name'] in rm_filenames:
                    if any([d[sce] for sce in self.conn_errors['sp']]):
                        sp_error_features = self.retry(d, 'sp')
                        d.update(sp_error_features)
                    if any([d[yce] for yce in self.conn_errors['yt']]):
                        yt_error_features = self.retry(d, 'yt')
                        d.update(yt_error_features)

        return self.data_mm
