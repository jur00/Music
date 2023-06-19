from joblib import dump, load
import numpy as np
import pandas as pd
import librosa
from spafe.features.gfcc import gfcc
from entropy_estimators import continuous
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis, skew, beta
import os
import warnings
import re
import subprocess
import shutil
import stat
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import tensorflow as tf
import time
from datetime import datetime
from unidecode import unidecode
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools import add_constant
import lightgbm as lgb
from BorutaShap import BorutaShap
from ml_razor import Razor
from statsmodels.regression.linear_model import OLS
import optuna
from autokeras import StructuredDataRegressor
import autokeras as ak
from keras.models import load_model
from music import helpers
import json


class DatasetMy:

    def __init__(self,
                 feature_categories,
                 features,
                 model_data_features,
                 my_raw_data_filename,
                 my_app_data_filename,
                 my_model_data_filename,
                 rekordbox_filename,
                 tracks_dir,
                 logging):
        self.my_raw_data_filename = my_raw_data_filename
        self.my_app_data_filename = my_app_data_filename
        self.my_model_data_filename = my_model_data_filename
        self.rekordbox_filename = rekordbox_filename
        self.tracks_dir = tracks_dir
        self.sp = None
        self.driver = None
        self.youtube = None
        self.feature_categories = feature_categories
        self.features = features
        self.model_data_features = model_data_features
        self.old_version = None
        self.new_version = None
        self.data = None
        self.rekordbox_data = None
        self.rb_data = None
        self.sp_yt_updates = None
        self.waveform_updates = None
        self.rl = None
        self.logging = logging

    def _on_rm_error(self, func, path, exc_info):
        # path contains the path of the file that couldn't be removed
        # let's just assume that it's read-only and unlink it.
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)

    def _make_spotify(self):
        cid = 'b95520db8f364f05ab83660503c92df5'
        secret = 'b7c268eeb5f14ef9b5286382ae2f66e0'
        ccm = SpotifyClientCredentials(client_id=cid,
                                       client_secret=secret)
        self.sp = spotipy.Spotify(client_credentials_manager=ccm)

    def _make_youtube(self):
        YT_API_Key = 'AIzaSyA5AsORnkuR2Wj0xsS2vKFtwZ5iHCgVx1Y'
        self.youtube = build('youtube', 'v3', developerKey=YT_API_Key)

        chrome_path = r'C:\SeleniumDriver\chromedriver.exe'
        browser_option = webdriver.ChromeOptions()
        browser_option.add_argument('headless')
        browser_option.add_argument('log-level = 2')
        # driver = webdriver.Chrome(chrome_path, options = browser_option)
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_option)

    def _open_data_file(self):
        self.data = load(f'{self.my_raw_data_filename}_{self.old_version}.sav')

    def _clean_rekordbox(self):
        self.rekordbox_data = pd.read_csv(self.rekordbox_filename, sep=None, header=0, encoding='utf-16',
                                          engine='python')
        self.rekordbox_data = self.rekordbox_data.rename(columns={'#': 'id'})
        self.rekordbox_data['id'] = list(range(1, (self.rekordbox_data.shape[0] + 1)))
        self.rekordbox_data['Composer'] = self.rekordbox_data['Composer'].fillna('')
        self.rekordbox_data['Album'] = self.rekordbox_data['Album'].fillna('')
        self.rekordbox_data['Label'] = self.rekordbox_data['Label'].fillna('')
        genres = ['Afro Disco', 'Balearic', 'Cosmic', 'Disco', 'Italo Disco', 'Nu Disco',
                  'Acid House', 'Deep House', 'House', 'Indie', 'Techno', 'Nostalgia', 'Old Deep House',
                  'Offbeat']
        self.rekordbox_data['Comments'] = self.rekordbox_data['Comments']. \
            str.lstrip(' /* '). \
            str.rstrip(' */'). \
            str.split(' / ')
        for g in genres:
            m = []
            for i in range(len(self.rekordbox_data['Comments'])):
                m.append(g in self.rekordbox_data['Comments'][i])

            self.rekordbox_data[g.replace(' ', '_')] = m

        del self.rekordbox_data['Comments']
        self.rekordbox_data['Rating'] = self.rekordbox_data['Rating'].str.rstrip().str.len()
        minutes = self.rekordbox_data['Time'].str.rpartition(":")[0].astype(int) * 60
        seconds = self.rekordbox_data['Time'].str.rpartition(":")[2].astype(int)
        self.rekordbox_data['rb_duration'] = (minutes + seconds) * 1000
        self.rekordbox_data = self.rekordbox_data.loc[~self.rekordbox_data.duplicated(subset='File Name'), :]
        duplications = {
            'tracktitle': self.rekordbox_data.loc[self.rekordbox_data.duplicated(subset='Track Title'), 'id'],
            'trackartist': self.rekordbox_data.loc[
                self.rekordbox_data.duplicated(subset=['Mix Name', 'Artist']), 'id']}
        self.rekordbox_data = self.rekordbox_data.reset_index(drop=True)
        if self.rekordbox_data.dtypes['BPM'] == str:
            self.rekordbox_data['BPM'] = self.rekordbox_data['BPM'].str.replace(',', '.').astype(float)

        self.rekordbox_data['track_kind'] = 'original'
        self.rekordbox_data.loc[~self.rekordbox_data['Label'].str.isin(['', ' ']), 'track_kind'] = 'remix'
        self.rekordbox_data.loc[~self.rekordbox_data['Album'].str.lower().isin(
            ['', ' ', 'original', 'original mix']
        ), 'track_kind'] = 'version'

        self.logging.info(f'Level 1 (Track Title) duplicate rekordbox ids: {duplications["tracktitle"].tolist()}')
        self.logging.info(
            f'Level 2 (Mix Name + Artist) duplicate rekordbox ids: {duplications["trackartist"].tolist()}')

    def _get_spotify_features(self, i, drift_check=False):
        artist = self.rekordbox_data['Artist'].iloc[i].split(", ")[0]
        track = self.rekordbox_data['Mix Name'].iloc[i]
        remixer = self.rekordbox_data['Composer'].iloc[i]
        original_kind = self.rekordbox_data['Album'].iloc[i]
        remix_kind = self.rekordbox_data['Label'].iloc[i]
        rekordbox_names = artist + ' ' + track + ' ' + remixer + ' ' + original_kind + ' ' + remix_kind
        rekordbox_names = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', rekordbox_names)).lower()


        connection_error = True
        spotify_features = {
            'sp_artist': '',
            'sp_trackname': '',
            'sp_id': '',
            'sp_popularity': 0,
            'sp_danceability': 0,
            'sp_energy': 0,
            'sp_key': 0,
            'sp_mode': 8,
            'sp_loudness': 0,
            'sp_speechiness': 0,
            'sp_acousticness': 0,
            'sp_instrumentalness': 0,
            'sp_valence': 0,
            'sp_tempo': 0,
            'sp_rb_name_dif': [],
            'rb_sp_name_dif': [],
            'sp_duration': 0,
            'sp_conn_error': connection_error

        }

        correct_id = []
        sp_query = artist + ' ' + track
        sp_query = sp_query.lower()

        sp_counter = 0
        while connection_error & (sp_counter < 30):
            try:
                results = self.sp.search(q=sp_query, type="track", limit=50)
                connection_error = False
            except (requests.exceptions.RequestException, spotipy.exceptions.SpotifyException):
                time.sleep(1)
                sp_counter += 1

        if connection_error:
            pass
        else:
            if len(results['tracks']['items']) > 0:
                for j, t in enumerate(results['tracks']['items']):
                    spotify_names = t['name'] + ' ' + t['artists'][0]['name']
                    spotify_names = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', spotify_names)).lower()
                    correct_id.append(
                        helpers.jaccard_similarity(rekordbox_names, spotify_names))

                if remixer == remix_kind == '' and original_kind.lower() in {'', 'original mix'}:
                    if original_kind.lower() == '':
                        original_kind_extra = 'original mix'
                    elif original_kind.lower() == 'original mix':
                        original_kind_extra = ''
                    rekordbox_names_extra = artist + ' ' + track + ' ' + remixer + ' ' + original_kind_extra + ' ' + remix_kind
                    rekordbox_names_extra = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', rekordbox_names_extra)).lower()
                    correct_id_extra = []
                    for j, t in enumerate(results['tracks']['items']):
                        spotify_names = t['name'] + ' ' + t['artists'][0]['name']
                        spotify_names = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', spotify_names)).lower()
                        correct_id_extra.append(
                            helpers.jaccard_similarity(rekordbox_names_extra, spotify_names))

                    if np.max(correct_id_extra) > np.max(correct_id):
                        best = np.argmax(correct_id_extra)
                    else:
                        best = np.argmax(correct_id)
                    rb_name_set = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', rekordbox_names_extra)).lower().split()
                else:
                    best = np.argmax(correct_id)
                    rb_name_set = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', rekordbox_names)).lower().split()

                sp_results = results['tracks']['items'][best]
                sp_name_set = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '',
                                               sp_results['artists'][0]['name'] + ' ' + sp_results[
                                                   'name'])).lower().split()
                id_sp = sp_results['id']
                audio_features = self.sp.audio_features([id_sp])
                spotify_features = {
                    'sp_artist': sp_results['artists'][0]['name'],
                    'sp_trackname': sp_results['name'],
                    'sp_id': id_sp,
                    'sp_popularity': sp_results['popularity'],
                    'sp_danceability': audio_features[0]['danceability'],
                    'sp_energy': audio_features[0]['energy'],
                    'sp_key': audio_features[0]['key'],
                    'sp_mode': audio_features[0]['mode'],
                    'sp_loudness': audio_features[0]['loudness'],
                    'sp_speechiness': audio_features[0]['speechiness'],
                    'sp_acousticness': audio_features[0]['acousticness'],
                    'sp_instrumentalness': audio_features[0]['instrumentalness'],
                    'sp_valence': audio_features[0]['valence'],
                    'sp_tempo': audio_features[0]['tempo'],
                    'sp_rb_name_dif': list(set(sp_name_set) - set(rb_name_set)),
                    'rb_sp_name_dif': list(set(rb_name_set) - set(sp_name_set)),
                    'sp_duration': sp_results['duration_ms'],
                    'sp_conn_error': connection_error

                }
            else:
                spotify_features['sp_conn_error'] = connection_error

        if drift_check:
            spotify_features = {k: v for k, v in spotify_features.items() if
                                k in ['sp_danceability', 'sp_energy', 'sp_valence']}

        return spotify_features

    def _get_youtube_features(self, i):
        connection_error = True
        artist = self.rekordbox_data['Artist'].iloc[i].split(", ")[0]
        track = self.rekordbox_data['Mix Name'].iloc[i]
        remixer = self.rekordbox_data['Composer'].iloc[i]
        original_kind = self.rekordbox_data['Album'].iloc[i]
        remix_kind = self.rekordbox_data['Label'].iloc[i]
        rekordbox_names = artist + ' ' + track + ' ' + remixer + ' ' + original_kind + ' ' + remix_kind
        rekordbox_names = unidecode(re.sub('[^a-zA-Z 0-9À-ú]+', '', rekordbox_names)).lower()

        yt_name = ''
        yt_publish_date = ''
        yt_views = 0
        yt_duration = 0
        yt_category = 0
        yt_name_set = []
        if remixer == remix_kind == '' and original_kind.lower() in {'', 'original mix'}:
            if original_kind.lower() == '':
                original_kind_extra = 'original mix'
            elif original_kind.lower() == 'original mix':
                original_kind_extra = ''
            rekordbox_names_extra = artist + ' ' + track + ' ' + remixer + ' ' + original_kind_extra + ' ' + remix_kind
            rekordbox_names_extra = unidecode(re.sub('[^a-zA-Z 0-9À-ú]+', '', rekordbox_names_extra)).lower()
            rb_name_set = unidecode(re.sub('[^a-zA-Z 0-9À-ú]+', '', rekordbox_names_extra)).lower().split()
        else:
            rb_name_set = unidecode(re.sub('[^a-zA-Z 0-9À-ú]+', '', rekordbox_names)).lower().split()

        search_query = self.rekordbox_data['Track Title'].iloc[i].replace(' ', '+').replace('&', '%26')
        search_link = 'https://www.youtube.com/results?search_query=' + search_query
        self.driver.get(search_link)
        yt_counter = 0
        while (len(self.driver.find_elements('xpath', '//*[@id="video-title"]')) == 0) & (yt_counter > 9):
            time.sleep(1)
            yt_counter += 1
        user_data = self.driver.find_elements('xpath', '//*[@id="video-title"]')
        if len(user_data) == 0:
            pass
        else:
            if user_data[0].get_attribute('href') is None:
                pass
            else:
                yt_id = user_data[0].get_attribute('href').split('/watch?v=')[1]
                cn_counter = 0
                while connection_error & (cn_counter < 30):
                    try:
                        yt_result = self.youtube.videos().list(part='snippet,statistics,contentDetails',
                                                               id=yt_id).execute()
                        connection_error = False
                    except ConnectionResetError:
                        time.sleep(1)
                        cn_counter += 1
                        connection_error = True
                if connection_error:
                    pass
                else:
                    yt_name = yt_result['items'][0]['snippet']['title']
                    yt_category = int(yt_result['items'][0]['snippet']['categoryId'])
                    yt_views = int(yt_result['items'][0]['statistics']['viewCount'])
                    yt_duration_str = yt_result['items'][0]['contentDetails']['duration']
                    yt_publish_date = yt_result['items'][0]['snippet']['publishedAt'].split('T')[0]
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

                    yt_name_set = unidecode(re.sub('[^a-zA-Z 0-9À-ú]+', '', yt_name)).lower().split()

        youtube_features = {
            'yt_name': yt_name,
            'yt_views': yt_views,
            'yt_publish_date': yt_publish_date,
            'yt_duration': yt_duration,
            'yt_category': yt_category,
            'yt_rb_name_dif': list(set(yt_name_set) - set(rb_name_set)),
            'rb_yt_name_dif': list(set(rb_name_set) - set(yt_name_set)),
            'yt_connection_error': connection_error
        }

        return youtube_features

    def _load_waveform(self, i):
        filename = self.rekordbox_data['File Name'].iloc[i]
        track_path = self.tracks_dir + filename
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            y, sr = librosa.load(track_path)

        bpm = self.rekordbox_data.loc[i, 'BPM']

        return filename, y, sr, bpm

    def _get_chord_features(self, y, sr, bpm):
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

    def _get_inst_feature(self, filename, y_o):
        with helpers.set_dir(self.tracks_dir):
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

        with helpers.set_dir(f'{self.tracks_dir}/output/{mapname}'):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                y_v, sr_v = librosa.load('vocals.wav', sr=None)

        with helpers.set_dir(f'{self.tracks_dir}/output/'):
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

        inst_feature = {'vocalness': round(vocalness, 2)}

        return inst_feature

    def _get_librosa_features(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_lib, beats = librosa.beat.beat_track(y=y, sr=sr)
        lib_duration = librosa.get_duration(y=y, sr=sr)
        beat_strength_mean = np.mean(onset_env[beats])
        offbeat_strength = np.mean(onset_env[-beats])

        S = np.abs(librosa.stft(y))
        melbands = librosa.feature.melspectrogram(S=S, n_mels=13)
        half = round(melbands.shape[0] / 2)

        features_num = {'beat_strength': np.max(onset_env[beats]),
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

    def _check_spotify_data_drift(self, n_spotify_checks, spotify_check_features):
        complete_n = max([len(self.data[i]) for i in range(len(self.data))])
        data_complete = [d for d in self.data if len(d) == complete_n]
        spotify_exist_filenames = [self.data[i]['File Name'] for i in range(len(data_complete)) if
                                   data_complete[i]['sp_id'] != '']
        rekordbox_exist_filenames = [self.rekordbox_data.loc[i, 'File Name']
                                     for i in self.rekordbox_data.index
                                     if self.rekordbox_data.loc[i, 'File Name'] in spotify_exist_filenames]
        drift_check_filenames = np.random.choice(a=rekordbox_exist_filenames, size=n_spotify_checks, replace=False)
        old_mood_features = [{'filename': self.data[i]['File Name'],
                              'sp_danceability': self.data[i]['sp_danceability'],
                              'sp_energy': self.data[i]['sp_energy'],
                              'sp_valence': self.data[i]['sp_valence']} for i in range(len(self.data))
                             if self.data[i]['File Name'] in drift_check_filenames]
        rekordbox_idxs = self.rekordbox_data.loc[
            self.rekordbox_data['File Name'].isin(drift_check_filenames)].index.tolist()
        new_mood_features = []
        for i in rekordbox_idxs:
            fn_dict = {'filename': self.rekordbox_data.loc[i, 'File Name']}
            fn_dict.update(self._get_spotify_features(i, drift_check=True))
            new_mood_features.append(fn_dict)

        drifted = []
        for f in drift_check_filenames:
            old_idx = [i for i in range(n_spotify_checks) if old_mood_features[i]['filename'] == f][0]
            new_idx = [i for i in range(n_spotify_checks) if new_mood_features[i]['filename'] == f][0]
            drift = []
            for scf in spotify_check_features:
                drift.append(old_mood_features[old_idx][scf] != new_mood_features[new_idx][scf])
            drifted.append(any(drift))
        any_data_drift = any(drifted)

        return any_data_drift

    def _sync_to_rekordbox(self):
        new_tracks = list(set(self.rb_data[i]['File Name'] for i in range(len(self.rb_data))) -
                          set(self.data[i]['File Name'] for i in range(len(self.data))))
        new_track_idxs = [i for i in range(len(self.rb_data)) if self.rb_data[i]['File Name'] in new_tracks]
        new_track_data = [self.rb_data[i] for i in new_track_idxs]

        deleted_tracks = list(set(self.data[i]['File Name'] for i in range(len(self.data))) -
                              set(self.rb_data[i]['File Name'] for i in range(len(self.rb_data))))
        deleted_track_idxs = [i for i in range(len(self.data)) if self.data[i]['File Name'] in deleted_tracks]
        self.logging.info(f'New tracks: {new_tracks}')
        self.logging.info(f'Deleted tracks: {deleted_tracks}')
        if len(deleted_track_idxs) >= len(self.data) / 3:
            self.logging.error('Too many files will be deleted.')
        assert len(deleted_track_idxs) < len(self.data) / 3, 'Too many files will be deleted.'
        self.data = [self.data[i] for i in range(len(self.data)) if i not in deleted_track_idxs]
        self.data.extend(new_track_data)
        for idx in range(len(self.data)):
            self.data[np.where([self.data[i]['File Name'] == self.rb_data[idx]['File Name']
                                for i in range(len(self.data))])[0][0]]['id'] = self.rb_data[idx]['id']
        self.data = sorted(self.data, key=lambda d: d['id'])
        if len(self.data) != len(self.rb_data):
            self.logging.error("Number of tracks in rekordbox differ from number of tracks in datafile.")
        assert len(self.data) == len(self.rb_data), \
            "Number of tracks in rekordbox differ from number of tracks in datafile."
        if any([self.data[i]['File Name'] != self.rb_data[i]['File Name'] for i in range(len(self.data))]):
            self.logging.error("File names don't match between rekordbox and datafile.")
        assert all([self.data[i]['File Name'] == self.rb_data[i]['File Name'] for i in range(len(self.data))]), \
            "File names don't match between rekordbox and datafile."

    def _define_updates(self):
        spotify_data_drift = self._check_spotify_data_drift(n_spotify_checks=5,
                                                            spotify_check_features=['sp_danceability', 'sp_energy',
                                                                                    'sp_valence'])

        to_update = []
        for idx in range(len(self.data)):
            nan_features = [k for k, v in self.data[idx].items() if v == np.nan]
            non_existing_features = list(set(self.features) - set(self.data[idx].keys()))
            update_features = list(set(nan_features + non_existing_features))
            update_categories = [f for f in self.feature_categories.keys() if
                                 any([update_features[i] in self.feature_categories[f]
                                      for i in range(len(update_features))])]
            if ('spotify' not in update_categories) | spotify_data_drift:
                if (self.data[idx]['sp_id'] == '') | self.data[idx]['sp_conn_error']:
                    update_categories.append('spotify')
            if 'youtube' not in update_categories:
                if (self.data[idx]['yt_name'] == '') | self.data[idx]['yt_connection_error']:
                    update_categories.append('youtube')
            tracktitle = self.data[idx]['Track Title']
            if 'rekordbox' in update_categories:
                self.logging.error('Rekordbox file corrupted. ' \
                                   'Check columns of Rekordbox file or ' \
                                   'check following Track Title: {}'.format(tracktitle))
            assert 'rekordbox' not in update_categories, 'Rekordbox file corrupted. ' \
                                                         'Check columns of Rekordbox file or ' \
                                                         'check following Track Title: {}'.format(tracktitle)
            if len(update_categories) > 0:
                to_update.append({'idx': idx, 'categories': update_categories})

        self.sp_yt_updates = {k: [to_update[i]['idx'] for i in range(len(to_update))
                                  if k in to_update[i]['categories']] for k in ['spotify', 'youtube']}
        self.waveform_updates = {k: [to_update[i]['idx'] for i in range(len(to_update))
                                     if k in to_update[i]['categories']] for k in
                                 ['librosa', 'chord', 'instrumentalness']}

    def _redundant_missing_files(self):
        redundant_files = list(set([f for f in os.listdir(self.tracks_dir) if f.__contains__('.')]) -
                               set(self.rekordbox_data['File Name']))
        self.logging.info(f'Remove the following files from {self.tracks_dir}: {redundant_files}')
        missing_files = list(set(self.rekordbox_data['File Name']) -
                             set([f for f in os.listdir(self.tracks_dir) if f.__contains__('.')]))
        if len(missing_files) > 0:
            self.logging.error(f'Add the following files to the Tracks directory: {missing_files}')
        assert len(missing_files) == 0, f'Add the following files to the Tracks directory: {missing_files}'

    def _missing_features(self):

        data_lengths = [len(self.data[i]) for i in range(len(self.data))]
        missing_features = list(set(self.features) -
                                set(self.data[np.where(np.array(data_lengths) == max(data_lengths))[0][0]]))

        if len(missing_features) > 0:
            self.logging.error(f"Data not complete. Following features are missing: {missing_features}")
        assert len(
            missing_features) == 0, f"Data not complete. Following features are missing: {missing_features}"

    def _update(self):
        for update_category in ['spotify', 'youtube']:
            print(update_category)
            update_idxs = self.sp_yt_updates[update_category]
            self.logging.info(f'{len(update_idxs)} {update_category} updates')
            for update_idx in update_idxs:
                print(f"{update_idxs.index(update_idx) + 1} / {len(update_idxs)}", end='\r')
                filename = self.data[update_idx]['File Name']
                update_id = self.rekordbox_data.loc[self.rekordbox_data['File Name'] == filename, 'id']
                i = update_id.index[0]
                if update_category == 'spotify':
                    self.data[update_idx].update(self._get_spotify_features(i))
                else:
                    self.data[update_idx].update(self._get_youtube_features(i))

        waveform_idxs = sorted(
            set(self.waveform_updates['librosa'] + self.waveform_updates['chord'] + self.waveform_updates[
                'instrumentalness']))
        self.logging.info(f'{len(waveform_idxs)} waveform updates')
        print('librosa, chord, instrumentalness')
        progress = helpers.Progress()
        for update_idx in waveform_idxs:
            filename = self.data[update_idx]['File Name']
            update_id = self.rekordbox_data.loc[self.rekordbox_data['File Name'] == filename, 'id']
            i = update_id.index[0]
            filename, y, sr, bpm = self._load_waveform(i)
            if update_idx in self.waveform_updates['librosa']:
                self.data[update_idx].update(self._get_librosa_features(y, sr))
            if update_idx in self.waveform_updates['chord']:
                self.data[update_idx].update(self._get_chord_features(y, sr, bpm))
            if update_idx in self.waveform_updates['instrumentalness']:
                self.data[update_idx].update(self._get_inst_feature(filename, y))

            progress.show(waveform_idxs, update_idx)

    def _remove_accented_characters(self):
        unaccent_cols = ['Track Title', 'Mix Name', 'Artist', 'Composer', 'Label',
                         'Album', 'sp_artist', 'sp_trackname', 'yt_name']
        for i in self.rl:
            for uc in unaccent_cols:
                self.data[i][uc] = unidecode(self.data[i][uc])

    def _remove_sp_artist_from_name_dif(self):
        for i in self.rl:
            s_name_dif = self.data[i]['rb_sp_name_dif']
            s_artist = re.sub(r'[^A-Za-z0-9 À-ú]', '', self.data[i]['Artist']).lower().split(' ')
            if not all([a in s_name_dif for a in s_artist]):
                self.data[i]['rb_sp_name_dif'] = [s for s in s_name_dif if s not in s_artist]

    def _remove_typical_sp_yt_words_from_name_dif(self):
        for i in self.rl:
            yt_words = ['official', 'audio', 'music', 'video', 'full',
                        'enhanced', 'track', 'feat', 'ft', 'featuring',
                        'hq', 'premiere', 'records', 'hd', 'and', 'the']
            years = [str(i) for i in range(1900, datetime.now().year)]
            all_yt_words = yt_words + years
            all_sp_words = ['and', 'feat']

            self.data[i]['yt_rb_name_dif'] = [w for w in self.data[i]['yt_rb_name_dif'] if w not in all_yt_words]
            self.data[i]['sp_rb_name_dif'] = [w for w in self.data[i]['sp_rb_name_dif'] if w not in all_sp_words]

    def _remove_yt_artist_from_name_dif(self):
        for i in self.rl:
            r_artist = unidecode(re.sub(r'[^a-zA-Z0-9 À-ú]', '', self.data[i]['Artist'])).lower().split(' ')
            self.data[i]['yt_rb_name_dif'] = [w for w in self.data[i]['yt_rb_name_dif'] if w not in r_artist]

    def _get_sp_yt_dif_types(self):
        sp_yt = ['sp', 'yt']
        for sy in sp_yt:
            for i in self.rl:
                s0 = self.data[i][f'rb_{sy}_name_dif']
                s1 = self.data[i][f'{sy}_rb_name_dif']
                if (len(s0) > 0) & (len(s1) > 0):
                    for s in s0:
                        sd = [helpers.levenshtein_distance(s, ss) for ss in s1]
                        if 1 in sd:
                            s0 = [x for x in s0 if x != s]
                            s1 = [x for x in s1 if x != s1[sd.index(1)]]

                ss = [s0, s1]
                name_dif_strings = [f'rb_{sy}_name_dif', f'{sy}_rb_name_dif']
                for s, nd in zip(ss, name_dif_strings):
                    if ('original' in s) and ('mix' in s):
                        self.data[i][nd] = [om for om in s if om not in ['original', 'mix']]
                    elif ('original' in s) and ('mix' not in s):
                        self.data[i][nd] = [o for o in s if o != 'original']
                    elif s == ['']:
                        self.data[i][nd] = []
                    else:
                        self.data[i][nd] = s

                if ((self.data[i][f'{sy}_duration'] * 1.05 > self.data[i]['rb_duration']) &
                        (self.data[i][f'{sy}_duration'] * .95 < self.data[i]['rb_duration'])):
                    self.data[i][f'{sy}_same_duration'] = True
                else:
                    self.data[i][f'{sy}_same_duration'] = False

                if (len(self.data[i][f'rb_{sy}_name_dif']) > 1) | (len(self.data[i][f'{sy}_rb_name_dif']) > 1):
                    self.data[i][f'{sy}_same_name'] = False
                else:
                    self.data[i][f'{sy}_same_name'] = True

                sy_id = 'sp_id' if sy == 'sp' else 'yt_name'
                if self.data[i][sy_id] == '':
                    self.data[i][f'{sy}_dif_type'] = 'no song'
                elif self.data[i][f'{sy}_same_name'] & self.data[i][f'{sy}_same_duration']:
                    self.data[i][f'{sy}_dif_type'] = 'same'
                elif self.data[i][f'{sy}_same_name'] & ~self.data[i][f'{sy}_same_duration']:
                    self.data[i][f'{sy}_dif_type'] = 'other version'
                else:
                    self.data[i][f'{sy}_dif_type'] = 'other song'

    def _set_camelot_pitch_mode(self):
        for i in self.rl:
            self.data[i]['Key'] = helpers.replace_keys(self.data[i]['Key'], 'tonal_to_camelot')
        for i in self.rl:
            if self.data[i]['sp_dif_type'] == 'same':
                self.data[i]['tempo'] = self.data[i]['sp_tempo']
                self.data[i]['key'] = self.data[i]['sp_key']
                self.data[i]['mode'] = self.data[i]['sp_mode']
            else:
                self.data[i]['tempo'] = self.data[i]['BPM']
                self.data[i]['key'], self.data[i]['mode'] = helpers.replace_keys(self.data[i]['Key'],
                                                                                 'camelot_to_pitch_mode')

    def _get_popularity_score(self):
        for i in self.rl:
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
        for i in self.rl:
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

        for i in self.rl:
            self.data[i]['sp_popularity'] = sp_pop_dist[i]
            self.data[i]['yt_popularity'] = yt_pop_dist[i]
            self.data[i]['popularity'] = (self.data[i]['sp_popularity'] + self.data[i]['yt_popularity']) / 2

    def _feature_engineering(self):
        self._remove_accented_characters()
        self._remove_sp_artist_from_name_dif()
        self._remove_typical_sp_yt_words_from_name_dif()
        self._remove_yt_artist_from_name_dif()
        self._get_sp_yt_dif_types()
        self._set_camelot_pitch_mode()
        self._get_popularity_score()

    def _save_raw_data_file(self):
        self.new_version = self.old_version + 1
        new_my_music_raw_version = f'{self.my_raw_data_filename}_{self.new_version}.sav'
        dump(self.data, new_my_music_raw_version)

        self.logging.info(f'Saved {new_my_music_raw_version} successfully in {os.getcwd()}')

    def _split_save_model_app_data(self):

        data_model = [{f: self.data[i][f] for f in self.model_data_features} for i in self.rl if
                      self.data[i]['sp_dif_type'] == 'same']

        app_data_features = ['File Name', 'Track Title', 'Rating', 'BPM', 'Time', 'Key', 'Afro_Disco', 'Balearic',
                             'Cosmic',
                             'Disco', 'Italo_Disco', 'Nu_Disco', 'Acid_House', 'Deep_House', 'House', 'Indie', 'Techno',
                             'Nostalgia', 'Old_Deep_House', 'Offbeat', 'vocalness', 'popularity']
        data_app = [{f: self.data[i][f] for f in app_data_features} for i in self.rl]

        dump(data_model, f'{self.my_model_data_filename}_{self.new_version}.sav')
        dump(data_app, f'{self.my_app_data_filename}_{self.new_version}.sav')

        self.logging.info(f'Saved {self.my_model_data_filename}_{self.new_version} successfully in {os.getcwd()}')
        self.logging.info(f'Saved {self.my_app_data_filename}_{self.new_version} successfully in {os.getcwd()}')

    def _remove_old_my_music_files(self):
        my_music_files = [f for f in os.listdir() if f.startswith('music_my') & (re.sub(r'[^0-9]', '', f) != '')]
        old_my_music_files = [f for f in my_music_files if int(re.sub(r'[^0-9]', '', f)) <= self.new_version - 2]
        for f in old_my_music_files:
            os.remove(f)

        self.logging.info(f'Removed {old_my_music_files} successfully from {os.getcwd()}')

    def _check_original_my_music_librosa_features(self):
        old_data = load(f'{self.my_raw_data_filename}.sav')

        librosa_same_all = []
        for i in range(len(old_data)):
            filename = old_data[i]['Bestandsnaam']
            where_bool = np.where([self.data[i]['File Name'] == filename for i in self.rl])[0]
            if where_bool.size > 0:
                data_idx = where_bool[0]
            else:
                continue
            librosa_same_values = all(
                [old_data[i][f] == self.data[data_idx][f] for f in self.feature_categories['librosa']])
            librosa_same_all.append(librosa_same_values)

        assert all(librosa_same_all), "Not all librosa values match, check new data file."
        self.logging.info(
            "All librosa features match with the original my_music file, redundant files can be removed safely.")

    def create(self):

        self._make_spotify()
        self._make_youtube()

        self.old_version = helpers.get_latest_version(file_kind='music',
                                                      music_dataset='my',
                                                      dataset_type='raw')
        self._open_data_file()

        self._clean_rekordbox()
        self.rb_data = self.rekordbox_data.to_dict('records')
        self._sync_to_rekordbox()
        self._define_updates()
        self._redundant_missing_files()
        self._update()
        self._missing_features()
        self.rl = range(len(self.data))

        self._feature_engineering()
        self._save_raw_data_file()
        self._split_save_model_app_data()
        self._check_original_my_music_librosa_features()
        self._remove_old_my_music_files()


class DatasetFull:

    def __init__(self, full_model_data_filename):
        self.full_model_data_filename = full_model_data_filename
        self.version = None
        self.data = None

    def create(self):
        music_datasets = ['my', 'random']
        versions = {md: helpers.get_latest_version(file_kind='music',
                                                   music_dataset=md,
                                                   dataset_type='model',
                                                   latest=True) for md in music_datasets}
        datas = {md: load(f'music_{md}_model_{versions[md]}.sav') for md in music_datasets}
        self.data = datas['my'] + datas['random']
        self.version = versions['my']

        dump(self.data, f'{self.full_model_data_filename}_{self.version}.sav')
        os.remove(f'{self.full_model_data_filename}_{self.version - 2}.sav')

    def check_ranges(self, feature_categories):
        o_n_l = ['old', 'new']
        data = {'old': load(f'{self.full_model_data_filename}_{self.version - 1}.sav'),
                'new': load(f'{self.full_model_data_filename}_{self.version}.sav')}
        model_data_features = feature_categories['librosa']
        model_data_features.extend(feature_categories['chord'])
        minmaxs = {o_n: {'feature': [],
                         'min': [],
                         'max': []} for o_n in o_n_l}
        for o_n in o_n_l:
            for mdf in model_data_features:
                minmaxs[o_n]['feature'].append(mdf)
                o_n_range = [data[o_n][i][mdf] for i in range(len(data[o_n]))]
                minmaxs[o_n]['min'].append(min(o_n_range))
                minmaxs[o_n]['max'].append(max(o_n_range))

        lower_min = any(np.array(minmaxs['new']['min']) < np.array(minmaxs['old']['min']))
        greater_max = any(np.array(minmaxs['new']['max']) > np.array(minmaxs['old']['max']))

        ranges_got_wider = lower_min | greater_max

        return ranges_got_wider

    def scale(self, scaled_model_data_filename,
              scaler_filename):
        df, id, targets, sp_targets, predictors, X = helpers.prepare_for_modeling(data=self.data)
        X = pd.get_dummies(data=X, columns=['key'])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        for f in sp_targets + [id]:
            df_scaled[f] = df[f]
        scaled_data = df_scaled.to_dict('records')

        dump(scaler, f'{scaler_filename}_{self.version}.sav')
        dump(scaled_data, f'{scaled_model_data_filename}_{self.version}.sav')

        os.remove(f'{scaler_filename}_{self.version - 1}.sav')
        os.remove(f'{scaled_model_data_filename}_{self.version - 1}.sav')


class LassoFeatureSelection:

    def __init__(self, scaled_model_data_filename):
        self.scaled_model_data_filename = scaled_model_data_filename
        self.important_features = []
        self.data = None

    def _compute_importance_values(self, df, X, sp_target):
        y = df[sp_target]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            logit_lasso = helpers.LogitLassoCV().fit(X, y)

        importances = np.abs(logit_lasso.coef_)

        return y, importances

    def _select_optimal_feature_set(self, df, y, df_importances, sp_target):
        information_criterions = {'AIC': [],
                                  'BIC': []}
        for i in range(300):
            important_predictors = df_importances.loc[0:i, 'feature'].to_list()
            linear_model = OLS(y, add_constant(df[important_predictors])).fit()
            information_criterions['AIC'].append(linear_model.aic)
            information_criterions['BIC'].append(linear_model.bic)
            print(i, end='\r')

        idx_mins = {ic: information_criterions[ic].index(min(information_criterions[ic])) for ic in ['AIC', 'BIC']}
        idx_min = min(idx_mins.values())
        print(f'{sp_target}: {idx_mins} features')
        df_importances = df_importances.loc[0:idx_min, :]

        return df_importances

    def _save_to_data_file(self, df, predictors, sp_targets, id, version):
        self.important_features = list(set(self.important_features))
        self.important_features = [f for f in predictors if f in self.important_features]
        keep_features = [id] + self.important_features + sp_targets
        self.data = df[keep_features].to_dict('records')
        dump(self.data, f'music_lasso_model_{version}.sav')
        lasso_model_files = [f for f in os.listdir() if f.startswith('music_lasso_model_')]
        if len(lasso_model_files) > 1:
            old_lasso_model_file = sorted(lasso_model_files)[0]
            os.remove(old_lasso_model_file)

    def execute(self):
        version = helpers.get_latest_version(file_kind='music', music_dataset='scaled', dataset_type='model',
                                             latest=True)
        data = load(f'{self.scaled_model_data_filename}_{version}.sav')
        df, id, targets, sp_targets, predictors, X = helpers.prepare_for_modeling(data=data)

        for sp_target, target in zip(sp_targets, targets):
            print(target)
            y, importances = self._compute_importance_values(df, X, sp_target)
            df_importances = helpers.create_df_importances(X.columns, importances)
            df_importances = self._select_optimal_feature_set(df, y, df_importances, sp_target)

            self.important_features += df_importances['feature'].to_list()

            dump(df_importances, f'feature_importance_lasso_{version}_{target}.sav')

        self._save_to_data_file(df, predictors, sp_targets, id, version)


class BorutaFeatureSelection:

    def __init__(self, scaled_model_data_filename):
        self.scaled_model_data_filename = scaled_model_data_filename
        self.important_features = []
        self.data = None

    def _fit_algorithm(self, df, X, sp_target):
        y = df[[sp_target]].values.ravel()
        regressor = lgb.LGBMRegressor(max_depth=5, n_jobs=os.cpu_count() - 1)
        shap_selector = BorutaShap(model=regressor,
                                   importance_measure='shap',
                                   classification=False)
        shap_selector.fit(X, y, random_state=8, n_trials=100, verbose=True)
        shap_selector.TentativeRoughFix()
        accepted = shap_selector.columns
        importances = shap_selector.feature_importance(normalize=True)[0]
        df_importances = helpers.create_df_importances(accepted, importances)

        return df_importances

    def _save_to_data_file(self, df, predictors, sp_targets, id, version):
        self.important_features = list(set(self.important_features))
        self.important_features = [f for f in predictors if f in self.important_features]
        keep_features = [id] + self.important_features + sp_targets
        self.data = df[keep_features].to_dict('records')
        dump(self.data, f'music_boruta_model_{version}.sav')
        boruta_model_files = [f for f in os.listdir() if f.startswith('music_boruta_model_')]
        if len(boruta_model_files) > 1:
            old_boruta_model_file = sorted(boruta_model_files)[0]
            os.remove(old_boruta_model_file)

    def execute(self):
        version = helpers.get_latest_version(file_kind='music', music_dataset='scaled', dataset_type='model',
                                             latest=True)
        data = load(f'{self.scaled_model_data_filename}_{version}.sav')
        df, id, targets, sp_targets, predictors, X = helpers.prepare_for_modeling(data=data)

        for target, sp_target in zip(targets, sp_targets):
            df_importances = self._fit_algorithm(df, X, sp_target)
            self.important_features += df_importances['feature'].to_list()
            dump(df_importances, f'feature_importance_boruta_{version}_{target}.sav')

        self._save_to_data_file(df, predictors, sp_targets, id, version)


class RazorFeatureSelection:

    def __init__(self, pre_selection):
        self.pre_selection = pre_selection
        self.important_features = []
        self.data = None

    def _get_pre_selection_feature_importances(self, target):
        filename = [f for f in os.listdir()
                    if f.startswith(f'feature_importance_{self.pre_selection}')
                    and f.endswith(f'{target}.sav')][0]
        version = re.sub(r'[^0-9]', '', filename)
        df_importances = load(filename)
        feature_importances = {k: v for k, v in zip(df_importances['feature'].to_list(),
                                                    df_importances['importance'].to_list())}

        return feature_importances, df_importances, version

    def _fit_algorithm(self, df, sp_target, feature_importances, df_importances):
        if self.pre_selection == 'boruta':
            estimator = lgb.LGBMRegressor(max_depth=5, n_jobs=os.cpu_count() - 1)
        else:  # if self.pre_selection == 'lasso'
            estimator = helpers.LogitLinearSklearn()
        razor = Razor(estimator=estimator, method='correlation', step=.01)
        razor.shave(df=df, target=sp_target, feature_importances=feature_importances)
        # razor.plot(plot_type='ks_analysis')
        correlation_features = razor.features_left
        correlation_feature_importances = {k: v for k, v in zip(df_importances['feature'].to_list(),
                                                                df_importances['importance'].to_list())
                                           if k in correlation_features}

        razor = Razor(estimator=estimator, method='importance', lower_bound=2)
        razor.shave(df=df, target=sp_target, feature_importances=correlation_feature_importances)
        # razor.plot(plot_type='ks_analysis')

        return razor.features_left

    def _save_to_data_file(self, df, predictors, sp_targets, id, version):
        self.important_features = list(set(self.important_features))
        self.important_features = [f for f in predictors if f in self.important_features]
        keep_features = [id] + self.important_features + sp_targets
        self.data = df[keep_features].to_dict('records')
        dump(self.data, f'music_razor_{self.pre_selection}_model_{version}.sav')
        razor_model_files = [f for f in os.listdir() if f.startswith(f'music_razor_{self.pre_selection}_model_')]
        if len(razor_model_files) > 1:
            old_razor_model_file = sorted(razor_model_files)[0]
            os.remove(old_razor_model_file)

    def execute(self, data):
        df, id, targets, sp_targets, predictors, X = helpers.prepare_for_modeling(data=data)

        for target, sp_target in zip(targets, sp_targets):
            feature_importances, df_importances, version = self._get_pre_selection_feature_importances(target)
            final_features = self._fit_algorithm(df, sp_target, feature_importances, df_importances)
            self.important_features += final_features
            df_importances = df_importances.loc[df_importances['feature'].isin(final_features), :]
            dump(df_importances, f'feature_importance_razor_{self.pre_selection}_{version}_{target}.sav')

        self._save_to_data_file(df, predictors, sp_targets, id, version)


class TrainModel:

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 feature_selection,
                 test_tracks,
                 target,
                 metric='mean_absolute_error'):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.feature_selection = feature_selection
        self.test_tracks = test_tracks
        self.target = target
        self.metric = metric

        self.version = None
        self.df = None
        self.data = None
        self.predictors = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def _load_data(self):
        self.version = helpers.get_latest_version(file_kind='music', music_dataset=self.feature_selection)
        self.data = load(f'music_{self.feature_selection}_model_{self.version}.sav')
        self.data = [d for d in self.data if d['File Name'] not in self.test_tracks]
        if self.train_dataset != 'full':
            train_data = load(f'music_my_model_{self.version}.sav')
            train_filenames = [td['File Name'] for td in train_data]
            if self.train_dataset == 'my':
                self.data = [d for d in self.data if d['File Name'] in train_filenames]
            else:  # if self.train_dataset == 'random'
                self.data = [d for d in self.data if d['File Name'] not in train_filenames]
        self.df = pd.DataFrame(self.data)
        self.predictors = load(
            f'feature_importance_{self.feature_selection}_{self.version}_{self.target}.sav').feature.to_list()
        self.df = self.df[self.predictors + [f'sp_{self.target}']]
        if 'key' in self.df.columns:
            self.df = pd.get_dummies(data=self.df, columns=['key'])

    def _prepare_data(self):
        self.X = self.df[self.predictors].values
        self.y = self.df[f'sp_{self.target}'].values

    def _fit_model(self):
        pass

    def execute(self):
        self._load_data()
        self._prepare_data()
        self._fit_model()


class TrainLinear(TrainModel):

    def _fit_model(self):
        t0 = time.time()
        estimator = helpers.LogitLinearSklearn()
        estimator.fit(self.X, self.y)
        t1 = time.time()
        time_taken = t1 - t0

        for value, descr in zip([estimator, time_taken], ['model', 'time_taken']):
            dump(value, f'.\\models\\training_{self.train_dataset}\\testing_{self.test_dataset}\\linear'
                        f'\\features_{self.feature_selection}\\{self.target}\\{descr}_{self.version}.sav')


class TrainLGBM(TrainModel):

    def __create_tracks_split(self,
                              k=10,
                              random_state=8):
        validation_tracks = [t['File Name'] for t in
                             load(f'music_{self.feature_selection}_model_{self.version}.sav')
                             if t['File Name'] not in self.test_tracks]
        np.random.seed(random_state)
        np.random.shuffle(validation_tracks)
        self.validation_tracks_split = np.array_split(validation_tracks, k)

    def _prepare_data(self):
        super()._prepare_data()
        self.__create_tracks_split()

    def __my_cross_validation(self, estimator):
        splits = len(self.validation_tracks_split)
        X_l = range(len(self.X))
        y_l = range(len(self.y))
        scores = []
        for i in range(splits):
            validation_idxs = np.where([d['File Name'] in self.validation_tracks_split[i] for d in self.data])[0]
            X_train = np.array([self.X[i] for i in X_l if i not in validation_idxs])
            X_val = np.array([self.X[i] for i in X_l if i in validation_idxs])
            y_train = np.array([self.y[i] for i in y_l if i not in validation_idxs])
            y_val = np.array([self.y[i] for i in y_l if i in validation_idxs])

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_val)
            if self.metric == 'mean_squared_error':
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            else:  # if self.metric == 'mean_absolute_error'
                score = mean_absolute_error(y_val, y_pred)

            scores.append(score)

            return scores

    def __objective(self, trial):
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
        estimator = lgb.LGBMRegressor(**params)
        scores = self.__my_cross_validation(estimator)
        metric = np.mean(scores)

        return metric

    def _fit_model(self):
        t0 = time.time()
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='minimize')
        func = lambda trial: self.__objective(trial)
        study.optimize(func, n_trials=100)

        best_value = f'{self.metric}: {study.best_value}'
        history = study.trials_dataframe(attrs=('number', 'value'))
        estimator = lgb.LGBMRegressor(**study.best_params)
        estimator.fit(self.X, self.y)
        t1 = time.time()
        time_taken = t1 - t0
        for value, descr in zip([estimator, history, best_value, time_taken],
                                ['model', 'history', 'val_score', 'time_taken']):
            dump(value, f'.\\models\\training_{self.train_dataset}\\testing_{self.test_dataset}\\lgbm'
                        f'\\features_{self.feature_selection}\\{self.target}\\{descr}_{self.version}.sav')


class TrainNeuralNetwork(TrainModel):

    def __train_val_split(self,
                          fraction=.1,
                          random_state=8):
        train_val_tracks = [t['File Name'] for t in
                            load(f'music_{self.feature_selection}_model_{self.version}.sav')
                            if t['File Name'] not in self.test_tracks]
        np.random.seed(random_state)
        np.random.shuffle(train_val_tracks)

        split_thres = int(len(train_val_tracks) * fraction)

        train_tracks = train_val_tracks[split_thres:]
        validation_tracks = train_val_tracks[:split_thres]

        self.X_train = pd.DataFrame([{k: v for k, v in d.items() if k in self.predictors}
                                     for d in self.data if d['File Name'] in train_tracks]).values
        self.X_val = pd.DataFrame([{k: v for k, v in d.items() if k in self.predictors}
                                   for d in self.data if d['File Name'] in validation_tracks]).values
        self.y_train = pd.DataFrame([{k: v for k, v in d.items() if k == f"sp_{self.target}"}
                                     for d in self.data if d['File Name'] in train_tracks]).values
        self.y_val = pd.DataFrame([{k: v for k, v in d.items() if k == f"sp_{self.target}"}
                                   for d in self.data if d['File Name'] in validation_tracks]).values

    def _prepare_data(self):
        self.__train_val_split()

    def _fit_model(self):
        t0 = time.time()
        estimator = StructuredDataRegressor(max_trials=20,
                                            loss=self.metric,
                                            overwrite=True,
                                            directory=f'.\\models\\training_{self.train_dataset}'
                                                      f'\\testing_{self.test_dataset}\\neuralnetwork'
                                                      f'\\features_{self.feature_selection}',
                                            project_name=self.target)
        estimator.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                      epochs=100)
        t1 = time.time()
        time_taken = t1 - t0
        dump(time_taken, f'.\\models\\training_{self.train_dataset}\\testing_{self.test_dataset}\\neuralnetwork'
                         f'\\features_{self.feature_selection}\\{self.target}\\time_taken_{self.version}.sav')


class CompareModels:

    def __init__(self, train_datasets,
                 test_sets,
                 train_test_match_case,
                 targets,
                 model_kinds):

        self.train_datasets = train_datasets
        self.test_sets = test_sets
        self.train_test_match_case = train_test_match_case
        self.targets = targets
        self.model_kinds = model_kinds

        self.model_results = {'train_dataset': [],
                              'test_dataset': [],
                              'feature_selection': [],
                              'target': [],
                              'test_score': [],
                              'time_taken': [],
                              'validation_score': [],
                              'history': []}

    @staticmethod
    def _get_data_and_predictors(feature_selection, target):
        data_version = helpers.get_latest_version(file_kind='music',
                                                  music_dataset=f'{feature_selection}',
                                                  dataset_type='model')
        predictor_version = helpers.get_latest_version(
            file_kind=f'feature_importance_{feature_selection}')

        data = load(f'music_{feature_selection}_model_{data_version}.sav')
        predictors = load(
            f'feature_importance_{feature_selection}_{predictor_version}_{target}.sav').feature.to_list()

        return data, predictors

    @staticmethod
    def _get_test_score(data, predictors, test_tracks, model_kind, target):
        if model_kind == 'neuralnetwork':
            model = load_model('best_model',
                               custom_objects=ak.CUSTOM_OBJECTS)
        else:
            model_version = helpers.get_latest_version(file_kind='model')
            model = load(f'model_{model_version}.sav')
        test_idxs = np.where([d['File Name'] in test_tracks for d in data])[0]
        X = pd.DataFrame(
            [{k: v for k, v in data[idx].items() if k in predictors} for idx in range(len(data))
             if
             idx in test_idxs]).values
        y = pd.DataFrame(
            [{k: v for k, v in data[idx].items() if k == f'sp_{target}'} for idx in
             range(len(data)) if
             idx in test_idxs]).values
        y_pred = model.predict(X)
        test_score = mean_absolute_error(y, y_pred)

        return test_score

    @staticmethod
    def _get_time_taken():
        time_taken_version = helpers.get_latest_version(file_kind='time_taken')
        time_taken = load(f'time_taken_{time_taken_version}.sav')

        return time_taken

    @staticmethod
    def _get_history_and_validation_score(model_kind):
        if model_kind == 'lgbm':
            history_version = helpers.get_latest_version(file_kind='history')
            history = load(f'history_{history_version}.sav')['value'].to_list()
            validation_score = min(history)
        elif model_kind == 'neuralnetwork':
            files = os.listdir()
            trial_ns = [f.split('_')[1] for f in files if f.startswith('trial_')]
            history = [json.load(open(f'.\\trial_{n}\\trial.json'))
                       ['metrics']['metrics']['val_loss']['observations'][0]['value'][0]
                       for n in trial_ns]
            validation_score = min(history)
        else:
            history = []
            validation_score = np.nan

        return history, validation_score

    def create_results(self):
        for train_dataset in self.train_datasets:
            for test_dataset in self.train_test_match_case[train_dataset]:
                test_tracks = self.test_sets[test_dataset]

                for target in self.targets:
                    for model_kind in self.model_kinds:
                        feature_selection_sets = ['razor_lasso', 'lasso'] if model_kind == 'linear' else [
                            'razor_boruta',
                            'boruta']
                        for feature_selection in feature_selection_sets:

                            self.model_results['train_dataset'].append(train_dataset)
                            self.model_results['test_dataset'].append(test_dataset)
                            self.model_results['feature_selection'].append(feature_selection)
                            self.model_results['target'].append(target)

                            data, predictors = self._get_data_and_predictors(feature_selection, target)

                            with helpers.set_dir(
                                    f'.\\models\\training_{train_dataset}\\testing_{test_dataset}\\{model_kind}'
                                    f'\\features_{feature_selection}\\{target}\\'):

                                test_score = self._get_test_score(self, data, predictors, test_tracks, model_kind, target)
                                self.model_results['test_score'].append(test_score)

                                time_taken = self._get_time_taken()
                                self.model_results['time_taken'].append(time_taken)

                                history, validation_score = self._get_history_and_validation_score(model_kind)
                                self.model_results['history'].append(history)
                                self.model_results['validation_score'].append(validation_score)

        dump(self.model_results, '.\\model_results.sav')