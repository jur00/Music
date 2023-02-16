import pandas as pd
import numpy as np
import spotipy
from unidecode import unidecode
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
from music.other import Progress, make_spotify, make_youtube
from datetime import datetime


@contextmanager
def set_dir(path):
    origin = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


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


class RekordboxDataset:

    def __init__(self,
                 raw_data_dir,
                 rekordbox_filename):

        self.raw_data_dir = raw_data_dir
        self.rekordbox_filename = rekordbox_filename

        self.sp = None
        self.driver = None
        self.youtube = None
        self.rekordbox_data = None

    def clean_rekordbox_data(self):
        self.rekordbox_data = pd.read_csv(f'{self.raw_data_dir}{self.rekordbox_filename}', sep=None, header=0,
                                          encoding='utf-16',
                                          engine='python')
        self.rekordbox_data = self.rekordbox_data.rename(columns={'#': 'row'})
        self.rekordbox_data['row'] = list(range(1, (self.rekordbox_data.shape[0] + 1)))
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
            'tracktitle': self.rekordbox_data.loc[self.rekordbox_data.duplicated(subset='Track Title'), 'row'],
            'trackartist': self.rekordbox_data.loc[
                self.rekordbox_data.duplicated(subset=['Mix Name', 'Artist']), 'row']}
        self.rekordbox_data = self.rekordbox_data.reset_index(drop=True)
        if self.rekordbox_data.dtypes['BPM'] == str:
            self.rekordbox_data['BPM'] = self.rekordbox_data['BPM'].str.replace(',', '.').astype(float)

        self.rekordbox_data['track_kind'] = 'original'
        self.rekordbox_data.loc[~self.rekordbox_data['Label'].isin(['', ' ']), 'track_kind'] = 'remix'
        self.rekordbox_data.loc[~self.rekordbox_data['Album'].str.lower().isin(
            ['', ' ', 'original', 'original mix']
        ), 'track_kind'] = 'version'
        self.rekordbox_data.drop('row', axis=1, inplace=True)


class FeatureEngineering:

    def __init__(self,
                 rekordbox_data,
                 tracks_dir,
                 db,
                 sp,
                 youtube,
                 driver):

        self.rekordbox_data = rekordbox_data.reset_index(drop=True)
        self.data = self.rekordbox_data.to_dict('records')
        self.rl = range(len(self.data))
        self.tracks_dir = tracks_dir
        self.db = db
        self.db_columns = {table: self.db.get_column_names(table) for table in ['spotify', 'youtube', 'wave']}

        self.sp = sp
        self.youtube, self.driver = (youtube, driver)

    @staticmethod
    def _on_rm_error(func, path, exc_info):
        # path contains the path of the file that couldn't be removed
        # let's just assume that it's read-only and unlink it.
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)

    def _load_waveform(self, i):
        filename = self.rekordbox_data['File Name'].iloc[i]
        track_path = self.tracks_dir + filename
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            y, sr = librosa.load(track_path)

        bpm = self.rekordbox_data.loc[i, 'BPM']

        return filename, y, sr, bpm

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

        with set_dir(f'{self.tracks_dir}/output/{mapname}'):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                y_v, sr_v = librosa.load('vocals.wav', sr=None)

        with set_dir(f'{self.tracks_dir}/output/'):
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

    def _get_spotify_features(self, i):
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
                        jaccard_similarity(rekordbox_names, spotify_names))

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
                            jaccard_similarity(rekordbox_names_extra, spotify_names))

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

    def _get_wave_features(self, i):
        filename, y, sr, bpm = self._load_waveform(i)
        wave_features = self._get_librosa_features(y, sr)
        wave_features.update(self._get_chord_features(y, sr, bpm))
        wave_features.update(self._get_vocalness_feature(filename, y))

        return wave_features

    def collect_from_wavs_and_apis(self, i):
        features = self._get_spotify_features(i)
        features.update(self._get_youtube_features(i))
        features.update(self._get_wave_features(i))

        self.data[i].update(features)

    def _remove_accented_characters(self, i):
        unaccent_cols = ['Track Title', 'Mix Name', 'Artist', 'Composer', 'Label',
                         'Album', 'sp_artist', 'sp_trackname', 'yt_name']

        for uc in unaccent_cols:
            self.data[i][uc] = unidecode(self.data[i][uc])

    def _remove_sp_artist_from_name_dif(self, i):
        s_name_dif = self.data[i]['rb_sp_name_dif']
        s_artist = re.sub(r'[^A-Za-z0-9 À-ú]', '', self.data[i]['Artist']).lower().split(' ')
        if not all([a in s_name_dif for a in s_artist]):
            self.data[i]['rb_sp_name_dif'] = [s for s in s_name_dif if s not in s_artist]

    def _remove_typical_sp_yt_words_from_name_dif(self, i):
        yt_words = ['official', 'audio', 'music', 'video', 'full',
                    'enhanced', 'track', 'feat', 'ft', 'featuring',
                    'hq', 'premiere', 'records', 'hd', 'and', 'the']
        years = [str(i) for i in range(1900, datetime.now().year)]
        all_yt_words = yt_words + years
        all_sp_words = ['and', 'feat']

        self.data[i]['yt_rb_name_dif'] = [w for w in self.data[i]['yt_rb_name_dif'] if w not in all_yt_words]
        self.data[i]['sp_rb_name_dif'] = [w for w in self.data[i]['sp_rb_name_dif'] if w not in all_sp_words]

    def _remove_yt_artist_from_name_dif(self, i):
        r_artist = unidecode(re.sub(r'[^a-zA-Z0-9 À-ú]', '', self.data[i]['Artist'])).lower().split(' ')
        self.data[i]['yt_rb_name_dif'] = [w for w in self.data[i]['yt_rb_name_dif'] if w not in r_artist]

    def _get_sp_yt_dif_types(self, i):
        sp_yt = ['sp', 'yt']
        for sy in sp_yt:
            s0 = self.data[i][f'rb_{sy}_name_dif']
            s1 = self.data[i][f'{sy}_rb_name_dif']
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

    def _get_popularity_score(self, i):
        if self.data[i]['yt_publish_date'] != '':
            self.data[i]['yt_days_since_publish'] = (datetime.now() -
                                                     datetime.strptime(self.data[i]['yt_publish_date'],
                                                                       '%Y-%m-%d')).days
            self.data[i]['yt_views_per_day'] = round(
                self.data[i]['yt_views'] / self.data[i]['yt_days_since_publish'], 2)
        else:
            self.data[i]['yt_days_since_publish'] = 0
            self.data[i]['yt_views_per_day'] = 0

        if self.data[i]['sp_dif_type'] in ['same', 'other_version']:
            sp_pop = self.data[i]['sp_popularity']
        else:
            sp_pop = 0
        if self.data[i]['yt_dif_type'] in ['same', 'other_version']:
            yt_pop = self.data[i]['yt_views_per_day']
        else:
            yt_pop = 0
        if yt_pop <= 0:
            yt_pop = .001
        yt_pop = np.log(yt_pop)
        if yt_pop < 0:
            yt_pop = 0
        pop_dists = self.db.get_sp_yt_popularity_dists()
        sp_pop_dist = pop_dists['spotify']
        yt_pop_dist = pop_dists['youtube']
        yt_pop *= (max([max(sp_pop_dist), sp_pop]) / max([max(yt_pop_dist), yt_pop]))

        self.data[i]['sp_popularity'] = sp_pop
        self.data[i]['yt_popularity'] = yt_pop
        self.data[i]['popularity'] = (sp_pop + yt_pop) / 2

    def create_additional_from_collected(self, i):
        self._remove_accented_characters(i)
        self._remove_sp_artist_from_name_dif(i)
        self._remove_typical_sp_yt_words_from_name_dif(i)
        self._remove_yt_artist_from_name_dif(i)
        self._get_sp_yt_dif_types(i)
        self._get_popularity_score(i)

    @staticmethod
    def __change_list_to_string(row):
        return ' | '.join(row)

    def write_to_db_tables(self, i):
        for table in ['spotify', 'youtube', 'wave']:
            list_cols = ['sp_rb_name_dif', 'rb_sp_name_dif', 'yt_rb_name_dif', 'rb_yt_name_dif']
            values = [' | '.join(self.data[i][col]) if col in list_cols else self.data[i][col] for col in
                      self.db_columns[table]]
            float_types = [np.float16, np.float32, np.float64]
            values = tuple([float(val) if type(val) in float_types else val for val in values])
            s_vals = self.db.create_insert_values_string_part(values)
            mysql_string = f"""INSERT INTO tracks_my_{table} VALUES {s_vals}"""

            self.db.insert_row_to_db(mysql_string, values)

    def collect_create_and_write_one_by_one(self):
        progress = Progress()
        for i in self.rl:
            self.collect_from_wavs_and_apis(i)
            self.create_additional_from_collected(i)
            self.write_to_db_tables(i)
            progress.show(self.rl, i)
