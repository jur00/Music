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
import logging

from unidecode import unidecode

tf.get_logger().setLevel('ERROR')


def on_rm_error(func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)


def make_spotify():
    cid = 'b95520db8f364f05ab83660503c92df5'
    secret = 'b7c268eeb5f14ef9b5286382ae2f66e0'
    ccm = SpotifyClientCredentials(client_id=cid,
                                   client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=ccm)

    return sp


def make_youtube():
    YT_API_Key = 'AIzaSyA5AsORnkuR2Wj0xsS2vKFtwZ5iHCgVx1Y'
    youtube = build('youtube', 'v3', developerKey=YT_API_Key)

    chrome_path = r'C:\SeleniumDriver\chromedriver.exe'
    browser_option = webdriver.ChromeOptions()
    browser_option.add_argument('headless')
    browser_option.add_argument('log-level = 2')
    # driver = webdriver.Chrome(chrome_path, options = browser_option)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_option)

    return youtube, driver


def get_feature_categories(filename):
    feature_categories = load(filename)
    features = []
    for v in feature_categories.values():
        features.extend(v)

    return feature_categories, features


def get_latest_data_file_version(filename):

    latest_my_music_version = [file for file in os.listdir() if file.startswith(filename)][-1]
    logging.info(f'Latest {filename}_version: {latest_my_music_version}')

    return latest_my_music_version


def open_data_file(latest_my_music_version):
    data = load(latest_my_music_version)

    return data


def clean_rekordbox(text_file):
    rekordbox_data = pd.read_csv(text_file, sep=None, header=0, encoding='utf-16', engine='python')
    rekordbox_data = rekordbox_data.rename(columns={'#': 'id'})
    rekordbox_data['id'] = list(range(1, (rekordbox_data.shape[0] + 1)))
    rekordbox_data['Composer'] = rekordbox_data['Composer'].fillna('')
    rekordbox_data['Album'] = rekordbox_data['Album'].fillna('')
    rekordbox_data['Label'] = rekordbox_data['Label'].fillna('')
    genres = ['Afro Disco', 'Balearic', 'Cosmic', 'Disco', 'Italo Disco', 'Nu Disco',
              'Acid House', 'Deep House', 'House', 'Indie', 'Techno', 'Nostalgia', 'Old Deep House',
              'Offbeat']
    rekordbox_data['Comments'] = rekordbox_data['Comments']. \
        str.lstrip(' /* '). \
        str.rstrip(' */'). \
        str.split(' / ')
    for g in genres:
        m = []
        for i in range(len(rekordbox_data['Comments'])):
            m.append(g in rekordbox_data['Comments'][i])

        rekordbox_data[g.replace(' ', '_')] = m

    del rekordbox_data['Comments']
    rekordbox_data['Rating'] = rekordbox_data['Rating'].str.rstrip().str.len()
    minutes = rekordbox_data['Time'].str.rpartition(":")[0].astype(int) * 60
    seconds = rekordbox_data['Time'].str.rpartition(":")[2].astype(int)
    rekordbox_data['rb_duration'] = (minutes + seconds) * 1000
    rekordbox_data = rekordbox_data.loc[~rekordbox_data.duplicated(subset='File Name'), :]
    duplications = {'tracktitle': rekordbox_data.loc[rekordbox_data.duplicated(subset='Track Title'), 'id'],
                    'trackartist': rekordbox_data.loc[rekordbox_data.duplicated(subset=['Mix Name', 'Artist']), 'id']}
    rekordbox_data = rekordbox_data.reset_index(drop=True)
    if rekordbox_data.dtypes['BPM'] == str:
        rekordbox_data['BPM'] = rekordbox_data['BPM'].str.replace(',', '.').astype(float)

    logging.info(f'Level 1 (Track Title) duplicate rekordbox ids: {duplications["tracktitle"].tolist()}')
    logging.info(f'Level 2 (Mix Name + Artist) duplicate rekordbox ids: {duplications["trackartist"].tolist()}')

    return rekordbox_data


def jaccard_similarity(test, real):
    intersection = set(test).intersection(set(real))
    union = set(test).union(set(real))
    return len(intersection) / len(union)


def get_spotify_features(i, drift_check=False):
    artist = rekordbox_data['Artist'].iloc[i].split(", ")[0]
    track = rekordbox_data['Mix Name'].iloc[i]
    remixer = rekordbox_data['Composer'].iloc[i]
    original_kind = rekordbox_data['Album'].iloc[i]
    remix_kind = rekordbox_data['Label'].iloc[i]
    rekordbox_names = artist + ' ' + track + ' ' + remixer + ' ' + original_kind + ' ' + remix_kind
    rekordbox_names = unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', rekordbox_names)).lower()

    if remix_kind not in ['', ' ']:
        track_kind = 'remix'
    elif original_kind.lower() not in ['', ' ', 'original', 'original mix']:
        track_kind = 'version'
    else:
        track_kind = 'original'

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
        'track_kind': '',
        'sp_conn_error': connection_error

    }

    correct_id = []
    sp_query = artist + ' ' + track
    sp_query = sp_query.lower()

    sp_counter = 0
    while connection_error & (sp_counter < 30):
        try:
            results = sp.search(q=sp_query, type="track", limit=50)
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
                                           sp_results['artists'][0]['name'] + ' ' + sp_results['name'])).lower().split()
            id_sp = sp_results['id']
            audio_features = sp.audio_features([id_sp])
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
                'track_kind': track_kind,
                'sp_conn_error': connection_error

            }
        else:
            spotify_features['sp_conn_error'] = connection_error

    if drift_check:
        spotify_features = {k: v for k, v in spotify_features.items() if
                            k in ['sp_danceability', 'sp_energy', 'sp_valence']}

    return spotify_features


def get_youtube_features(i):
    connection_error = True
    artist = rekordbox_data['Artist'].iloc[i].split(", ")[0]
    track = rekordbox_data['Mix Name'].iloc[i]
    remixer = rekordbox_data['Composer'].iloc[i]
    original_kind = rekordbox_data['Album'].iloc[i]
    remix_kind = rekordbox_data['Label'].iloc[i]
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

    search_query = rekordbox_data['Track Title'].iloc[i].replace(' ', '+').replace('&', '%26')
    search_link = 'https://www.youtube.com/results?search_query=' + search_query
    driver.get(search_link)
    yt_counter = 0
    while (len(driver.find_elements('xpath', '//*[@id="video-title"]')) == 0) & (yt_counter > 9):
        time.sleep(1)
        yt_counter += 1
    user_data = driver.find_elements('xpath', '//*[@id="video-title"]')
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
                    yt_result = youtube.videos().list(part='snippet,statistics,contentDetails', id=yt_id).execute()
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


def load_waveform(i, tracks_dir):
    filename = rekordbox_data['File Name'].iloc[i]
    track_path = tracks_dir + filename
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        y, sr = librosa.load(track_path, sr=None)
        if sr < 22050:
            y, sr = librosa.load(track_path, sr=41000)

    bpm = rekordbox_data.loc[i, 'BPM']

    return filename, y, sr, bpm


def get_chord_features(y, sr, bpm):
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


def get_inst_feature(filename, y_o):
    os.chdir('../../tracks')
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

    os.chdir('./output/' + mapname)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        y_v, sr_v = librosa.load('vocals.wav', sr=None)

    os.chdir('createMyDataset')
    shutil.rmtree(mapname, onerror=on_rm_error)

    os.chdir('/')

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


def get_librosa_features(y, sr):
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


def check_spotify_data_drift(data, n_spotify_checks, spotify_check_features):
    complete_n = max([len(data[i]) for i in range(len(data))])
    data_complete = [d for d in data if len(d) == complete_n]
    spotify_exist_filenames = [data[i]['File Name'] for i in range(len(data_complete)) if
                               data_complete[i]['sp_id'] != '']
    rekordbox_exist_filenames = [rekordbox_data.loc[i, 'File Name']
                                 for i in rekordbox_data.index
                                 if rekordbox_data.loc[i, 'File Name'] in spotify_exist_filenames]
    drift_check_filenames = np.random.choice(a=rekordbox_exist_filenames, size=n_spotify_checks, replace=False)
    old_mood_features = [{'filename': data[i]['File Name'],
                          'sp_danceability': data[i]['sp_danceability'],
                          'sp_energy': data[i]['sp_energy'],
                          'sp_valence': data[i]['sp_valence']} for i in range(len(data))
                         if data[i]['File Name'] in drift_check_filenames]
    rekordbox_idxs = rekordbox_data.loc[rekordbox_data['File Name'].isin(drift_check_filenames)].index.tolist()
    new_mood_features = []
    for i in rekordbox_idxs:
        fn_dict = {'filename': rekordbox_data.loc[i, 'File Name']}
        fn_dict.update(get_spotify_features(i, drift_check=True))
        new_mood_features.append(fn_dict)

    drifted = []
    for f in drift_check_filenames:
        old_idx = [i for i in range(n_spotify_checks) if old_mood_features[i]['filename'] == f][0]
        new_idx = [i for i in range(n_spotify_checks) if new_mood_features[i]['filename'] == f][0]
        drift = []
        for scf in spotify_check_features:
            drift.append(old_mood_features[old_idx][scf] != old_mood_features[old_idx][scf])
        drifted.append(any(drift))
    any_data_drift = any(drifted)

    return any_data_drift


def sync_to_rekordbox(data, rb_data):
    new_tracks = list(set(rb_data[i]['File Name'] for i in range(len(rb_data))) -
                      set(data[i]['File Name'] for i in range(len(data))))
    new_track_idxs = [i for i in range(len(rb_data)) if rb_data[i]['File Name'] in new_tracks]
    new_track_data = [rb_data[i] for i in new_track_idxs]

    deleted_tracks = list(set(data[i]['File Name'] for i in range(len(data))) -
                          set(rb_data[i]['File Name'] for i in range(len(rb_data))))
    deleted_track_idxs = [i for i in range(len(data)) if data[i]['File Name'] in deleted_tracks]
    logging.info(f'New tracks: {new_tracks}')
    logging.info(f'Deleted tracks: {deleted_tracks}')
    if len(deleted_track_idxs) >= len(data) / 3:
        logging.error('Too many files will be deleted.')
    assert len(deleted_track_idxs) < len(data) / 3, 'Too many files will be deleted.'
    data = [data[i] for i in range(len(data)) if i not in deleted_track_idxs]
    data.extend(new_track_data)
    for idx in range(len(data)):
        data[np.where([data[i]['File Name'] == rb_data[idx]['File Name']
                       for i in range(len(data))])[0][0]]['id'] = rb_data[idx]['id']
    data = sorted(data, key=lambda d: d['id'])
    if len(data) != len(rb_data):
        logging.error("Number of tracks in rekordbox differ from number of tracks in datafile.")
    assert len(data) == len(rb_data), \
        "Number of tracks in rekordbox differ from number of tracks in datafile."
    if any([data[i]['File Name'] != rb_data[i]['File Name'] for i in range(len(data))]):
        logging.error("File names don't match between rekordbox and datafile.")
    assert all([data[i]['File Name'] == rb_data[i]['File Name'] for i in range(len(data))]), \
        "File names don't match between rekordbox and datafile."

    return data


def define_updates(data):
    spotify_data_drift = check_spotify_data_drift(data=data,
                                                  n_spotify_checks=5,
                                                  spotify_check_features=['sp_danceability', 'sp_energy', 'sp_valence'])

    to_update = []
    for idx in range(len(data)):
        nan_features = [k for k, v in data[idx].items() if v == np.nan]
        non_existing_features = list(set(features) - set(data[idx].keys()))
        update_features = list(set(nan_features + non_existing_features))
        update_categories = [f for f in feature_categories.keys() if any([update_features[i] in feature_categories[f]
                                                                          for i in range(len(update_features))])]
        if ('spotify' not in update_categories) | spotify_data_drift:
            if (data[idx]['sp_id'] == '') | data[idx]['sp_conn_error']:
                update_categories.append('spotify')
        if 'youtube' not in update_categories:
            if (data[idx]['yt_name'] == '') | data[idx]['yt_connection_error']:
                update_categories.append('youtube')
        tracktitle = data[idx]['Track Title']
        if 'rekordbox' in update_categories:
            logging.error('Rekordbox file corrupted. ' \
                          'Check columns of Rekordbox file or ' \
                          'check following Track Title: {}'.format(tracktitle))
        assert 'rekordbox' not in update_categories, 'Rekordbox file corrupted. ' \
                                                     'Check columns of Rekordbox file or ' \
                                                     'check following Track Title: {}'.format(tracktitle)
        if len(update_categories) > 0:
            to_update.append({'idx': idx, 'categories': update_categories})

    sp_yt_updates = {k: [to_update[i]['idx'] for i in range(len(to_update))
                         if k in to_update[i]['categories']] for k in ['spotify', 'youtube']}
    waveform_updates = {k: [to_update[i]['idx'] for i in range(len(to_update))
                            if k in to_update[i]['categories']] for k in ['librosa', 'chord', 'instrumentalness']}

    return sp_yt_updates, waveform_updates


def redundant_missing_files(rekordbox_data):
    redundant_files = list(set([f for f in os.listdir('../../tracks') if f.__contains__('.')]) -
                           set(rekordbox_data['File Name']))
    logging.info(f'Remove the following files from the Tracks directory: {redundant_files}')
    missing_files = list(set(rekordbox_data['File Name']) -
                         set([f for f in os.listdir('../../tracks') if f.__contains__('.')]))
    if len(missing_files) > 0:
        logging.error(f'Add the following files to the Tracks directory: {missing_files}')
    assert len(missing_files) == 0, f'Add the following files to the Tracks directory: {missing_files}'

    return redundant_files, missing_files


def missing_features(data):

    data_lengths = [len(data[i]) for i in range(len(data))]
    missing_features = list(set(features) -
                            set(data[np.where(np.array(data_lengths) == max(data_lengths))[0][0]]))

    if len(missing_features) > 0:
        logging.error(f"Data not complete. Following features are missing: {missing_features}")
    assert len(
        missing_features) == 0, f"Data not complete. Following features are missing: {missing_features}"


    return missing_features


def update(data):
    for update_category in ['spotify', 'youtube']:
        print(update_category)
        update_idxs = sp_yt_updates[update_category]
        logging.info(f'{len(update_idxs)} {update_category} updates')
        for update_idx in update_idxs:
            print(f"{update_idxs.index(update_idx) + 1} / {len(update_idxs)}", end='\r')
            filename = data[update_idx]['File Name']
            update_id = rekordbox_data.loc[rekordbox_data['File Name'] == filename, 'id']
            i = update_id.index[0]
            if update_category == 'spotify':
                data[update_idx].update(get_spotify_features(i))
            else:
                data[update_idx].update(get_youtube_features(i))

    waveform_idxs = sorted(
        set(waveform_updates['librosa'] + waveform_updates['chord'] + waveform_updates['instrumentalness']))
    logging.info(f'{len(waveform_idxs)} waveform updates')
    print('librosa, chord, instrumentalness')
    progress = Progress()
    for update_idx in waveform_idxs:
        filename = data[update_idx]['File Name']
        update_id = rekordbox_data.loc[rekordbox_data['File Name'] == filename, 'id']
        i = update_id.index[0]
        filename, y, sr, bpm = load_waveform(i, tracks_dir)
        if update_idx in waveform_updates['librosa']:
            data[update_idx].update(get_librosa_features(y, sr))
        if update_idx in waveform_updates['chord']:
            data[update_idx].update(get_chord_features(y, sr, bpm))
        if update_idx in waveform_updates['instrumentalness']:
            data[update_idx].update(get_inst_feature(filename, y))

        progress.show(waveform_idxs, update_idx)

    return data


def levenshteinDistance(s1, s2):
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


def remove_accented_characters(data):
    unaccent_cols = ['Track Title', 'Mix Name', 'Artist', 'Composer', 'Label',
                     'Album', 'sp_artist', 'sp_trackname', 'yt_name']
    for i in rl:
        for uc in unaccent_cols:
            data[i][uc] = unidecode(data[i][uc])

    return data


def remove_sp_artist_from_name_dif(data):
    for i in rl:
        s_name_dif = data[i]['rb_sp_name_dif']
        s_artist = re.sub(r'[^A-Za-z0-9 À-ú]', '', data[i]['Artist']).lower().split(' ')
        if not all([a in s_name_dif for a in s_artist]):
            data[i]['rb_sp_name_dif'] = [s for s in s_name_dif if s not in s_artist]

    return data


def remove_typical_sp_yt_words_from_name_dif(data):
    for i in rl:
        yt_words = ['official', 'audio', 'music', 'video', 'full',
                    'enhanced', 'track', 'feat', 'ft', 'featuring',
                    'hq', 'premiere', 'records', 'hd', 'and', 'the']
        years = [str(i) for i in range(1900, datetime.now().year)]
        all_yt_words = yt_words + years
        all_sp_words = ['and', 'feat']

        data[i]['yt_rb_name_dif'] = [w for w in data[i]['yt_rb_name_dif'] if w not in all_yt_words]
        data[i]['sp_rb_name_dif'] = [w for w in data[i]['sp_rb_name_dif'] if w not in all_sp_words]

    return data


def remove_yt_artist_from_name_dif(data):
    for i in rl:
        r_artist = unidecode(re.sub(r'[^a-zA-Z0-9 À-ú]', '', data[i]['Artist'])).lower().split(' ')
        data[i]['yt_rb_name_dif'] = [w for w in data[i]['yt_rb_name_dif'] if w not in r_artist]

    return data


def get_sp_yt_dif_types(data):
    sp_yt = ['sp', 'yt']
    for sy in sp_yt:
        for i in rl:
            s0 = data[i][f'rb_{sy}_name_dif']
            s1 = data[i][f'{sy}_rb_name_dif']
            if (len(s0) > 0) & (len(s1) > 0):
                for s in s0:
                    sd = [levenshteinDistance(s, ss) for ss in s1]
                    if 1 in sd:
                        s0 = [x for x in s0 if x != s]
                        s1 = [x for x in s1 if x != s1[sd.index(1)]]

            ss = [s0, s1]
            name_dif_strings = [f'rb_{sy}_name_dif', f'{sy}_rb_name_dif']
            for s, nd in zip(ss, name_dif_strings):
                if ('original' in s) and ('mix' in s):
                    data[i][nd] = [om for om in s if om not in ['original', 'mix']]
                elif ('original' in s) and ('mix' not in s):
                    data[i][nd] = [o for o in s if o != 'original']
                elif s == ['']:
                    data[i][nd] = []
                else:
                    data[i][nd] = s

            if ((data[i][f'{sy}_duration'] * 1.05 > data[i]['rb_duration']) &
                    (data[i][f'{sy}_duration'] * .95 < data[i]['rb_duration'])):
                data[i][f'{sy}_same_duration'] = True
            else:
                data[i][f'{sy}_same_duration'] = False

            if (len(data[i][f'rb_{sy}_name_dif']) > 1) | (len(data[i][f'{sy}_rb_name_dif']) > 1):
                data[i][f'{sy}_same_name'] = False
            else:
                data[i][f'{sy}_same_name'] = True

            sy_id = 'sp_id' if sy == 'sp' else 'yt_name'
            if data[i][sy_id] == '':
                data[i][f'{sy}_dif_type'] = 'no song'
            elif data[i][f'{sy}_same_name'] & data[i][f'{sy}_same_duration']:
                data[i][f'{sy}_dif_type'] = 'same'
            elif data[i][f'{sy}_same_name'] & ~data[i][f'{sy}_same_duration']:
                data[i][f'{sy}_dif_type'] = 'other version'
            else:
                data[i][f'{sy}_dif_type'] = 'other song'

    return data


def replace_keys(value, kind = 'tonal_to_camelot'):
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


def set_camelot_pitch_mode(data):
    for i in rl:
        data[i]['Key'] = replace_keys(data[i]['Key'], 'tonal_to_camelot')
    for i in rl:
        if data[i]['sp_dif_type'] == 'same':
            data[i]['tempo'] = data[i]['sp_tempo']
            data[i]['key'] = data[i]['sp_key']
            data[i]['mode'] = data[i]['sp_mode']
        else:
            data[i]['tempo'] = data[i]['BPM']
            data[i]['key'], data[i]['mode'] = replace_keys(data[i]['Key'], 'camelot_to_pitch_mode')

    return data


def get_popularity_score(data):
    for i in rl:
        if data[i]['yt_publish_date'] != '':
            data[i]['yt_days_since_publish'] = (datetime.now() -
                                                datetime.strptime(data[i]['yt_publish_date'], '%Y-%m-%d')).days
            data[i]['yt_views_per_day'] = round(data[i]['yt_views'] / data[i]['yt_days_since_publish'], 2)
        else:
            data[i]['yt_days_since_publish'] = 0
            data[i]['yt_views_per_day'] = 0
    sp_pop_dist = []
    yt_pop_dist = []
    for i in rl:
        if data[i]['sp_dif_type'] in ['same', 'other_version']:
            sp_pop_dist.append(data[i]['sp_popularity'])
        else:
            sp_pop_dist.append(0)
        if data[i]['yt_dif_type'] in ['same', 'other_version']:
            yt_pop_dist.append(data[i]['yt_views_per_day'])
        else:
            yt_pop_dist.append(0)
    yt_pop_dist = np.array(yt_pop_dist)
    yt_pop_dist[yt_pop_dist <= 0] = .001
    yt_pop_dist = np.log(yt_pop_dist)
    yt_pop_dist[yt_pop_dist < 0] = 0
    yt_pop_dist *= (max(sp_pop_dist) / max(yt_pop_dist))

    for i in rl:
        data[i]['sp_popularity'] = sp_pop_dist[i]
        data[i]['yt_popularity'] = yt_pop_dist[i]
        data[i]['popularity'] = (data[i]['sp_popularity'] + data[i]['yt_popularity']) / 2

    return data


def save_raw_data_file(data, latest_my_music_version, data_filename):
    version_number = int(latest_my_music_version.split('.')[0].split(f'{data_filename}_')[1]) + 1
    new_my_music_raw_version = f'{data_filename}_{version_number}.sav'
    dump(data, open(new_my_music_raw_version, 'wb'))

    logging.info(f'Saved {new_my_music_raw_version} successfully in {os.getcwd()}')

    return version_number


def split_save_model_app_data(data, feature_categories, latest_my_music_version):
    model_data_features = ['File Name']
    model_data_features.extend(feature_categories['librosa'])
    model_data_features.extend(feature_categories['chord'])
    model_data_features.extend(['key', 'mode', 'tempo'])
    model_data_features.extend(['sp_danceability', 'sp_energy', 'sp_valence'])
    data_model = [{f: data[i][f] for f in model_data_features} for i in rl if data[i]['sp_dif_type'] == 'same']

    app_data_features = ['File Name', 'Track Title', 'Rating', 'BPM', 'Time', 'Key', 'Afro_Disco', 'Balearic', 'Cosmic',
                         'Disco', 'Italo_Disco', 'Nu_Disco', 'Acid_House', 'Deep_House', 'House', 'Indie', 'Techno',
                         'Nostalgia', 'Old_Deep_House', 'Offbeat', 'vocalness', 'popularity']
    data_app = [{f: data[i][f] for f in app_data_features} for i in rl]

    dump(data_model, open(f'my_music_model_{latest_my_music_version}.sav', 'wb'))
    dump(data_app, open(f'my_music_app_{latest_my_music_version}.sav', 'wb'))

    logging.info(f'Saved my_music_model_{latest_my_music_version} successfully in {os.getcwd()}')
    logging.info(f'Saved my_music_app_{latest_my_music_version} successfully in {os.getcwd()}')


def remove_old_my_music_files(version_number):
    my_music_files = [f for f in os.listdir() if f.startswith('my_music_') & (re.sub(r'[^0-9]', '', f) != '')]
    old_my_music_files = [f for f in my_music_files if int(re.sub(r'[^0-9]', '', f)) <= version_number - 2]
    for f in old_my_music_files:
        os.remove(f)

    logging.info(f'Removed {old_my_music_files} successfully from {os.getcwd()}')


def check_original_my_music_librosa_features(data):
    old_data = load('my_music_raw.sav')

    librosa_same_all = []
    for i in range(len(old_data)):
        filename = old_data[i]['Bestandsnaam']
        where_bool = np.where([data[i]['File Name'] == filename for i in rl])[0]
        if where_bool.size > 0:
            data_idx = where_bool[0]
        else:
            continue
        librosa_same_values = all([old_data[i][f] == data[data_idx][f] for f in feature_categories['librosa']])
        librosa_same_all.append(librosa_same_values)

    assert all(librosa_same_all), "Not all librosa values match, check new data file."
    logging.info("All librosa features match with the original my_music file, redundant files can be removed safely.")





feature_categories_filename = '../../feature_categories_my.sav'
data_filename = 'my_music_raw'
rekordbox_filename = '../../rekordbox_file.txt'
tracks_dir = '../../tracks/my'




sp = make_spotify()
youtube, driver = make_youtube()

feature_categories, features = get_feature_categories(feature_categories_filename)
latest_my_music_version = get_latest_data_file_version(data_filename)
data = open_data_file(latest_my_music_version)

rekordbox_data = clean_rekordbox(rekordbox_filename)
rb_data = rekordbox_data.to_dict('records')
data = sync_to_rekordbox(data, rb_data)
sp_yt_updates, waveform_updates = define_updates(data)
redundant_files, missing_files = redundant_missing_files(rekordbox_data)
data = update(data)
missing_features = missing_features(data)
rl = range(len(data))
data = remove_accented_characters(data)
data = remove_sp_artist_from_name_dif(data)
data = remove_typical_sp_yt_words_from_name_dif(data)
data = remove_yt_artist_from_name_dif(data)
data = get_sp_yt_dif_types(data)
data = set_camelot_pitch_mode(data)
data = get_popularity_score(data)
version_number = save_raw_data_file(data, latest_my_music_version, data_filename)
split_save_model_app_data(data, feature_categories, version_number)
check_original_my_music_librosa_features(data)
remove_old_my_music_files(version_number)
