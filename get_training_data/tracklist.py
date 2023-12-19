from config import Config

import pandas as pd
import numpy as np
from joblib import load, dump
from tqdm import tqdm
from unidecode import unidecode
import re

from base.helpers import levenshtein_distance
from base.connect import SpotifyConnect
from base.spotify_youtube import (get_spotify_recommendations, get_spotify_track, get_spotify_artist_top_tracks,
                                  get_spotify_audio_features)


class SelectTrainingTracklist:

    def __init__(self):
        self.n_tracks = Config.n_tracks_per_day
        self.my_music_path = Config.my_music_path
        self.recommendations_path = Config.training_recommendations_path
        self.artist_tracks_path = Config.training_artist_tracks_path

        self.sp = SpotifyConnect().sp

        self.keep_init_cols = ['File Name', 'sp_id', 'sp_artist', 'sp_trackname', 'sp_popularity']
        self.keep_final_cols = ['File Name', 'sp_id', 'sp_artist', 'sp_trackname', 'track_id', 'track_artists',
                                'track_name', 'track_duration_ms', 'track_popularity', 'type']

        self.path_df = (self.recommendations_path
                        if self.__class__.__name__ == 'Recommendations'
                        else self.artist_tracks_path)

        self.df_tracklist = None

        self.df = None
        self.df_tracklist_base = None
        self.load()

        self.n_versions = None
        self.get_n_versions()

        self.in_last_version_mask = None
        self.correct_track_mask = None
        self.to_do_mask = None
        self.create_base_masks()

        self.n_tracks_to_go = None

    def load(self):
        self.df = pd.DataFrame(load(self.my_music_path))
        self.df_tracklist_base = load(self.path_df)

    def get_n_versions(self):
        version_cols = [col for col in self.df.columns if col.startswith('version')]
        self.n_versions = len(version_cols)

    def create_base_masks(self):
        self.in_last_version_mask = self.df[f'version_{self.n_versions}'] == 1
        self.correct_track_mask = self.df['sp_dif_type'] == 'same'

    def select_n_tracks(self):
        self.n_tracks_to_go = self.df_tracklist.shape[0]
        print(f'{self.__class__.__name__}: {self.n_tracks_to_go} tracks to go')
        if self.n_tracks_to_go == 0:
            print(f'For all tracks {self.__class__.__name__} are done.')
        elif self.n_tracks_to_go >= self.n_tracks:
            self.df_tracklist = self.df_tracklist.iloc[:self.n_tracks]
        else:
            self.df_tracklist = self.df_tracklist.iloc[:self.n_tracks_to_go]

    @staticmethod
    def extract_track_properties(tracks):
        rc_rl = range(len(tracks))
        rc_name = [tracks[i]['name'] for i in rc_rl]
        rc_artists = [', '.join([a['name'] for a in tracks[i]['artists']]) for i in rc_rl]
        rc_id = [tracks[i]['id'] for i in rc_rl]
        rc_popularity = [tracks[i]['popularity'] for i in rc_rl]
        rc_duration_ms = [tracks[i]['duration_ms'] for i in rc_rl]

        return rc_name, rc_artists, rc_id, rc_popularity, rc_duration_ms

    def write_training_df(self):
        df_tracklist_total = pd.concat([self.df_tracklist, self.df_tracklist_base]).reset_index(drop=True)
        dump(df_tracklist_total, self.path_df)


class Recommendations(SelectTrainingTracklist):

    def __init__(self):
        super().__init__()

        self.file_names_done = None
        self.list_file_names_done()

    def list_file_names_done(self):
        self.file_names_done = self.df_tracklist_base['File Name'].unique()

    def filter_df(self):
        self.to_do_mask = ~self.df['File Name'].isin(self.file_names_done)
        self.df_tracklist = self.df.loc[self.correct_track_mask &
                                        self.in_last_version_mask &
                                        self.to_do_mask, self.keep_init_cols]

    def decide_n_recommendations_per_track(self):
        max_sp_popularity = self.df_tracklist['sp_popularity'].max()
        cuts = np.unique(np.linspace(0, max_sp_popularity + 1, 21).astype(int))
        cut_labels = np.unique(np.linspace(1, 20, len(cuts) - 1).astype(int))
        self.df_tracklist['limit'] = 0
        for lower, upper, label in zip(cuts[:-1], cuts[1:], cut_labels):
            n_recommendations_mask = self.df_tracklist['sp_popularity'].between(lower, upper, inclusive='left')
            self.df_tracklist.loc[n_recommendations_mask, 'limit'] = label

    @staticmethod
    def _get_recommendations(row, sp):
        return get_spotify_recommendations(sp, row['sp_id'], row['limit'])

    def collect_recommendations(self):
        print('Collecting recommendations')
        tqdm.pandas()
        (self.df_tracklist['recommendations'], self.df_tracklist['sp_conn_error']) = zip(
            *self.df_tracklist.progress_apply(self._get_recommendations, args=(self.sp,), axis=1)
        )

    def collect_recommendation_properties(self):
        (self.df_tracklist['track_name'], self.df_tracklist['track_artists'],
         self.df_tracklist['track_id'], self.df_tracklist['track_popularity'],
         self.df_tracklist['track_duration_ms']) = zip(
            *self.df_tracklist['recommendations'].apply(self.extract_track_properties)
        )
        self.df_tracklist['type'] = 'recommendation'

    def explode_recommendations(self):
        self.df_tracklist = self.df_tracklist[self.keep_final_cols].explode(
            [col for col in self.df_tracklist.columns if col.startswith('track_')]
        )

    def run(self):
        self.filter_df()

        self.select_n_tracks()

        if self.df_tracklist.shape[0] == 0:
            return None

        self.decide_n_recommendations_per_track()

        self.collect_recommendations()

        self.collect_recommendation_properties()

        self.explode_recommendations()

        self.write_training_df()


class ArtistTracks(SelectTrainingTracklist):

    def __init__(self):
        super().__init__()

        self.artists_done = None
        self.list_artists_done()

    def list_artists_done(self):
        self.artists_done = self.df_tracklist_base['sp_artist'].unique()

    @staticmethod
    def _get_end_artist(row):
        if row['Composer'] != '':
            return row['Composer']
        else:
            return row['Artist']

    @staticmethod
    def _compare_end_spotify_artist(row):
        sp_lower = row['sp_artist'].lower()
        end_lower = [ea.lower() for ea in row['end_artist']]
        if sp_lower in end_lower:
            return 'same'
        else:
            dists = [levenshtein_distance(sp_lower, el) for el in end_lower]
            if 1 in dists:
                return 'same enough'
            else:
                return 'different'

    def filter_df(self):
        self.df['end_artist'] = self.df.apply(self._get_end_artist, axis=1).str.split(', ')
        self.df['sp_end_artist_dif_type'] = self.df.apply(self._compare_end_spotify_artist, axis=1)

        correct_artist_mask = self.correct_track_mask & (self.df['sp_end_artist_dif_type'] != 'different')

        self.to_do_mask = ~self.df['sp_artist'].isin(self.artists_done)
        self.df_tracklist = self.df.loc[correct_artist_mask & self.in_last_version_mask & self.to_do_mask,
                                        self.keep_init_cols].reset_index(drop=True)

    @staticmethod
    def _get_artists_id(sp_id, sp):
        track, sp_conn_error = get_spotify_track(sp, sp_id)

        return [ar['id'] for ar in track['artists']]

    def collect_artist_ids(self):
        print('Collecting artist ids')
        tqdm.pandas()
        self.df_tracklist['artists_id'] = self.df_tracklist['sp_id'].progress_apply(
            self._get_artists_id, args=(self.sp,)
        )

    def explode_artists(self):
        self.df_tracklist = self.df_tracklist.explode('artists_id')

    @staticmethod
    def _get_top_tracks(artist_id, sp):
        return get_spotify_artist_top_tracks(sp, artist_id)

    def collect_top_tracks_per_artist(self):
        print('Collecting top tracks per artist')
        tqdm.pandas()
        self.df_tracklist['top_tracks'], self.df_tracklist['sp_conn_error'] = zip(
            *self.df_tracklist['artists_id'].progress_apply(self._get_top_tracks, args=(self.sp,))
        )

    def collect_track_properties(self):
        (self.df_tracklist['track_name'], self.df_tracklist['track_artists'], self.df_tracklist['track_id'],
         self.df_tracklist['track_popularity'], self.df_tracklist['track_duration_ms']) = zip(
            *self.df_tracklist['top_tracks'].apply(self.extract_track_properties)
        )
        self.df_tracklist['type'] = 'top_track'

    def explode_artist_top_tracks(self):
        self.df_tracklist = self.df_tracklist[self.keep_final_cols].explode(
            [col for col in self.df_tracklist.columns if col.startswith('track_')]
        )

    def run(self):

        self.filter_df()

        self.select_n_tracks()

        if self.df_tracklist.shape[0] == 0:
            return None

        self.collect_artist_ids()

        self.explode_artists()

        self.collect_top_tracks_per_artist()

        self.collect_track_properties()

        self.explode_artist_top_tracks()

        self.write_training_df()


class AssembleCleanTracklist:

    def __init__(self):
        self.df_artists_top = None
        self.df_recommendations = None
        self.df_my_music = None

        self.sp = SpotifyConnect().sp

        self.df = None
        self.df_base = None

    def load(self):
        self.df_artists_top = load(Config.training_artist_tracks_path)
        self.df_recommendations = load(Config.training_recommendations_path)
        self.df_my_music = pd.DataFrame(load(Config.my_music_path))
        self.df_base = load(Config.training_tracklist_path)

    def assemble_artist_top_and_recommendations_dfs(self):
        self.df = (pd
                   .concat([self.df_artists_top, self.df_recommendations])
                   .drop_duplicates(subset=['track_artists', 'track_name'])
                   .dropna()
                   .reset_index(drop=True)
                   )

    @staticmethod
    def add_track_artist_comb_col(xdf, artist_col, trackname_col):
        xdf = xdf.assign(track_artist_name=xdf[artist_col] + ' - ' + xdf[trackname_col])
        xdf['track_artist_name'] = (
            xdf['track_artist_name'].apply(
                lambda x: unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', str(x)))
            ).str.lower())
        return xdf

    def remove_my_tracks_from_tracklist(self):
        self.df_my_music = self.df_my_music.pipe(self.add_track_artist_comb_col, 'sp_artist', 'sp_trackname')
        self.df = self.df.pipe(self.add_track_artist_comb_col, 'track_artists', 'track_name')

        self.df = self.df.loc[~self.df['track_artist_name'].isin(self.df_my_music['track_artist_name'])]

    def remove_existing_tracks(self):
        self.df = self.df.loc[~self.df['track_artist_name'].isin(self.df_base['track_artist_name'])]

    @staticmethod
    def get_audio_features(track_id, sp):
        audio_features, _ = get_spotify_audio_features(sp, track_id)

        danceability = audio_features[0]['danceability']
        energy = audio_features[0]['energy']
        valence = audio_features[0]['valence']
        instrumentalness = audio_features[0]['instrumentalness']
        acousticness = audio_features[0]['acousticness']
        key = audio_features[0]['key']
        mode = audio_features[0]['mode']

        return danceability, energy, valence, instrumentalness, acousticness, key, mode

    def collect_audio_features(self):
        print('Collecting audio features')
        tqdm.pandas()
        (self.df['track_danceability'], self.df['track_energy'], self.df['track_valence'],
         self.df['track_instrumentalness'], self.df['track_acousticness'], self.df['track_key'],
         self.df['track_mode']) = zip(*self.df['track_id'].progress_apply(self.get_audio_features, args=(self.sp,)))

    def write(self):
        df_total = (pd
                    .concat([self.df_base, self.df])
                    .drop_duplicates(subset='track_id')
                    .reset_index(drop=True))
        df_total['track_artist_name'] = df_total['track_artist_name'].str.replace('  ', ' ')
        dump(df_total, Config.training_tracklist_path)

    def run(self):
        self.load()

        self.assemble_artist_top_and_recommendations_dfs()

        self.remove_my_tracks_from_tracklist()

        self.remove_existing_tracks()

        print(f'{self.df.shape[0]} tracks to go')
        self.df = self.df.sample(n=5)

        self.collect_audio_features()

        self.write()


rc = Recommendations()
rc.run()

at = ArtistTracks()
at.run()

act = AssembleCleanTracklist()
act.run()

from joblib import load, dump
import pandas as pd
from config import Config
from tkinter import Tk
import pyperclip
import pyautogui as pyauto
import time
from unidecode import unidecode
import re
from base.helpers import string_similarity

tk = Tk()

df = load(Config.training_tracklist_path)

def create_compare_names(df, i):
    artist_track_name = df.loc[i, 'track_artist_name']
    track_name = unidecode(
        re.sub(r'[^a-zA-Z 0-9À-ú]+', '', str(df.loc[i, 'track_name']))
    ).lower()
    duration_ms = df.loc[i, 'track_duration_ms']
    return artist_track_name, track_name, duration_ms

def search_soulseek(track):
    # for downloading via soulseek, open soulseek in full-screen and navigate to the 'Manual Search' section. the
    # location parameters of the mouse-clicks depend on the screen resulution. I use a standard (1920x1080) resolution.
    time.sleep(5)
    pyperclip.copy(track)
    pyauto.click((53, 150))
    pyauto.hotkey("ctrl", "v")
    pyauto.press('enter')

    # before you screen all results, let the system sleep for 6 seconds minimum, in order to have a static search list.
    time.sleep(10)  # 15

def gather_soulseek_result_info(Y, kind):
    feature_x_coord = {
        'file': 900,
        'size': 1600,
        'attributes': 1750
    }
    x = feature_x_coord[kind]
    pyauto.click((x, Y))
    pyauto.hotkey('ctrl', 'c')
    time.sleep(.1)
    return tk.clipboard_get()

def create_soulseek_df(Y=254, row_height=13, nrows=56):

    empty_row_count = 0
    soulseek_results = []
    for i in range(nrows):
        soulseek_info = {result_info: gather_soulseek_result_info(Y, result_info) for result_info in
                         ['file', 'size', 'attributes']}
        empty_row = all(v == soulseek_info['file'] for v in soulseek_info.values())
        if empty_row:
            empty_row_count += 1
            if empty_row_count == 3:
                break
        else:
            soulseek_info.update({'Y': Y})
            soulseek_results.append(soulseek_info)
            empty_row_count = 0
        Y += row_height

    return pd.DataFrame(soulseek_results)

def filter_soulseek_df(xdf):
    less_than_hour_mask = xdf['attributes'].apply(lambda x: 'h' not in x)
    normal_size_mask = xdf['size'].apply(lambda x: ',' in x)
    mp3_mask = xdf['file'].str.endswith('.mp3')
    attributes_mask = xdf['attributes'].apply(lambda x: ',' in x)
    keep_mask = (less_than_hour_mask & normal_size_mask & mp3_mask & attributes_mask)

    return xdf.loc[keep_mask].reset_index(drop=True)

def clean_soulseek_df(xdf):
    xdf = (
        xdf
        .assign(
            **xdf['attributes'].str.split(', ', n=1, expand=True).rename(columns={0: 'bitrate', 1: 'duration_ms'})
        )
        .assign(
            bitrate=lambda xxdf: xxdf['bitrate'].str.replace('kbps', '').astype(int),
            duration_ms=lambda xxdf: xxdf['duration_ms'].str.replace('s', '').str.split('m')
        )
        .assign(
            duration_ms=lambda xxdf: xxdf['duration_ms'].apply(lambda x: 1000 * ((int(x[0]) * 60) + int(x[1]))),
            file_clean=lambda xxdf: xxdf['file']
                .str.rsplit('.', 1)
                .str[0]
                .str.replace('_', ' ')
                .str.replace('-', ' ')
                .str.replace('  ', ' ')
                .apply(lambda x: unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', str(x))))
                .str.lower()
        )
        .drop(columns='attributes')
    )
    return xdf

def compare_names(xdf, artist_track_name, track_name):
    artist_track_name_original = artist_track_name + ' original mix'
    track_name_original = track_name + ' original mix'
    comparisons = [artist_track_name, artist_track_name_original, track_name, track_name_original]
    similarity_columns = []
    for i, comparison in enumerate(comparisons):
        similarity_column = f'name_similarity_{i}'
        xdf = xdf.assign(**{similarity_column: xdf['file_clean'].apply(
            lambda x: string_similarity(x, comparison))
        })
        similarity_columns.append(similarity_column)

    xdf = (
        xdf
        .assign(name_similarity=xdf[similarity_columns].max(axis=1))
        .drop(columns=similarity_columns)
    )
    return xdf

def compare_durations(xdf, duration_ms):
    longer = xdf['duration_ms'] >= duration_ms
    shorter = xdf['duration_ms'] < duration_ms
    duration_similarity = pd.concat([
        xdf.loc[shorter, 'duration_ms'] / duration_ms,
        duration_ms / xdf.loc[longer, 'duration_ms']
    ]).sort_index()
    xdf = xdf.assign(duration_similarity=duration_similarity)
    return xdf

def keep_only_high_similarity_scores(xdf, score):
    score_mask = xdf['similarity_score'] > score
    return xdf.loc[score_mask].reset_index(drop=True)

t0 = time.time()
i = 3
artist_track_name, track_name, duration_ms = create_compare_names(df, i)

search_soulseek(artist_track_name)

df_qt = create_soulseek_df()

df_soulseek = (
    df_qt
    .pipe(filter_soulseek_df)
)

if df_soulseek.shape[0] > 0:
    df_soulseek = (
        df_soulseek
        .pipe(clean_soulseek_df)
        .pipe(compare_names, artist_track_name, track_name)
        .pipe(compare_durations, duration_ms)
        .assign(similarity_score=lambda xdf: (xdf['name_similarity'] + xdf['duration_similarity'] * 2) / 3)
        .pipe(keep_only_high_similarity_scores, .9)
    )
if df_soulseek.shape[0] > 0:
    df_soulseek.loc[0]

t1 = time.time()
tdif = (t1-t0)
