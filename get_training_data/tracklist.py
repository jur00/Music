from config import Config

import pandas as pd
import numpy as np
from joblib import load, dump
from tqdm import tqdm

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
        # self.df = self.df.sample(n=20, random_state=8)

        self.df_tracklist_base = load(self.path_df)

    def get_n_versions(self):
        version_cols = [col for col in self.df.columns if col.startswith('version')]
        self.n_versions = len(version_cols)

    def create_base_masks(self):
        self.in_last_version_mask = self.df[f'version_{self.n_versions}'] == 1
        self.correct_track_mask = self.df['sp_dif_type'] == 'same'

    def select_n_tracks(self):
        self.n_tracks_to_go = self.df_tracklist.shape[0]
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

        self.collect_artist_ids()

        self.explode_artists()

        self.collect_top_tracks_per_artist()

        self.collect_track_properties()

        self.explode_artist_top_tracks()

        self.write_training_df()


# class AssembleTrainingTracklist:
#
#     def __init__(self):
#         pass
#
#
#
#         self.df = None
#         self.n_versions = None
#         self.rs = None
#
#         self.df_training_tracks = None
#         self.rec = None
#
#     def get_training_tracks(self):
#         self.rec = self._recommendations()
#         self.df_training_tracks = pd.concat([self._recommendations(), self._artist_top_tracks()])
#
#     def deduplicate(self):
#         self.df_training_tracks = self.df_training_tracks.drop_duplicates(
#             subset=['track_artists', 'track_name']
#         ).reset_index(drop=True)
#
#     def run(self):
#
#         self.load()
#
#         self.get_n_versions()
#
#         self.get_training_tracks()
#
#         self.deduplicate()

rc = Recommendations()
rc.run()

at = ArtistTracks()
at.run()


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


(dfe['track_danceability'], dfe['track_energy'], dfe['track_valence'],
 dfe['track_instrumentalness'], dfe['track_acousticness'], dfe['track_key'],
 dfe['track_mode']) = zip(*dfe['track_id'].apply(get_audio_features, args=(sp,)))
