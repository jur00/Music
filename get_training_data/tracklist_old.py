from pathlib import Path
from functools import reduce
import os

from joblib import load, dump
import pandas as pd
import numpy as np

from base.connect import SpotifyConnect
from base.spotify_youtube import get_spotify_track, get_spotify_artist_genres, get_spotify_related_artists
from base.helpers import levenshtein_distance, Progress


class UpdateArtists:

    def __init__(self, working_dir,
                 file_dir,
                 my_music_fn,
                 my_artists_fn,
                 credential_dir,
                 credential_fn):

        os.chdir(working_dir)

        self.my_music_path = Path(file_dir, my_music_fn)
        self.my_artists_path = Path(file_dir, my_artists_fn)
        self.sp = SpotifyConnect().sp

        self.df = None
        self.current_version = None
        self.added_artists = None
        self.my_artists = None
        self.sp_id = None

    @staticmethod
    def __get_end_artist(row):
        if row['Composer'] != '':
            return row['Composer']
        else:
            return row['Artist']

    def __load_df(self):
        return pd.DataFrame(load(self.my_music_path))

    def _version_info(self):
        self.version_cols = list(sorted([col for col in self.df.columns if col.startswith('version_')]))
        self.current_version = max(self.version_cols)
        self.current_version_n = int(self.current_version.split('_')[1])

    # @staticmethod
    # def __filter_on_current_version(x_df):
    #     current_version = max([col for col in x_df.columns if col.startswith('version_')])
    #     return x_df.loc[(x_df[current_version] == 1)]

    def __filter_on_same_and_cols(self, x_df):
        use_cols = ['Artist', 'Composer', 'sp_artist', 'sp_id', 'sp_popularity']
        keep_cols = use_cols + self.version_cols
        return x_df.loc[x_df['sp_dif_type'] == 'same', keep_cols]

    @staticmethod
    def __separate_all_my_artists(added_artists):
        added_artists_list = [a.split(', ') for a in added_artists if a != '']
        if len(added_artists_list) > 0:
            added_artists_all = reduce(lambda x, y: x + y, added_artists_list)
            added_artists_unique = set(list(map(str.lower, added_artists_all)))
        else:
            added_artists_unique = set()

        return added_artists_unique

    def _get_my_artists_with_one_version(self):
        added_artists = set(self.df['end_artist'])
        self.added_artists = self.__separate_all_my_artists(added_artists)
        self.my_artists = [{'artist': ar, 'version_1': 1} for ar in self.added_artists]

    def _create_artist_categories(self):
        l_artists_cats = ['added', 'removed', 'remained', 'remained_not']
        d_artists = {cat: {} for cat in l_artists_cats}
        for cat, pre, cur in zip(l_artists_cats, [0, 1, 1, 0], [1, 0, 1, 0]):
            in_version_mask = ((self.df[f'version_{self.current_version_n - 1}'] == pre) &
                               (self.df[self.current_version] == cur))
            d_artists[cat] = set(self.df.loc[in_version_mask, 'end_artist'])

        return d_artists

    def __add_current_version(self, previous_version_artists, artists_set, value):
        for artist in artists_set:
            d_previous_artist = [d for d in previous_version_artists if d['artist'] == artist]
            if len(d_previous_artist) > 0:
                d = d_previous_artist[0]
                previous_version_artists.remove(d)
                d.update({self.current_version: value})
            else:
                d = {'artist': artist}
                d.update({vc: 0 for vc in self.version_cols[:-1]})
                d.update({self.version_cols[-1]: 1})

            previous_version_artists.append(d)

        return previous_version_artists

    def _get_my_artists_with_multiple_versions(self, d_artists):
        self.added_artists = self.__separate_all_my_artists(d_artists['added'])
        previous_version_artists = load(self.my_artists_path)
        previous_version_artists = self.__add_current_version(
            previous_version_artists, d_artists['removed'] | d_artists['remained_not'], 0)
        self.my_artists = self.__add_current_version(
            previous_version_artists, d_artists['added'] | d_artists['remained'], 1)

    def _get_id_of_one_track(self, a):
        df_tracks_from_added = self.df.loc[self.df['l_end_artist'].apply(lambda x: a in x),
                                           ['sp_id', 'sp_popularity']]
        self.sp_id = df_tracks_from_added.loc[df_tracks_from_added['sp_popularity'].idxmax(), 'sp_id']

    def _get_artists_from_track_id(self):
        track, _ = get_spotify_track(self.sp, self.sp_id)
        result_artist_names = [artist['name'] for artist in track['artists']]
        result_artist_ids = [artist['id'] for artist in track['artists']]

        return result_artist_names, result_artist_ids

    @staticmethod
    def _find_corresponding_artist(a, result_artist_names):
        distances = [levenshtein_distance(a, result_artist_names[i].lower())
                     for i in range(len(result_artist_names))]
        nearest = np.argmin(distances)

        return nearest

    def _get_features_of_corresponding_artist(self, nearest, result_artist_names, result_artist_ids):
        artist_name = result_artist_names[nearest]
        artist_id = result_artist_ids[nearest]
        artist_genres, _ = get_spotify_artist_genres(self.sp, artist_id)
        related_artists, _ = get_spotify_related_artists(self.sp, artist_id)
        artist_features = dict(artist_name_sp=artist_name, artist_id=artist_id,
                               artist_genres=artist_genres, related_artists=related_artists)

        return artist_features

    def run(self):
        # preparation
        self.df = self.__load_df()
        self._version_info()
        self.df = (self.df
                   .pipe(self.__filter_on_same_and_cols)
                   .assign(end_artist=lambda x_df: x_df.apply(self.__get_end_artist, axis=1))
                   )

        # get list of artists from the latest version
        if len(self.version_cols) == 1:
            self._get_my_artists_with_one_version()
        else:
            d_artists = self._create_artist_categories()
            self._get_my_artists_with_multiple_versions(d_artists)

        # split artists by comma
        self.df = self.df.assign(l_end_artist=lambda x_df: x_df['end_artist'].str.lower().str.split(', '))

        progress = Progress()
        for i in range(len(self.my_artists)):
            # select artist
            a = self.my_artists[i]['artist']
            if a in self.added_artists:

                # get artist features
                self._get_id_of_one_track(a)
                result_artist_names, result_artist_ids = self._get_artists_from_track_id()
                nearest = self._find_corresponding_artist(a, result_artist_names)
                artist_features = self._get_features_of_corresponding_artist(
                    nearest, result_artist_names, result_artist_ids)

                self.my_artists[i].update(artist_features)
                progress.show(i, range(len(self.my_artists)))

        dump(self.my_artists, self.my_artists_path)
