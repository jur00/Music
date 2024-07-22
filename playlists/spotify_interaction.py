from config import Config
from base.connect import load_credentials
import spotipy

import pandas as pd


class SpotifyAuthentication:

    def __init__(self, objective):
        # objective must be one of ['playlist_info', 'create_playlist']
        self.objective = objective
        self.scope = None
        
        self._define_scope()
        
        credentials = load_credentials("../credentials.json")
        self.sp_oauth = spotipy.SpotifyOAuth(
            username=credentials['sp']['username'],
            client_id=credentials['sp']['cid'],
            client_secret=credentials['sp']['secret'],
            redirect_uri=credentials['sp']['redirect_uri'],
            scope=self.scope
        )

        self.sp = None

    def _define_scope(self):
        if self.objective == 'playlist_info':
            self.scope = 'user-library-read'
        elif self.objective == 'create_playlist':
            self.scope = 'playlist-modify-public'
        else:
            ValueError('objective parameter must be one of [playlist_info, create_playlist]')
    
    def print_auth_link(self):
        auth_url = self.sp_oauth.get_authorize_url()
        print(f'Please navigate here to authorize: {auth_url}')

    def input_response_link_to_create_sp(self, link):
        code = self.sp_oauth.parse_response_code(link)
        token = self.sp_oauth.get_access_token(code, as_dict=False)
        self.sp = spotipy.Spotify(auth=token)


class PlaylistInfo:

    def __init__(self, sp):
        self.sp = sp

        self.id_ = None

    @staticmethod
    def _create_df_from_sp_items(items):
        df = (
            pd.DataFrame([
                {'name': tr['track']['name'],
                 'artists': tr['track']['artists'][0]['name'],
                 'id': tr['track']['id'],
                 'popularity': tr['track']['popularity'],
                 'added_at': tr['added_at']
                 }
                for tr in items
            ])
        )
        return df

    def _get_liked_tracks(self, limit=50, save_as_csv=False):
        offset = 0
        n_tracks_added = limit
        dfs = []
        while n_tracks_added == limit:
            tracks = self.sp.current_user_saved_tracks(limit=limit, offset=offset)['items']
            df_part = self._create_df_from_sp_items(tracks)
            dfs.append(df_part)

            n_tracks_added = len(tracks)
            offset += limit

        df = pd.concat(dfs, ignore_index=True)

        if save_as_csv:
            df.to_csv(f'{Config.playlists_dir}tracks_liked_no_features.csv', index=False)
        else:
            return df

    def _get_playlist_id(self, playlist_name):
        playlists_props = self.sp.current_user_playlists()['items']
        playlist_ids = {pl['name']: pl['id'] for pl in playlists_props}
        id_ = playlist_ids[playlist_name]
        return id_

    def get_playlist_tracks(self, playlist_name, limit=100, save_as_csv=False):
        self.id_ = self._get_playlist_id(playlist_name)

        offset = 0
        n_tracks_added = limit
        dfs = []
        while n_tracks_added == limit:
            tracks = self.sp.playlist_items(self.id_, limit=limit, offset=offset, additional_types=('track',))['items']
            df_part = self._create_df_from_sp_items(tracks)
            dfs.append(df_part)

            n_tracks_added = len(tracks)
            offset += limit

        df = pd.concat(dfs, ignore_index=True)

        if save_as_csv:
            df.to_csv(f'{Config.playlists_dir}tracks_{playlist_name}_no_features.csv', index=False)
        else:
            return df

    @staticmethod
    def _cap_tempo(xdf, tempo):
        xdf.loc[xdf['tempo'] >= tempo, 'tempo'] /= 2
        return xdf

    def _add_features_to_tracklist(
            self,
            playlist_name,
            features=('id', 'energy', 'danceability', 'valence', 'instrumentalness', 'tempo'),
            limit=100
    ):
        csv_path = f'{Config.playlists_dir}tracks_{playlist_name.lower()}_no_features.csv'
        df_tracks = pd.read_csv(csv_path)

        offset = 0
        n_rows_left = df_tracks.shape[0]
        dfs = []
        while n_rows_left > 0:
            if n_rows_left < limit:
                limit = n_rows_left
            results = self.sp.audio_features(df_tracks['id'].iloc[offset:offset + limit])
            df_part = (
                pd.DataFrame([{feature: result[feature] for feature in features} for result in results])
                .pipe(self._cap_tempo, 150)
            )
            dfs.append(df_part)
            n_rows_left -= limit
            offset += limit

        df_features = pd.concat(dfs, ignore_index=True)
        df_with_features = df_tracks.merge(df_features, on='id', how='left')
        new_csv_path = csv_path.replace('_no_features', '')
        df_with_features.to_csv(new_csv_path, index=False)

    @staticmethod
    def _add_liked_column_to_playlist_stats():
        df_sweety = pd.read_csv(f'{Config.playlists_dir}tracks_sweety.csv')
        df_liked = pd.read_csv(f'{Config.playlists_dir}tracks_liked_no_features.csv')

        df_sweety_with_liked = (
            df_sweety
            .merge(df_liked[['id']], how='left', on='id', indicator=True)
            .assign(liked=lambda xdf: xdf['_merge'] == 'both')
            .drop(columns='_merge')
            .drop_duplicates()
            .reset_index(drop=True)
        )

        df_sweety_with_liked.to_csv(f'{Config.playlists_dir}tracks_sweety_liked.csv', index=False)

    def update_liked(self):
        self._get_liked_tracks(save_as_csv=True)
        self._add_features_to_tracklist('liked')

    def update_playlist(self, playlist_name):
        self.get_playlist_tracks(playlist_name, save_as_csv=True)
        self._add_features_to_tracklist(playlist_name.lower())
        self._add_liked_column_to_playlist_stats()


class CreateInTheMix:

    def __init__(self, sp, filter_col, filter_min, filter_max):
        self.sp = sp
        self.filter_col = filter_col
        self.filter_min = filter_min
        self.filter_max = filter_max

        self.user_id = sp.current_user()['id']
        self.name_playlist = 'InTheMix'

        self.df = None
        self.playlists = None
        self.playlist_exists = None
        self.id_ = None

    def _get_data(self):
        self.df = pd.read_csv(f'{Config.playlists_dir}tracks_sweety_liked.csv')

    @staticmethod
    def _convert_ids_to_uris(ids):
        return 'spotify:track:' + ids

    def _select_playlist_track_uris(self):
        self.new_uris = (
            self.df
            .pipe(lambda xdf: xdf.loc[xdf[self.filter_col].between(self.filter_min, self.filter_max)])
            .assign(uri=lambda xdf: xdf['id'].apply(self._convert_ids_to_uris))
            .sort_values(by='tempo')
            .reset_index(drop=True)
            ['uri']
        )

    def _check_if_playlist_exists(self):
        self.playlists = self.sp.current_user_playlists()['items']
        self.playlist_exists = self.name_playlist in [playlist['name'] for playlist in self.playlists]

    def _get_playlist_track_uris(self):
        pi = PlaylistInfo(self.sp)
        self.old_uris = (
            pi.get_playlist_tracks(self.name_playlist)
            .assign(uri=lambda xdf: xdf['id'].apply(self._convert_ids_to_uris))
            .reset_index(drop=True)
            ['uri']
        )

        self.id_ = pi.id_

    def _add_tracks_to_playlist(self, limit=100):
        add_uris = list(set(self.new_uris) - set(self.old_uris))
        n_add_tracks = len(add_uris)

        if n_add_tracks <= limit:
            self.sp.playlist_add_items(self.id_, add_uris)
        else:
            offset = 0
            n_rows_left = n_add_tracks
            while n_rows_left > 0:
                if n_rows_left < limit:
                    limit = n_rows_left
                self.sp.playlist_add_items(self.id_, add_uris[offset:offset + limit])
                n_rows_left -= limit
                offset += limit

    def _delete_tracks_from_playlist(self):
        delete_uris = list(set(self.old_uris) - set(self.new_uris))
        self.sp.playlist_remove_all_occurrences_of_items(self.id_, delete_uris)

    def _create_new_playlist(self):
        self.id_ = sp.user_playlist_create(
            user=load_credentials()['sp']['username'],
            name=self.name_playlist,
            public=True
        )['id']

        self.old_uris = []

    def create(self):
        self._get_data()
        self._select_playlist_track_uris()
        self._check_if_playlist_exists()

        if self.playlist_exists:
            self._get_playlist_track_uris()
            self._add_tracks_to_playlist()
            self._delete_tracks_from_playlist()
        else:
            self._create_new_playlist()
            self._add_tracks_to_playlist()


sa = SpotifyAuthentication(objective='playlist_info')
sa.print_auth_link()
sa.input_response_link_to_create_sp('http://localhost:8080/?code=AQCoUWRe7D19KnKd9BWuegLy-y2xvu7az6i32TrEr6FTidgTc5Ma_K2dScS3H1BT1_5R8nfcWD2nQIgPP3jDFi-3JteKjiwcptNnZVpRIkANdoRp7zOGfRvubNxrUsahRdVVv_edRwAgsx8USaOT9xySXhZMHqQFlO-d6WXi3orWm3gFpTge6mPqAO0')
sp = sa.sp

pi = PlaylistInfo(sa.sp)
pi.update_liked()
pi.update_playlist('Sweety')

citm = CreateInTheMix(sa.sp)
citm.create()
