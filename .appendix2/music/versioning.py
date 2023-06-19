import os
from functools import reduce

import numpy as np
import pandas as pd

from music.database import MySQLdb
from music.other import overrule_spotify_errors


class LastVersion:

    def __init__(self):
        self.database = None
        self.scaler = None
        self.models = None

    def get_last_database_version(self, df_id):
        self.database = [
            col.split('_')[1] for col in df_id.columns if col.startswith('version_')
        ][-1]


class Disjoints:

    def __init__(self,
                 raw_data_dir,
                 lake_dir,
                 raw_df_rekordbox):
        self.raw_data_dir = raw_data_dir
        self.lake_dir = lake_dir
        self.raw_df_rekordbox = raw_df_rekordbox

        self.rb_set = set(self.raw_df_rekordbox['File Name'])
        self.all_deleted_filenames_from_rb_ever = None
        self.added_filenames_to_rb = None
        self.all_deleted_ids_from_rb_ever = None
        self.all_deleted_tracknames_from_rb_ever = None
        self.added_tracknames_to_rb = None
        self.deleted_tracknames_from_rb = None
        self.add_to_lake = None

    def rekordbox_with_lake(self):
        lake_set = set(os.listdir(f'{self.lake_dir}\\music\\tracks_my'))
        self.add_to_lake = list(self.rb_set - lake_set)

    def rekordbox_with_db(self, df_db_id, df_db_rekordbox, last_version):
        db_set = set(df_db_id.loc[df_db_id[f'version_{last_version}'] == 1, 'File Name'])
        self.all_deleted_filenames_from_rb_ever = list(db_set - self.rb_set)
        self.all_deleted_ids_from_rb_ever = df_db_id.loc[
            df_db_id['File Name'].isin(self.all_deleted_filenames_from_rb_ever), 'id'].to_list()
        self.all_deleted_tracknames_from_rb_ever = df_db_rekordbox.loc[
            df_db_rekordbox['id'].isin(self.all_deleted_ids_from_rb_ever), 'Track Title'].to_list()
        self.added_filenames_to_rb = list(self.rb_set - db_set)
        self.added_tracknames_to_rb = self.raw_df_rekordbox.loc[
            self.raw_df_rekordbox['File Name'].isin(self.added_filenames_to_rb),
            'Track Title'].to_list()

    def get_just_deleted_tracknames(self, df_db_id, df_db_rekordbox, last_version, new_version):
        just_deleted_ids = df_db_id.loc[(df_db_id[f'version_{last_version}'] == 1) &
                                        (df_db_id[f'version_{new_version}'] == 0), 'id'].to_list()
        self.deleted_tracknames_from_rb = df_db_rekordbox.loc[
            df_db_rekordbox['id'].isin(just_deleted_ids), 'Track Title'].to_list()

    def check_if_anything_changed(self, df_db_id, last_version):
        active_db_filenames = set(df_db_id.loc[df_db_id[f'version_{last_version}'] == 1, 'File Name'])
        rb_filenames = set(self.raw_df_rekordbox['File Name'])

        if len(active_db_filenames ^ rb_filenames) > 0:
            return True
        else:
            return False


class NewVersion:

    def __init__(self,
                 last_version,
                 all_deleted_filenames_from_rb_ever,
                 added_filenames_to_rb,
                 all_deleted_ids_from_rb_ever):
        self.last_version = last_version
        self.new_version = int(last_version) + 1
        self.all_deleted_filenames_from_rb_ever = all_deleted_filenames_from_rb_ever
        self.added_filenames_to_rb = added_filenames_to_rb
        self.all_deleted_ids_from_rb_ever = all_deleted_ids_from_rb_ever

        self.db = MySQLdb()

    def insert_new_tracks_to_db_id(self, df_db_id):
        n_deleted = len(self.all_deleted_filenames_from_rb_ever)
        n_added = len(self.added_filenames_to_rb)
        n_mutations = n_deleted + n_added
        if n_mutations > 0:
            max_id = df_db_id['id'].max()
            new_ids = list(range(max_id + 1, max_id + n_added + 1))
            n_version_columns = int(self.last_version) + 1
            version_bools = [[0] * n_added] * n_version_columns
            db_id_input_rows = np.array([new_ids] + [self.added_filenames_to_rb] + version_bools).T.tolist()
            db_id_input_rows = [tuple(l) for l in db_id_input_rows]
            col_string = f'({", ".join(["%s"] * len(db_id_input_rows[0]))})'
            mysql_input_string = f"""INSERT INTO tracks_my_id
                                     VALUES {col_string}"""

            self.db.insert_rows_to_db(mysql_input_string, db_id_input_rows)

    def create_new_version_column_in_db_id(self):
        mysql_alter_string = f"ALTER TABLE tracks_my_id ADD COLUMN version_{self.new_version} tinyint(1) DEFAULT 1"

        self.db.create_new_version_column(mysql_alter_string)

    def change_deleted_filename_values_in_new_version_column(self):
        deleted_files_string = ", ".join(map(str, self.all_deleted_ids_from_rb_ever))
        mysql_update_string = f"UPDATE tracks_my_id SET version_{self.new_version} = 0 WHERE id IN ({deleted_files_string})"

        self.db.update_values_in_new_version_column(mysql_update_string)

    def insert_new_tracks_to_db_rekordbox(self, df_db_id, df_raw_rekordbox, added_filenames_to_rb):
        db_id_rb_merged = df_db_id.loc[df_db_id['File Name'].isin(added_filenames_to_rb)].merge(
            df_raw_rekordbox, on='File Name', how='left')
        from_rb_to_db_rekordbox = db_id_rb_merged.drop([col for col in df_db_id if col != 'id'] + ['row'], axis=1)
        db_rekordbox_input_rows = list(from_rb_to_db_rekordbox.itertuples(index=False, name=None))
        col_string = f'({", ".join(["%s"] * len(db_rekordbox_input_rows[0]))})'
        mysql_input_string = f"""INSERT INTO tracks_my_rekordbox
                                 VALUES {col_string}"""

        self.db.insert_rows_to_db(mysql_input_string, db_rekordbox_input_rows)

    @staticmethod
    def get_new_ids(df_db_id_new, df_db_id_old):
        return list(set(df_db_id_new['id']) - set(df_db_id_old['id']))


class UpdateRekordbox:

    def __init__(self, db,
                 df_db_id_old,
                 df_raw_rekordbox):
        self.db = db
        self.df_db_id_old = df_db_id_old
        self.df_db_rb_old = self.db.load_table('tracks_my_rekordbox')
        self.df_raw_rekordbox = df_raw_rekordbox

        self.differences = None

    def _locate_differences(self):
        self.df_db_rb_old = pd.merge(self.df_db_rb_old, self.df_db_id_old[['id', 'File Name']], how='left', on='id').drop('id', axis=1)
        self.df_db_rb_old = self.df_db_rb_old.loc[self.df_db_rb_old['File Name'].isin(self.df_raw_rekordbox['File Name'])]
        self.df_db_rb_old = self.df_db_rb_old.sort_values(by='File Name').reset_index(drop=True)
        self.df_raw_rekordbox = self.df_raw_rekordbox.sort_values(by='File Name').reset_index(drop=True)
        self.df_raw_rekordbox = self.df_raw_rekordbox[[c for c in self.df_db_rb_old.columns]]

        for col in self.df_raw_rekordbox.columns:
            if self.df_raw_rekordbox[col].dtype != self.df_db_rb_old[col].dtype:
                self.df_raw_rekordbox[col] = self.df_raw_rekordbox[col].astype(self.df_db_rb_old[col].dtype)

        self.differences = np.array(np.where(self.df_db_rb_old != self.df_raw_rekordbox)).transpose()
        self.df_raw_rekordbox = self.df_raw_rekordbox.merge(self.df_db_id_old[['File Name', 'id']], how='left', on='File Name')
        self.df_db_rb_old = self.df_db_rb_old.merge(self.df_db_id_old[['File Name', 'id']], how='left', on='File Name')
        self.differences = [d.tolist() for d in self.differences]
        for i in range(len(self.differences)):
            self.differences[i][1] = self.df_raw_rekordbox.columns[self.differences[i][1]]
            self.differences[i][0] = self.df_raw_rekordbox.loc[self.differences[i][0], 'id']
        self.differences = [d for d in self.differences if 'Date Added' not in d]

    def update(self):
        self._locate_differences()
        for i in range(len(self.differences)):
            col = self.differences[i][1]
            id = self.differences[i][0]
            old_value = self.df_db_rb_old.loc[self.df_db_rb_old['id'] == id, col].iloc[0]
            new_value = self.df_raw_rekordbox.loc[self.df_raw_rekordbox['id'] == id, col].iloc[0]
            if new_value != '':
                dtype = self.df_raw_rekordbox[col].dtype
                quot = "'" if dtype == 'object' else ""
                self.db.update_rekordbox_table(col, id, new_value, quot)

class UpdateArtistsSpotify:

    def __init__(self, db, sp):
        self.db = db
        self.sp = sp

    @staticmethod
    def __change_list_to_string(row):
        return ' | '.join(row)

    @staticmethod
    def __get_end_artist(row):
        if row['Composer'] != '':
            return row['Composer']
        else:
            return row['Artist']

    @staticmethod
    def __get_artist_genres(row, sp):
        genres = overrule_spotify_errors(sp.artist(row['id'])['genres'], empty=[])

        return genres

    @staticmethod
    def __get_related_artists(row, sp):
        related_artists = overrule_spotify_errors(sp.artist_related_artists(row['id'])['artists'], empty=[])

        return [ra['id'] for ra in related_artists]

    def update(self):
        df_rekordbox = self.db.load_table('tracks_my_rekordbox', only_current_tracks=True)
        df_spotify = self.db.load_table('tracks_my_spotify', only_current_tracks=True)
        df_artists_spotify = self.db.load_table('artists_my_spotify')
        df = df_rekordbox.merge(df_spotify, on='id', how='left')
        df = df.loc[df['sp_dif_type'] == 'same', ['id', 'Artist', 'Composer', 'sp_artist', 'sp_id', 'sp_dif_type']]
        df_id = self.db.load_table('tracks_my_id')
        version_cols = [col for col in df_id.columns if col.startswith('version')]
        if len(version_cols) > 1:
            version_cols = version_cols[-2:]
            df_id = df_id[['id'] + version_cols]
            df = df.merge(df_id, on='id', how='left')
            df = df.loc[(df[version_cols[0]] == 0) & (df[version_cols[1]] == 1)]

            df['end_artist'] = df.apply(self.__get_end_artist, axis=1)
            artists_list = [a.split(', ') for a in df['end_artist'].to_list() if a != '']
            my_artists = reduce(lambda x, y: x + y, artists_list)
            my_artists = list(map(str.lower, my_artists))
            my_artists_unique = np.unique(my_artists)

            artist_ids = {'artist': [],
                          'id': []}
            sp_artists_done = df_artists_spotify['artist'].to_list()
            for i in range(df.shape[0]):
                artist = df['sp_artist'].iloc[i]
                if (artist not in sp_artists_done) & (artist.lower() in my_artists_unique):
                    track_id = df['sp_id'].iloc[i]
                    track = overrule_spotify_errors(self.sp.track(track_id))

                    artist_ids['artist'].extend([artist['name'] for artist in track['artists']])
                    artist_ids['id'].extend([artist['id'] for artist in track['artists']])

                    sp_artists_done.append(artist)

            if len(artist_ids['artist']) > 0:
                df_artist_ids = pd.DataFrame(artist_ids).drop_duplicates(subset=['artist'])
                df_artist_ids = df_artist_ids.loc[df_artist_ids['artist'].str.lower().isin(my_artists)].reset_index(
                    drop=True)

                df_artist_ids['genres'] = df_artist_ids.apply(
                    self.__get_artist_genres, args=(self.sp,), axis=1)

                df_artist_ids['related_artists'] = df_artist_ids.apply(
                    self.__get_related_artists, args=(self.sp,), axis=1)

                df_artist_ids['version'] = int(version_cols[1].split('_')[-1])

                list_columns = ['genres', 'related_artists']
                for col in list_columns:
                    df_artist_ids[col] = df_artist_ids[col].apply(self.__change_list_to_string)

                sql_cols = str(tuple(df_artist_ids.columns)).replace("'", "")
                sql_vals = str(tuple(len(df_artist_ids.columns) * ['%s'])).replace("'", "")
                val = list(df_artist_ids.itertuples(index=False, name=None))
                sql_string = f"INSERT INTO artists_my_spotify {sql_cols} VALUES {sql_vals}"
                self.db.insert_rows_to_db(sql_string, val)
            else:
                print('No artists were added to the database')
