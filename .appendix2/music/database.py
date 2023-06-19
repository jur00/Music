import pandas as pd
from sqlalchemy import create_engine
import mysql.connector as con
from datetime import datetime


class MySQLdb:

    def __init__(self, db_config):
        self.db_config = db_config

    def _connect_with_db(self):
        self.db_engine = create_engine(
            f"mysql+mysqlconnector://{self.db_config['username']}:"
            f"{self.db_config['password']}@{self.db_config['host']}/{self.db_config['db_name']}").connect()
        self.db_connection = con.connect(host=self.db_config['host'],
                                         user=self.db_config['username'],
                                         password=self.db_config['password'],
                                         database=self.db_config['db_name'])
        self.db_cursor = self.db_connection.cursor()

    def _close_connection_with_db(self):
        self.db_engine.close()
        self.db_cursor.close()
        self.db_connection.close()

    def load_table(self, table_name, only_current_tracks=False):
        self._connect_with_db()
        chunksize = 3 if table_name == 'track_my_wave' else None
        df = pd.read_sql_table(table_name=table_name, con=self.db_engine.connect(),
                               chunksize=chunksize)
        if only_current_tracks:
            df_id = pd.read_sql_table(table_name='tracks_my_id', con=self.db_engine.connect(),
                                      chunksize=chunksize)
            last_version_column = [col for col in df_id.columns if col.startswith('version')][-1]
            df = df.merge(df_id[['id', 'File Name', last_version_column]], how='left', on='id')
            df = df.loc[df[last_version_column] == 1]
        self._close_connection_with_db()

        return df

    def insert_rows_to_db(self, insert_string, values):
        self._connect_with_db()
        self.db_cursor.executemany(insert_string, values)
        self.db_connection.commit()
        self._close_connection_with_db()

    def insert_row_to_db(self, insert_string, values):
        self._connect_with_db()
        self.db_cursor.execute(insert_string, values)
        self.db_connection.commit()
        self._close_connection_with_db()

    def create_new_version_column(self, alter_string):
        self._connect_with_db()
        self.db_cursor.execute(alter_string)
        self._close_connection_with_db()

    def update_values_in_new_version_column(self, update_string):
        self._connect_with_db()
        self.db_cursor.execute(update_string)
        self.db_connection.commit()
        self._close_connection_with_db()

    def get_sp_yt_popularity_dists(self):
        self._connect_with_db()
        dists = {}
        sp_yt = ['sp', 'yt']
        spotify_youtube = ['spotify', 'youtube']
        for sy, sp_yt in zip(sp_yt, spotify_youtube):
            self.db_cursor.execute(
                f"""SELECT {sy}_popularity FROM tracks_my_{sp_yt}""")
            dists[sp_yt] = [row[0] for row in self.db_cursor.fetchall()]

        self._close_connection_with_db()

        return dists

    def get_spotify_ids(self):
        self._connect_with_db()
        self.db_cursor.execute('SELECT id FROM tracks_my_spotify')
        ids_in_spotify_db = [row[0] for row in self.db_cursor.fetchall()]
        self._close_connection_with_db()

        return ids_in_spotify_db

    def get_column_names(self, table):
        self._connect_with_db()
        self.db_cursor.execute(f'SELECT * FROM tracks_my_{table} WHERE id = 1')
        tmp = self.db_cursor.fetchall()
        columns = self.db_cursor.column_names
        self._close_connection_with_db()

        return list(columns)

    @staticmethod
    def create_insert_values_string_part(values):
        comma = ', '
        s = '%s'
        s_vals = f'({comma.join([s] * len(values))})'

        return s_vals

    def insert_process_version(self, version, start_time, end_time):
        values = (int(version), start_time, end_time)
        s_vals = self.create_insert_values_string_part(values)
        mysql_string = f"INSERT INTO version_dates VALUES {s_vals}"

        self.insert_row_to_db(mysql_string, values)

    def update_rekordbox_table(self, col, id, new_value, quot):
        self._connect_with_db()
        if isinstance(new_value, str):
            if new_value.__contains__("'"):
                quot = '"' if quot == "'" else ''
                mysql_string = f"""UPDATE tracks_my_rekordbox SET `{col}` = {quot}{new_value}{quot} WHERE id = {id}"""
            else:
                mysql_string = f"""UPDATE tracks_my_rekordbox SET `{col}` = {quot}{new_value}{quot} WHERE id = {id}"""
        else:
            mysql_string = f"""UPDATE tracks_my_rekordbox SET `{col}` = {quot}{new_value}{quot} WHERE id = {id}"""
        self.db_cursor.execute(mysql_string)
        self.db_connection.commit()
        self._close_connection_with_db()

    def refresh(self):
        self._connect_with_db()
        for table in ['id', 'rekordbox', 'spotify', 'youtube', 'wave']:
            self.db_cursor.execute(f'DELETE FROM tracks_my_{table} WHERE id > 2481;')
        self.db_cursor.execute(f'DELETE FROM artists_my_spotify WHERE version > 0;')
        self.db_cursor.execute(f"ALTER TABLE tracks_my_id DROP COLUMN version_1;")
        self.db_cursor.execute('DELETE FROM version_dates WHERE version > 0;')
        self.db_connection.commit()
        self._close_connection_with_db()
