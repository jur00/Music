from functools import reduce
import requests
from joblib import load
import json
import time

import numpy as np
import pandas as pd
import mysql.connector as con
from sqlalchemy import create_engine

from music.other import make_spotify


def change_list_to_string(row):
    return ' | '.join(row)


def get_end_artist(row):
    if row['Composer'] != '':
        return row['Composer']
    else:
        return row['Artist']


def get_artist_genres(row, nrows, sp):
    genres = []
    conn_error = True
    sleep_counter = 0
    while conn_error & (sleep_counter < 300):
        try:
            genres = sp.artist(row['id'])['genres']
            conn_error = False
            print(f'{row.name} / {nrows}', end='\r')
        except requests.exceptions.ReadTimeout:
            sleep_counter += 1
            print('got an error, trying again...', end='\r')
            time.sleep(1)

    return genres

def get_related_artists(row, nrows, sp):
    related_artists = []
    conn_error = True
    sleep_counter = 0
    while conn_error & (sleep_counter < 300):
        try:
            related_artists = sp.artist_related_artists(row['id'])['artists']
            conn_error = False
            print(f'{row.name} / {nrows}', end='\r')
        except requests.exceptions.ReadTimeout:
            sleep_counter += 1
            print('got an error, trying again...', end='\r')
            time.sleep(1)

    return [ra['id'] for ra in related_artists]

def create_music_database_my():
    df = pd.DataFrame(load('.\\files\\music_my.sav'))
    list_columns = ['sp_rb_name_dif', 'rb_sp_name_dif', 'yt_rb_name_dif', 'rb_yt_name_dif']
    for col in list_columns:
        df[col] = df[col].apply(change_list_to_string)

    fc = load('.appendix1\\files\\feature_categories_my.sav')

    columns_per_table = {'id': ['id', 'File Name'],
                         'rekordbox': [f for f in fc['rekordbox'] if f not in ['File Name']] + ['track_kind'],
                         'spotify': ['id'] + [
                             f for f in fc['spotify'] if f not in ['track_kind']
                         ] + ['sp_same_name', 'sp_same_duration', 'sp_dif_type'],
                         'youtube': ['id'] + fc['youtube'] + [
                             'yt_same_name', 'yt_same_duration', 'yt_dif_type', 'yt_days_since_publish',
                             'yt_views_per_day', 'yt_popularity', 'popularity'],
                         'wave': ['id'] + fc['librosa'] + fc['chord'] + fc['instrumentalness']}

    mydb = con.connect(
        host=host,
        user=username,
        password=password
    )
    mycursor = mydb.cursor()
    mycursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")

    engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{db_name}').connect()
    for table, columns in columns_per_table.items():
        df[columns].to_sql(name=f'tracks_my_{table}', if_exists='replace',
                           con=engine, index=False, chunksize=3, method='multi')
        print(f'{table}: Done')

    mycursor.execute('CREATE TABLE music.version_dates (version INT, start VARCHAR(50), end VARCHAR(50))')

    engine.close()
    mycursor.execute(f"ALTER TABLE {db_name}.tracks_my_id ADD COLUMN version_0 tinyint(1) DEFAULT 1;")
    mydb.close()


def create_artists_table():
    sp = make_spotify()

    df = pd.DataFrame(load('.\\files\\music_my.sav'))
    print('data loading done')
    df = df.loc[df['sp_dif_type'] == 'same', ['Artist', 'Composer', 'sp_artist', 'sp_id', 'sp_dif_type']]

    df['end_artist'] = df.apply(get_end_artist, axis=1)
    artists_list = [a.split(', ') for a in df['end_artist'].to_list() if a != '']
    my_artists = reduce(lambda x, y: x + y, artists_list)
    my_artists = list(map(str.lower, my_artists))
    my_artists_unique = np.unique(my_artists)
    artist_ids = {'artist': [],
                  'id': []}
    sp_artists_done = []
    for i in range(df.shape[0]):
        print(f'{i} / {df.shape[0]}', end='\r')
        artist = df['sp_artist'].iloc[i]
        if (artist not in sp_artists_done) & (artist.lower() in my_artists_unique):
            track_id = df['sp_id'].iloc[i]
            track = {}
            conn_error = True
            sleep_counter = 0
            while conn_error & (sleep_counter < 300):
                try:
                    track = sp.track(track_id)
                    conn_error = False
                except requests.exceptions.ReadTimeout:
                    sleep_counter += 1
                    print('got an error, trying again...', end='\r')
                    time.sleep(1)

            artist_ids['artist'].extend([artist['name'] for artist in track['artists']])
            artist_ids['id'].extend([artist['id'] for artist in track['artists']])

            sp_artists_done.append(artist)

    print('artist ids done')
    df_artist_ids = pd.DataFrame(artist_ids).drop_duplicates(subset=['artist'])
    df_artist_ids = df_artist_ids.loc[df_artist_ids['artist'].str.lower().isin(my_artists)].reset_index(drop=True)

    df_artist_ids['genres'] = df_artist_ids.apply(get_artist_genres, args=(df_artist_ids.shape[0], sp), axis=1)
    print('artist genres done')

    df_artist_ids['related_artists'] = df_artist_ids.apply(
        get_related_artists, args=(df_artist_ids.shape[0], sp), axis=1)
    print('related artists done')

    list_columns = ['genres', 'related_artists']
    for col in list_columns:
        df_artist_ids[col] = df_artist_ids[col].apply(change_list_to_string)

    engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{db_name}').connect()
    df_artist_ids.to_sql(name='artists_my_spotify', if_exists='replace',
                         con=engine, index=False, chunksize=3, method='multi')
    print('artists_my_spotify: Done')
    engine.close()
    mydb = con.connect(
        host=host,
        user=username,
        password=password,
        db=db_name
    )
    mycursor = mydb.cursor()
    mycursor.execute(f"ALTER TABLE {db_name}.artists_my_spotify ADD COLUMN version int DEFAULT 0;")
    mydb.close()


CREDENTIAL_FILE = 'credentials.json'

with open(CREDENTIAL_FILE, 'rb') as f:
    db_config = json.load(f)

host = db_config['db']['host']
username = db_config['db']['username']
password = db_config['db']['password']
db_name = 'music'

