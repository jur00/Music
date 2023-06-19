from functools import reduce
import requests
from joblib import load
import json
import time

import numpy as np
import pandas as pd
import mysql.connector as con
from sqlalchemy import create_engine

from music.other import make_spotify, get_credentials, overrule_spotify_errors
from music.database import MySQLdb


def change_list_to_string(row):
    return ' | '.join(row)


def get_end_artist(row):
    if row['Composer'] != '':
        return row['Composer']
    else:
        return row['Artist']


def get_artist_genres(row, nrows, sp):
    genres = overrule_spotify_errors(sp.artist(row['id'])['genres'], empty=[])

    return genres

def get_related_artists(row, nrows, sp):
    related_artists = overrule_spotify_errors(sp.artist_related_artists(row['id'])['artists'], empty=[])

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
            track = overrule_spotify_errors(sp.track(track_id), empty={})

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


def create_training_table():
    # two separate runs because of spotipy rate limits
    db = MySQLdb(get_credentials('db'))

    df_artists = db.load_table('artists_my_spotify')
    training_data = {'root_artist': [],
                     'root_artist_id': [],
                     'album': [],
                     'album_id': [],
                     'artists': [],
                     'trackname': [],
                     'album_track_id': []}
    for index, row in df_artists.iterrows():
        artist_albums = overrule_spotify_errors(sp.artist_albums(row['id'])['items'], empty={})
        for artist_album in artist_albums:
            album_id = artist_album['id']
            album_tracks = overrule_spotify_errors(sp.album_tracks(album_id)['items'], empty={})
            for album_track in album_tracks:
                track_artists = album_track['artists']

                training_data['root_artist'].append(row['artist'])
                training_data['root_artist_id'].append(row['id'])
                training_data['album'].append(artist_album['name'])
                training_data['album_id'].append(artist_album['id'])
                training_data['artists'].append(
                    ', '.join([track_artist['name'] for track_artist in track_artists]))
                training_data['trackname'].append(album_track['name'])
                training_data['album_track_id'].append(album_track['id'])

                print(f'{index} / {df_artists.shape[0]} {row["artist"]} {len(artist_albums)}'
                      f' albums where found. Total tracks: {len(training_data["album"])}',
                      end='\r')

    df_training_data = pd.DataFrame(training_data).drop_duplicates(subset=['artists', 'trackname'])
    # df_training_data = pd.concat([load(f'tmp_training_data_{idxs}.sav') for idxs in ['0_386', '386_574', '574_']])
    df_training_data = df_training_data.reset_index(drop=True)
    df_l = df_training_data.shape[0]
    chunks = list(range(0, df_l, 50)) + [df_l]
    training_data_ext = {'track_id': [],
                         'popularity': [],
                         'duration': [],
                         'tempo': [],
                         'danceability': [],
                         'energy': [],
                         'valence': [],
                         'instrumentalness': [],
                         'mode': [],
                         'key': []}
    for i in range(1, len(chunks)):
        print(f'{i} / {len(chunks) - 1}', end='\r')
        start, stop = chunks[i - 1:i + 1]
        tracks = overrule_spotify_errors(
            sp.tracks(df_training_data.loc[start:stop - 1, 'album_track_id']),
            empty={})
        track_ids = [track['id'] for track in tracks['tracks']]
        audio_features = overrule_spotify_errors(sp.audio_features(track_ids))

        for track, audio_feature in zip(tracks['tracks'], audio_features):
            training_data_ext['track_id'].append(track['id'])
            training_data_ext['popularity'].append(track['popularity'])
            training_data_ext['duration'].append(track['duration_ms'])
            if audio_feature:
                for feature in ['tempo', 'danceability', 'energy', 'valence', 'instrumentalness', 'mode', 'key']:
                    training_data_ext[feature].append(audio_feature[feature])
            else:
                for feature in ['tempo', 'danceability', 'energy', 'valence', 'instrumentalness', 'mode', 'key']:
                    training_data_ext[feature].append(0)

    df_training_data_ext = pd.DataFrame(training_data_ext)

    df = pd.concat([df_training_data, df_training_data_ext], axis=1)
    df = df.reset_index().rename(columns={'index': 'id'})

    engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{db_name}').connect()
    df.to_sql(name='tracks_training', if_exists='replace',
              con=engine, index=False, chunksize=24, method='multi')
    print('tracks_training: Done')
    engine.close()


CREDENTIAL_FILE = 'credentials.json'

with open(CREDENTIAL_FILE, 'rb') as f:
    db_config = json.load(f)

host = db_config['db']['host']
username = db_config['db']['username']
password = db_config['db']['password']
db_name = 'music'

sp = make_spotify()

