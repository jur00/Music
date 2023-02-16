# code for __init__.Music.get_more_training_data

from functools import reduce
from collections import Counter
import requests
import time
from joblib import dump, load

import pandas as pd

from music.database import MySQLdb
from music.versioning import UpdateArtistsSpotify
from music.other import make_spotify, get_credentials, overrule_spotify_errors

db = MySQLdb(get_credentials('db'))
sp = make_spotify()

# artists_spotify = UpdateArtistsSpotify(db, sp)
# artists_spotify.update()

df_artists = db.load_table('artists_my_spotify')
artist_ids = df_artists['id'].to_list()
training_data = {'root_artist': [],
                 'root_artist_id': [],
                 'album': [],
                 'album_id': [],
                 'artists': [],
                 'trackname': [],
                 'album_track_id': []}
for index, row in df_artists.iloc[574:].iterrows():
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
dump(df_training_data, 'tmp_training_data_574_.sav')

related_artist_ids = df_artists['related_artists'].apply(lambda x: x.split(' | ')).to_list()
related_artist_ids = reduce(lambda x, y: x + y,
                            [ra for ra in related_artist_ids if ra not in artist_ids])
ra_id_counts = Counter(related_artist_ids)

# track = overrule_spotify_errors(sp.track(album_track['id']), empty={})
# audio_features = overrule_spotify_errors(sp.audio_features([track['id']]), empty={})
#
# training_data['popularity'].append(track['popularity'])
# training_data['duration'].append(track['duration_ms'])
# for feature in ['tempo', 'danceability', 'energy', 'valence']:
#     training_data[feature].append(audio_features[0][feature])
tmp = {'track_id': [],
       'popularity': [],
       'tempo': [],
       'duration': [],
       'danceability': [],
       'energy': [],
       'valence': []}
