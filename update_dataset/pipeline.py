import os

from joblib import load, dump

from update_dataset.engineering import (Credentials, MyMusic, RekordboxMusic, Disjoint,
                                        SpotifyFeatures, YoutubeFeatures, WaveFeatures,
                                        FeaturesImprovement, Popularity)
from update_dataset.helpers import Progress

tracks_dir = 'D:\\Data Science\\Lake\\music\\tracks_my\\'

credentials_sp = Credentials(directory='', filename='credentials.json', api='sp')
credentials_sp.get()
credentials_spotify = credentials_sp.credentials

credentials_yt = Credentials(directory='', filename='credentials.json', api='yt')
credentials_yt.get()
credentials_youtube = credentials_yt.credentials

os.chdir('D:\\Data Science\\Python zelfstudie\\Music')

mm = MyMusic(directory='files', filename='music_my.sav')
mm.get()

rm = RekordboxMusic(directory='files', filename='music_rekordbox.txt')
rm.get()

tracks_in_dir = os.listdir(tracks_dir)
dj_dir = Disjoint(rm.data, tracks_in_dir, datatype2='list')
copy_to_tracks_dir = dj_dir.not_in_data2()
n_tracks_to_copy = len(copy_to_tracks_dir)

if n_tracks_to_copy > 0:
    raise FileNotFoundError(f'{n_tracks_to_copy} tracks need to be copied to tracks_dir: {copy_to_tracks_dir}')

dj_data = Disjoint(rm.data, mm.data)
features_indexes = dj_data.get_indexes(type='not_in_data2')

if len(features_indexes) == 0:
    popularity = Popularity(mm.data)
    popularity.get()
    if popularity.complete:
        print('Dataset up to date')
    else:
        dump(popularity.data, 'music_my.sav')
        print('Popularity added, dataset up to date')
else:
    sf = SpotifyFeatures(rb_data=rm.data, credentials=credentials_spotify)
    yf = YoutubeFeatures(rb_data=rm.data, credentials=credentials_youtube)
    wf = WaveFeatures(tracks_dir=tracks_dir, rb_data=rm.data)

    my_music = mm.data.copy()
    progress = Progress()
    for i in features_indexes:
        my_music = load('music_my.sav')
        all_features = {}

        all_features.update(rm.data[i])

        sf.get(i)
        all_features.update(sf.spotify_features)

        yf.get(i)
        all_features.update(yf.youtube_features)

        wf.get(i)
        all_features.update(wf.wave_features)

        fi = FeaturesImprovement(all_features)
        fi.improve()
        all_features = fi.af.copy()
        my_music.append(all_features)
        dump(my_music, 'music_my.sav')

        progress.show(features_indexes, i)

# versie kolom toevoegen aan data, zodat ook getrackt kan worden welke nummers eruit zijn gehaald

