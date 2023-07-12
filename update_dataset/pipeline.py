import os
from pathlib import Path

from joblib import load, dump

from update_dataset.engineering import (load_credentials, RekordboxMusic, Disjoint,
                                        SpotifyFeatures, YoutubeFeatures, WaveFeatures,
                                        FeaturesImprovement, Popularity, Versioning)
from update_dataset.helpers import Progress

# set global variables
tracks_dir = 'D:\\Data Science\\Lake\\music\\tracks_my\\'
file_dir = 'files'
my_music_fn = 'music_my.sav'
my_music_path = Path(file_dir, my_music_fn)
rekordbox_music_fn = 'music_rekordbox.txt'
rekordbox_music_path = Path(file_dir, rekordbox_music_fn)
credential_dir = ''
credential_fn = 'credentials.json'
credential_path = Path(credential_dir, credential_fn)

# set working dir
os.chdir('D:\\Data Science\\Python zelfstudie\\Music')

# load data
data_mm = load(my_music_path)
rm = RekordboxMusic(rekordbox_music_path)
data_rm = rm.get()

# check which tracks are not in track_dir
tracks_in_dir = os.listdir(tracks_dir)
dj_dir = Disjoint(rm.data, tracks_in_dir, datatype2='list')
copy_to_tracks_dir = dj_dir.not_in_data2()
n_tracks_to_copy = len(copy_to_tracks_dir)
if n_tracks_to_copy > 0:
    raise FileNotFoundError(f'{n_tracks_to_copy} tracks need to be copied to tracks_dir: {copy_to_tracks_dir}')

dj_data = Disjoint(data_rm, data_mm)
added_indexes = dj_data.get_indexes(type='not_in_data2')
removed_indexes = dj_data.get_indexes(type='not_in_data1')


version = Versioning(data_mm, removed_indexes)
version.get_version()

if len(added_indexes) == 0:
    popularity = Popularity(data_mm, my_music_path)
    popularity.get()
else:
    credentials = load_credentials(credential_path)
    sf = SpotifyFeatures(rb_data=data_rm, credentials=credentials['sp'])
    yf = YoutubeFeatures(rb_data=data_rm, credentials=credentials['yt'])
    wf = WaveFeatures(tracks_dir=tracks_dir, rb_data=data_rm)

    progress = Progress()
    for i in [1950, 1620, 45]:
        my_music = load(my_music_path)
        all_features = {}

        all_features.update(data_rm[i])

        sf.get(i)
        all_features.update(sf.spotify_features)

        yf.get(i)
        all_features.update(yf.youtube_features)

        wf.get(i)
        all_features.update(wf.wave_features)

        all_features.update(version.set_version_column(i))

        fi = FeaturesImprovement(all_features)
        fi.improve()

        all_features = fi.af.copy()
        my_music.append(all_features)
        dump(my_music, my_music_path)

        progress.show([1950, 1620, 45], i)

