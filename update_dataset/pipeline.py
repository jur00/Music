import os
from pathlib import Path

from update_dataset.helpers import find
from joblib import load, dump

from update_dataset.engineering import (load_credentials, RekordboxMusic, ExplorerInterruption,
                                        Disjoint, SpotifyFeatures, YoutubeFeatures, WaveFeatures,
                                        FeaturesImprovement, Popularity, Versioning, ConnectionErrors)
from update_dataset.helpers import Progress

# set global variables
tracks_dir = 'D:\\Data Science\\Lake\\music\\tracks_my\\'
file_dir = 'files\\data\\update_dataset'
my_music_fn = 'music_my.sav'
my_music_path = Path(file_dir, my_music_fn)
rekordbox_music_fn = 'music_rekordbox.txt'
rekordbox_music_version_check_fn = 'music_rekordbox_version_check.txt'
rekordbox_music_path = Path(file_dir, rekordbox_music_fn)
rekordbox_music_version_check_path = Path(file_dir, rekordbox_music_version_check_fn)
credential_dir = ''
credential_fn = 'credentials.json'
credential_path = Path(credential_dir, credential_fn)

# dump([], my_music_path)

# set working dir
os.chdir('D:\\Data Science\\Python zelfstudie\\Music')

# load data
data_mm = load(my_music_path)
rm = RekordboxMusic(rekordbox_music_path)
data_rm = rm.get()

# check if vocalness feature extraction was interrupted
ei = ExplorerInterruption(data_rm, tracks_dir)
ei.change_shortened_filenames()
ei.empty_output_map()

# check which tracks are not in track_dir
disjoint = Disjoint(data_rm, data_mm, tracks_dir)
disjoint.check_missing_filenames_in_tracks_dir()
filenames_added = disjoint.get_added_tracks()
filenames_removed = disjoint.get_removed_tracks()
filenames_wave = disjoint.get_tracks_for_wave_analysis()

# check which tracks are added/removed
# dj_data = Disjoint(data_rm, data_mm)
# added_indexes_rm = dj_data.get_indexes(type='not_in_data2')
# added = dj_data.not_in_data2()
# removed = dj_data.not_in_data1()
# n_changes = len(added) + len(removed)

# only if n_changes > 0, update dataset with new version and features
if disjoint.n_changes == 0:
    popularity = Popularity(data_mm, my_music_path)
    popularity.get()
else:
    # create new version number
    version = Versioning(data_rm, rekordbox_music_path, rekordbox_music_version_check_path,
                         data_mm, my_music_path, filenames_wave, filenames_removed)
    version.check_new_rekordbox_file()
    version.get_version()
    version.expand_versions_of_existing_tracks()

    # initialize feature getters
    credentials = load_credentials(credential_path)
    sf = SpotifyFeatures(rb_data=data_rm, credentials=credentials['sp'])
    yf = YoutubeFeatures(rb_data=data_rm, credentials=credentials['yt'])
    progress = Progress()
    for fn in filenames_added:
        i = find(data_rm, 'File Name', fn)

        data_mm = load(my_music_path)
        sp_yt_features = {}

        sp_yt_features.update(data_rm[i])

        sf.get(i)
        sp_yt_features.update(sf.spotify_features)

        yf.get(i)
        sp_yt_features.update(yf.youtube_features)

        fi = FeaturesImprovement(sp_yt_features)
        fi.improve()

        sp_yt_features = fi.af.copy()

        # update tracks where connection errors occurred
        ce = ConnectionErrors(sp_yt_features, data_mm, data_rm, sf, yf)
        data_mm = ce.handle()

        data_mm.append(sp_yt_features)
        dump(data_mm, my_music_path)

        progress.show(filenames_added, fn)

    data_mm = load(my_music_path)
    popularity = Popularity(data_mm, my_music_path)
    popularity.get()

    wf = WaveFeatures(tracks_dir=tracks_dir, rb_data=data_rm)
    progress = Progress()
    for fn in filenames_wave:
        i_rm = find(data_rm, 'File Name', fn)
        i_mm = find(data_mm, 'File Name', fn)

        data_mm = load(my_music_path)
        all_features = data_mm[i_mm]

        wf.get(i_rm)
        all_features.update(wf.wave_features)

        all_features.update(version.set_version_column())

        del data_mm[i_mm]
        data_mm.append(all_features)
        dump(data_mm, my_music_path)

        progress.show(filenames_wave, fn)

