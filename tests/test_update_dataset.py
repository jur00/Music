from pathlib import Path
import time
import os
import shutil

from joblib import load, dump
import pandas as pd
import numpy as np
import pytest

from update_dataset import UpdateDataset
from get_training_data.tracklist_old import UpdateArtists

working_dir = 'D:\\Data Science\\Python zelfstudie\\Music'
tracks_dir = 'D:\\Data Science\\Lake\\music\\test\\tracks_my\\'
file_dir = 'files\\test_data\\update_dataset'
my_music_fn = 'music_my.sav'
my_artists_fn = 'artists_my.sav'
my_music_path = Path(file_dir, my_music_fn)
my_artists_path = Path(file_dir, my_artists_fn)
rekordbox_music_fn = 'music_rekordbox.txt'
credential_dir = ''
credential_fn = 'credentials.json'
dump([], Path(working_dir, my_music_path))
rekordbox_music_chunk_fns = [f'music_rekordbox_chunk_{i}.txt' for i in range(3)]
version_check_files = [file for file in os.listdir(file_dir) if file.endswith('version_check.txt')]
for version_check_file in version_check_files:
    os.remove(f'{file_dir}/{version_check_file}')

for rekordbox_music_chunk_fn in rekordbox_music_chunk_fns:
    if not os.path.exists(f'{file_dir}/{rekordbox_music_chunk_fn}'):
        os.rename(f'{file_dir}/{rekordbox_music_fn}',
                  f'{file_dir}/{rekordbox_music_chunk_fn}')

for rekordbox_music_chunk_fn in rekordbox_music_chunk_fns:

    # set current rekordbox chunk as main rekordbox file
    if os.path.exists(f'{file_dir}/{rekordbox_music_chunk_fn}'):
        os.rename(f'{file_dir}/{rekordbox_music_chunk_fn}',
                  f'{file_dir}/{rekordbox_music_fn}')

    update_dataset = UpdateDataset(working_dir,
                                   tracks_dir,
                                   file_dir,
                                   my_music_fn,
                                   rekordbox_music_fn,
                                   credential_dir,
                                   credential_fn)
    update_dataset.run(quick_test=True)

    time.sleep(5)

    update_artists = UpdateArtists(working_dir,
                                   file_dir,
                                   my_music_fn,
                                   my_artists_fn,
                                   credential_dir,
                                   credential_fn)
    update_artists.run()

    # reset original rekordbox chunk name
    os.rename(f'{file_dir}/{rekordbox_music_fn}',
              f'{file_dir}/{rekordbox_music_chunk_fn}')

    time.sleep(5)

tmp = pd.DataFrame(load(my_artists_path))
t = pd.DataFrame(load(my_music_path)).sort_values(by='rb_duration')

@pytest.fixture
def df_music():
    return pd.DataFrame(load(my_music_path)).sort_values(by='rb_duration')


@pytest.fixture
def df_artists():
    return pd.DataFrame(load(my_artists_path))


def test_music_versions(df_music):
    version_matrix = df_music[[col for col in df_music.columns if col.startswith('version_')]].to_numpy()
    expected = np.array([[1, 0, 0],
                         [1, 1, 1],
                         [0, 1, 1],
                         [0, 0, 1]])
    assert (version_matrix == expected).all()
