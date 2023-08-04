from pathlib import Path
import time

from joblib import load, dump
import pandas as pd
import numpy as np
import pytest

from update_dataset import UpdateDataset

working_dir = 'D:\\Data Science\\Python zelfstudie\\Music'
tracks_dir = 'D:\\Data Science\\Lake\\music\\test\\tracks_my\\'
file_dir = 'files\\test_data\\update_dataset'
my_music_fn = 'music_my.sav'
my_music_path = Path(file_dir, my_music_fn)
rekordbox_music_fns = [f'music_rekordbox_chunk_{i}.txt' for i in range(3)]
credential_dir = ''
credential_fn = 'credentials.json'
dump([], Path(working_dir, my_music_path))

@pytest.fixture
def df():
    for rekordbox_music_fn in rekordbox_music_fns:
        updating = UpdateDataset(working_dir,
                                 tracks_dir,
                                 file_dir,
                                 my_music_fn,
                                 rekordbox_music_fn,
                                 credential_dir,
                                 credential_fn)
        updating.run()
        time.sleep(5)

    return pd.DataFrame(load(my_music_path)).sort_values(by='rb_duration')

def test_versions(df):
    version_matrix = df[[col for col in df.columns if col.startswith('version_')]].to_numpy()
    expected = np.array([[1, 0, 0],
                         [1, 1, 1],
                         [0, 1, 1],
                         [0, 0, 1]])
    assert (version_matrix == expected).all()
