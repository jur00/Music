from pathlib import Path
from functools import reduce

from joblib import load
import pandas as pd
import numpy as np

working_dir = 'D:\\Data Science\\Python zelfstudie\\Music'
tracks_dir = 'D:\\Data Science\\Lake\\music\\tracks_my\\'
file_dir = 'files\\data\\update_dataset'
my_music_fn = 'music_my.sav'
rekordbox_music_fn = 'music_rekordbox.txt'
credential_dir = ''
credential_fn = 'credentials.json'

my_music_path = Path(file_dir, my_music_fn)


def _get_end_artist(row):
    if row['Composer'] != '':
        return row['Composer']
    else:
        return row['Artist']


def load_df(my_music_path):
    return pd.DataFrame(load(my_music_path))


def filter_on_same_and_cols(x_df):
    return x_df.loc[x_df['sp_dif_type'] == 'same', ['Artist', 'Composer', 'sp_artist', 'sp_id', 'sp_dif_type']]


def create_end_artist_column(x_df):
    x_df['end_artist'] = x_df.apply(_get_end_artist, axis=1)
    return x_df


def get_all_my_artists(df):
    artists_list = [a.split(', ') for a in df['end_artist'].to_list() if a != '']
    my_artists_all = reduce(lambda x, y: x + y, artists_list)
    my_artists_all = list(map(str.lower, my_artists_all))
    my_artists_unique = np.unique(my_artists_all)

    return my_artists_unique


tmp = load_df(my_music_path)
df = (load_df(my_music_path)
      .pipe(filter_on_same_and_cols)
      .pipe(create_end_artist_column))

my_artists = get_all_my_artists(df)
