from pathlib import Path

class Config:

    working_dir = 'D:\\Data Science\\Python zelfstudie\\Music'
    data_dir = 'files\\data\\'

    update_dir = f'{data_dir}update_dataset'
    tracks_dir = 'D:\\Data Science\\Lake\\music\\tracks_my\\'
    my_music_fn = 'music_my.sav'
    my_music_path = Path(update_dir, my_music_fn)
    rekordbox_music_fn = 'music_rekordbox.txt'

    training_data_dir = f'{data_dir}training_data'
    training_recommendations_fn = 'recommendations.sav'
    training_artist_tracks_fn = 'artist_tracks.sav'
    training_recommendations_path = Path(training_data_dir, training_recommendations_fn)
    training_artist_tracks_path = Path(training_data_dir, training_artist_tracks_fn)
    n_tracks_per_day = 140  # maximum n tracks spotify can analyse for recommendations/artisttoptracks

    credential_dir = ''
    credential_fn = 'credentials.json'
    credential_path = Path(credential_dir, credential_fn)
