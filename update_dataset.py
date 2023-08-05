from update_dataset import UpdateDataset

working_dir = 'D:\\Data Science\\Python zelfstudie\\Music'
tracks_dir = 'D:\\Data Science\\Lake\\music\\tracks_my\\'
file_dir = 'files\\data\\update_dataset'
my_music_fn = 'music_my.sav'
rekordbox_music_fn = 'music_rekordbox.txt'
credential_dir = ''
credential_fn = 'credentials.json'

updating = UpdateDataset(working_dir,
                         tracks_dir,
                         file_dir,
                         my_music_fn,
                         rekordbox_music_fn,
                         credential_dir,
                         credential_fn)
updating.run()
