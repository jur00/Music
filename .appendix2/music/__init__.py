import os
import json
from datetime import datetime

from music.other import get_credentials, make_spotify, make_youtube
from music.feature_engineering import RekordboxDataset, FeatureEngineering
from music.versioning import LastVersion, Disjoints, NewVersion, UpdateRekordbox
from music.database import MySQLdb

db_config = get_credentials('db')

class Music:

    def __init__(self,
                 db_config):
        self.db_config = db_config
        self.sp = make_spotify()
        self.youtube, self.driver = make_youtube()

    def update_my_database(self):
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        raw_data_dir = '.\\files\\'
        lake_dir = '..\\..\\Lake\\'
        my_tracks_dir = lake_dir + 'music\\tracks_my\\'
        rekordbox_filename = 'music_rekordbox.txt'

        rekordbox = RekordboxDataset(raw_data_dir=raw_data_dir,
                                     rekordbox_filename=rekordbox_filename)
        rekordbox.clean_rekordbox_data()
        df_raw_rekordbox = rekordbox.rekordbox_data
        db = MySQLdb(self.db_config)
        df_db_id_old = db.load_table('tracks_my_id')
        update_rekordbox = UpdateRekordbox(db, df_db_id_old, df_raw_rekordbox)
        update_rekordbox.update()

        last_version = LastVersion()
        last_version.get_last_database_version(df_db_id_old)

        disjoints = Disjoints(raw_data_dir=raw_data_dir,
                              lake_dir=lake_dir,
                              raw_df_rekordbox=df_raw_rekordbox)
        anything_changed = disjoints.check_if_anything_changed(df_db_id_old, last_version.database)
        if anything_changed:
            disjoints.rekordbox_with_lake()
            if len(disjoints.add_to_lake) != 0:
                raise FileNotFoundError(f'Before proceeding, add the following files to your Lake '
                                        f'({os.path.abspath(lake_dir)}): {disjoints.add_to_lake}')

            df_db_rekordbox_old = db.load_table('tracks_my_rekordbox')
            disjoints.rekordbox_with_db(df_db_id_old, df_db_rekordbox_old, last_version.database)

            new_version = NewVersion(last_version.database,
                                     disjoints.all_deleted_filenames_from_rb_ever,
                                     disjoints.added_filenames_to_rb,
                                     disjoints.all_deleted_ids_from_rb_ever)
            new_version.insert_new_tracks_to_db_id(df_db_id_old)
            new_version.create_new_version_column_in_db_id()
            new_version.change_deleted_filename_values_in_new_version_column()

            df_db_id_new = db.load_table('tracks_my_id')

            disjoints.get_just_deleted_tracknames(df_db_id_new, df_db_rekordbox_old, last_version.database,
                                                  new_version.new_version)
            new_version.insert_new_tracks_to_db_rekordbox(df_db_id_new, df_raw_rekordbox,
                                                          disjoints.added_filenames_to_rb)

            df_db_rekordbox_new = db.load_table('tracks_my_rekordbox')
            new_ids = new_version.get_new_ids(df_db_id_new, df_db_id_old)
            df_db_rb_added = df_db_rekordbox_new.loc[df_db_rekordbox_new['id'].isin(new_ids)]
            df_db_rb_added = df_db_rb_added.merge(df_db_id_new[['id', 'File Name']], how='left', on='id')

            ids_no_features_yet = list(set(df_db_id_new['id']) - set(db.get_spotify_ids()))
            df_db_no_features_yet = df_db_rb_added.loc[df_db_rb_added['id'].isin(ids_no_features_yet)]

            features = FeatureEngineering(rekordbox_data=df_db_rb_added,
                                          tracks_dir=my_tracks_dir,
                                          db=db,
                                          sp=self.sp,
                                          youtube=self.youtube,
                                          driver=self.driver)
            features.collect_create_and_write_one_by_one()

            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            db.insert_process_version(new_version.new_version, start_time, end_time)
        else:
            print('Nothing changed. No new version will be created.')

    def statistics_dashboard(self):
        pass

    def get_more_training_data(self):
        pass

    def train_models(self):
        pass

    def get_fingerprints(self):
        pass
