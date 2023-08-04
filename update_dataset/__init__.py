import os
from pathlib import Path

from joblib import load, dump

from update_dataset.helpers import Progress
from update_dataset.engineering import (load_credentials, RekordboxMusic, ExplorerInterruption,
                                        Disjoint, SpotifyFeatures, YoutubeFeatures, WaveFeatures,
                                        FeaturesImprovement, Popularity, Versioning, ConnectionErrors)


class UpdateDataset:

    def __init__(self,
                 working_dir,
                 tracks_dir,
                 file_dir,
                 my_music_fn,
                 rekordbox_music_fn,
                 credential_dir,
                 credential_fn):

        os.chdir(working_dir)

        self.tracks_dir = tracks_dir
        self.my_music_path = Path(file_dir, my_music_fn)
        self.rekordbox_music_path = Path(file_dir, rekordbox_music_fn)
        self.credential_path = Path(credential_dir, credential_fn)

        self.data_mm = None
        self.data_rm = None
        self.added = None
        self.removed = None
        self.added_indexes_rm = None
        self.n_changes = None
        self.version = None
        self.sf = None
        self.yf = None
        self.wf = None

    def _load_data(self):
        self.data_mm = load(self.my_music_path)
        rm = RekordboxMusic(self.rekordbox_music_path)
        self.data_rm = rm.get()

    def _check_vocalness_feature_interruption(self):
        ei = ExplorerInterruption(self.data_rm, self.tracks_dir)
        ei.change_shortened_filenames()
        ei.empty_output_map()

    def _check_missing_tracks_in_dir(self):
        tracks_in_dir = os.listdir(self.tracks_dir)
        dj_dir = Disjoint(self.data_rm, tracks_in_dir, datatype2='list')
        copy_to_tracks_dir = dj_dir.not_in_data2()
        n_tracks_to_copy = len(copy_to_tracks_dir)
        if n_tracks_to_copy > 0:
            raise FileNotFoundError(f'{n_tracks_to_copy} tracks need to be copied to tracks_dir: {copy_to_tracks_dir}')

    def _check_added_removed_tracks(self):
        dj_data = Disjoint(self.data_rm, self.data_mm)
        self.added_indexes_rm = dj_data.get_indexes(type='not_in_data2')
        self.added = dj_data.not_in_data2()
        self.removed = dj_data.not_in_data1()
        self.n_changes = len(self.added) + len(self.removed)

    def _calculate_popularity(self, data_mm):
        self.popularity = Popularity(data_mm, self.my_music_path)
        self.popularity.get()

    def _data_versioning(self):
        self.version = Versioning(self.data_mm, self.my_music_path, self.added, self.removed)
        self.version.get_version()
        self.version.expand_versions_of_existing_tracks()

    def _init_feature_getters(self):
        credentials = load_credentials(self.credential_path)
        self.sf = SpotifyFeatures(rb_data=self.data_rm, credentials=credentials['sp'])
        self.yf = YoutubeFeatures(rb_data=self.data_rm, credentials=credentials['yt'])
        self.wf = WaveFeatures(tracks_dir=self.tracks_dir, rb_data=self.data_rm)

    def _get_all_features(self, i):
        all_features = {}

        all_features.update(self.data_rm[i])

        self.sf.get(i)
        all_features.update(self.sf.spotify_features)

        self.yf.get(i)
        all_features.update(self.yf.youtube_features)

        self.wf.get(i)
        all_features.update(self.wf.wave_features)

        fi = FeaturesImprovement(all_features)
        fi.improve()

        all_features = fi.af.copy()

        all_features.update(self.version.set_version_column())

        return all_features

    def _retry_connection_error_features(self, all_features, data_mm):
        ce = ConnectionErrors(all_features, data_mm, self.data_rm, self.sf, self.yf)
        data_mm = ce.handle()

        return data_mm

    def run(self):

        self._load_data()
        self._check_vocalness_feature_interruption()
        self._check_missing_tracks_in_dir()
        self._check_added_removed_tracks()

        if self.n_changes == 0:
            self._calculate_popularity(self.data_mm)
        else:
            self._data_versioning()
            self._init_feature_getters()

            progress = Progress()
            for i in self.added_indexes_rm:
                data_mm = load(self.my_music_path)
                all_features = self._get_all_features(i)
                data_mm = self._retry_connection_error_features(all_features, data_mm)
                data_mm.append(all_features)
                dump(data_mm, self.my_music_path)
                progress.show(self.added_indexes_rm, i)

            data_mm = load(self.my_music_path)
            self._calculate_popularity(data_mm)