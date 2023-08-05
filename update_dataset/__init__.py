import os
from pathlib import Path

from joblib import load, dump

from update_dataset.helpers import find, Progress
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
        self.disjoint = None
        self.filenames_added = None
        self.filenames_removed = None
        self.filenames_wave = None
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
        self.disjoint = Disjoint(self.data_rm, self.data_mm, self.tracks_dir)
        self.disjoint.check_missing_filenames_in_tracks_dir()

    def _check_added_removed_tracks(self):
        self.filenames_added = self.disjoint.get_added_tracks()
        self.filenames_removed = self.disjoint.get_removed_tracks()
        self.filenames_wave = self.disjoint.get_tracks_for_wave_analysis()

    def _calculate_popularity(self, data_mm):
        self.popularity = Popularity(data_mm, self.my_music_path)
        self.popularity.get()

    def _data_versioning(self):
        self.version = Versioning(self.data_mm, self.my_music_path, self.filenames_wave, self.filenames_removed)
        self.version.get_version()
        self.version.expand_versions_of_existing_tracks()

    def _get_sp_yt_features(self):
        credentials = load_credentials(self.credential_path)
        sf = SpotifyFeatures(rb_data=self.data_rm, credentials=credentials['sp'])
        yf = YoutubeFeatures(rb_data=self.data_rm, credentials=credentials['yt'])
        progress = Progress()
        for fn in self.filenames_added:
            i = find(self.data_rm, 'File Name', fn)

            data_mm = load(self.my_music_path)
            sp_yt_features = {}

            sp_yt_features.update(self.data_rm[i])

            sf.get(i)
            sp_yt_features.update(sf.spotify_features)

            yf.get(i)
            sp_yt_features.update(yf.youtube_features)

            fi = FeaturesImprovement(sp_yt_features)
            fi.improve()

            sp_yt_features = fi.af.copy()

            # update tracks where connection errors occurred
            ce = ConnectionErrors(sp_yt_features, data_mm, self.data_rm, sf, yf)
            data_mm = ce.handle()

            data_mm.append(sp_yt_features)
            dump(data_mm, self.my_music_path)

            progress.show(self.filenames_added, fn)

    def _get_wave_features(self, data_mm):
        wf = WaveFeatures(tracks_dir=self.tracks_dir, rb_data=self.data_rm)
        progress = Progress()
        for fn in self.filenames_wave:
            i_rm = find(self.data_rm, 'File Name', fn)
            i_mm = find(data_mm, 'File Name', fn)

            data_mm = load(self.my_music_path)
            all_features = data_mm[i_mm]

            wf.get(i_rm)
            all_features.update(wf.wave_features)

            all_features.update(self.version.set_version_column())

            del data_mm[i_mm]
            data_mm.append(all_features)
            dump(data_mm, self.my_music_path)

            progress.show(self.filenames_wave, fn)

    def run(self):

        self._load_data()

        self._check_vocalness_feature_interruption()

        self._check_missing_tracks_in_dir()

        self._check_added_removed_tracks()

        if self.disjoint.n_changes == 0:

            self._calculate_popularity(self.data_mm)

        else:
            self._data_versioning()

            self._get_sp_yt_features()

            data_mm = load(self.my_music_path)

            self._calculate_popularity(data_mm)

            self._get_wave_features(data_mm)
