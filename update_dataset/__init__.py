from config import Config

import os
from pathlib import Path

from joblib import load, dump


from base.helpers import Progress
from update_dataset.helpers import find
from update_dataset.engineering import (RekordboxMusic, ExplorerInterruption,
                                        Disjoint, SpotifyFeatures, YoutubeFeatures, WaveFeatures,
                                        FeaturesImprovement, Popularity, Versioning, ConnectionErrors)


class UpdateDataset:

    def __init__(self):

        os.chdir(Config.working_dir)

        self.tracks_dir = Config.tracks_dir
        self.my_music_path = Path(Config.data_dir, Config.my_music_fn)
        self.rekordbox_music_path = Path(Config.data_dir, Config.rekordbox_music_fn)
        rb_fn, rb_ext = os.path.splitext(Config.rekordbox_music_fn)
        self.rekordbox_music_version_check_path = Path(Config.data_dir, f'{rb_fn}_version_check{rb_ext}')

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
        self.__quick_test = False

    def _load_data(self):
        print("Loading data, don't interrupt", end='\r')
        if not os.path.exists(self.my_music_path):
            dump([], self.my_music_path)
        self.data_mm = load(self.my_music_path)
        rm = RekordboxMusic(self.rekordbox_music_path)
        self.data_rm = rm.get()

    def _check_vocalness_feature_interruption(self):
        ei = ExplorerInterruption(self.data_rm, self.tracks_dir)
        ei.change_shortened_filenames()
        ei.empty_output_map()

    def _check_missing_tracks_in_dir(self):
        self.disjoint = Disjoint(self.data_rm, self.data_mm, self.tracks_dir, self.__quick_test)
        self.disjoint.check_missing_filenames_in_tracks_dir()

    def _check_added_removed_tracks(self):
        self.filenames_added = self.disjoint.get_added_tracks()
        self.filenames_removed = self.disjoint.get_removed_tracks()
        self.filenames_wave = self.disjoint.get_tracks_for_wave_analysis()

    def _calculate_popularity(self, data_mm):
        self.popularity = Popularity(data_mm, self.my_music_path)
        self.popularity.get()

    def _data_versioning(self):
        self.version = Versioning(self.data_rm, self.rekordbox_music_path, self.rekordbox_music_version_check_path,
                                  self.data_mm, self.my_music_path, self.filenames_added, self.filenames_removed)
        self.version.check_new_rekordbox_file()
        self.version.get_version()
        if self.version.new_version:
            self.version.expand_versions_of_existing_tracks()

    def _get_sp_yt_features(self):
        sf = SpotifyFeatures(rb_data=self.data_rm)
        yf = YoutubeFeatures(rb_data=self.data_rm)
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

            print("Dumping data, don't interrupt", end='\r')
            dump(data_mm, self.my_music_path)

            progress.show(fn, self.filenames_added)

    def _get_wave_features(self, data_mm):
        wf = WaveFeatures(tracks_dir=self.tracks_dir, rb_data=self.data_rm)
        progress = Progress()
        for fn in self.filenames_wave:
            i_rm = find(self.data_rm, 'File Name', fn)
            i_mm = find(data_mm, 'File Name', fn)

            data_mm = load(self.my_music_path)
            all_features = data_mm[i_mm]

            if not self.__quick_test:
                wf.get(i_rm)
                all_features.update(wf.wave_features)
            else:
                all_features.update({'wave_col': None})

            all_features.update(self.version.set_version_column())

            del data_mm[i_mm]
            data_mm.append(all_features)

            print("Dumping data, don't interrupt", end='\r')
            dump(data_mm, self.my_music_path)

            progress.show(fn, self.filenames_wave)

    def run(self, quick_test=False):
        self.__quick_test = quick_test

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
