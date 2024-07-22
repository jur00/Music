import datetime
import time
import re
import os

from joblib import load, dump
import pandas as pd
from unidecode import unidecode
from tkinter import Tk
import pyperclip
import pyautogui as pyauto

from base.helpers import string_similarity
from config import Config


class DownloadSoulseek:

    def __init__(self):

        self.tk = Tk()
        self.df = None

    def add_download_status_columns(self):
        if 'attempted' not in self.df.columns:
            self.df = self.df.assign(attempted=False)
        if 'datetime_attempted' not in self.df.columns:
            self.df = self.df.assign(datetime_attempted=None)
        if 'track_file_name' not in self.df.columns:
            self.df = self.df.assign(track_file_name='')
        if 'download_completed' not in self.df.columns:
            self.df = self.df.assign(download_completed=False)
        if 'datetime_downloaded' not in self.df.columns:
            self.df = self.df.assign(datetime_downloaded=None)

    def split_downloaded_attempted_tracks(self):
        df_download = self.df.loc[~self.df['attempted']].reset_index(drop=True)
        df_attempted = self.df.loc[self.df['attempted']].reset_index(drop=True)
        return df_download, df_attempted

    def _create_compare_names(self, i):
        artist_track_name = self.df.loc[i, 'track_artist_name']
        track_name = unidecode(
            re.sub(r'[^a-zA-Z 0-9À-ú]+', '', str(self.df.loc[i, 'track_name']))
        ).lower()
        duration_ms = self.df.loc[i, 'track_duration_ms']
        return artist_track_name, track_name, duration_ms

    @staticmethod
    def _search_soulseek(track):
        # for downloading via soulseek, open soulseek in full-screen and navigate to the
        # 'Manual Search' section. the
        # location parameters of the mouse-clicks depend on the screen resulution.
        # I use a standard (1920x1080) resolution.
        time.sleep(5)
        pyperclip.copy(track)
        pyauto.click((53, 150))
        pyauto.hotkey("ctrl", "v")
        pyauto.press('enter')

        # before you screen all results, let the system sleep for 6 seconds minimum, in order
        # to have a static search list.
        time.sleep(10)  # 15

    def _gather_soulseek_result_info(self, Y, kind):
        feature_x_coord = {
            'file': 900,
            'size': 1600,
            'attributes': 1750
        }
        x = feature_x_coord[kind]
        pyauto.click((x, Y))
        pyauto.hotkey('ctrl', 'c')
        time.sleep(.1)
        return self.tk.clipboard_get()

    def _create_soulseek_df(self, Y=254, row_height=13, nrows=56):

        empty_row_count = 0
        soulseek_results = []
        for i in range(nrows):
            soulseek_info = {result_info: self._gather_soulseek_result_info(Y, result_info) for result_info in
                             ['file', 'size', 'attributes']}
            empty_row = all(v == soulseek_info['file'] for v in soulseek_info.values())
            if empty_row:
                empty_row_count += 1
                if empty_row_count == 3:
                    break
            else:
                soulseek_info.update({'Y': Y})
                soulseek_results.append(soulseek_info)
                empty_row_count = 0
            Y += row_height

        return pd.DataFrame(soulseek_results)

    @staticmethod
    def _filter_soulseek_df(xdf):
        less_than_hour_mask = xdf['attributes'].apply(lambda x: 'h' not in x)
        normal_size_mask = xdf['size'].apply(lambda x: ',' in x)
        mp3_mask = xdf['file'].str.endswith('.mp3')
        attributes_mask = xdf['attributes'].apply(lambda x: ',' in x)
        keep_mask = (less_than_hour_mask & normal_size_mask & mp3_mask & attributes_mask)

        return xdf.loc[keep_mask].reset_index(drop=True)

    @staticmethod
    def _clean_soulseek_df(xdf):
        xdf = (
            xdf
            .assign(
                **xdf['attributes'].str.split(', ', n=1, expand=True).rename(columns={0: 'bitrate', 1: 'duration_ms'})
            )
            .assign(
                bitrate=lambda xxdf: xxdf['bitrate'].str.replace('kbps', '').astype(int),
                duration_ms=lambda xxdf: xxdf['duration_ms'].str.replace('s', '').str.split('m')
            )
            .assign(
                duration_ms=lambda xxdf: xxdf['duration_ms'].apply(lambda x: 1000 * ((int(x[0]) * 60) + int(x[1]))),
                file_clean=lambda xxdf: xxdf['file']
                    .str.rsplit('.', 1)
                    .str[0]
                    .str.replace('_', ' ')
                    .str.replace('-', ' ')
                    .apply(lambda x: re.sub(' +', '', x))
                    .apply(lambda x: unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', x)))
                    .str.lower()
            )
            .drop(columns='attributes')
        )
        return xdf

    @staticmethod
    def _compare_names(xdf, artist_track_name, track_name):
        artist_track_name_original = artist_track_name + ' original mix'
        track_name_original = track_name + ' original mix'
        comparisons = [artist_track_name, artist_track_name_original, track_name, track_name_original]
        similarity_columns = []
        for i, comparison in enumerate(comparisons):
            similarity_column = f'name_similarity_{i}'
            xdf = xdf.assign(**{similarity_column: xdf['file_clean'].apply(
                lambda x: string_similarity(x, comparison))
            })
            similarity_columns.append(similarity_column)

        xdf = (
            xdf
            .assign(name_similarity=xdf[similarity_columns].max(axis=1))
            .drop(columns=similarity_columns)
        )
        return xdf

    @staticmethod
    def _compare_durations(xdf, duration_ms):
        longer = xdf['duration_ms'] >= duration_ms
        shorter = xdf['duration_ms'] < duration_ms
        duration_similarity = pd.concat([
            xdf.loc[shorter, 'duration_ms'] / duration_ms,
            duration_ms / xdf.loc[longer, 'duration_ms']
        ]).sort_index()
        xdf = xdf.assign(duration_similarity=duration_similarity)
        return xdf

    @staticmethod
    def _keep_only_high_similarity_scores(xdf, score):
        score_mask = xdf['similarity_score'] > score
        return xdf.loc[score_mask].sort_values(by='similarity_score').reset_index(drop=True)

    def _download_track(self, df):
        # if there was no instant download, download the candidate with the highest similarity score.
        Y = df.loc[0, 'Y']
        pyauto.click((900, Y), clicks = 2)
        pyauto.hotkey("ctrl", "c")
        time.sleep(.2)
        file = self.tk.clipboard_get()
        return file

    def soulseek_interaction(self, df_download, i):

        artist_track_name, track_name, duration_ms = self._create_compare_names(i)

        self._search_soulseek(artist_track_name)

        df_soulseek = (
            self._create_soulseek_df()
                .pipe(self._filter_soulseek_df)
        )

        df_download.loc[i, 'attempted'] = True
        df_download.loc[i, 'datetime_attempted'] = datetime.datetime.now()

        if df_soulseek.shape[0] > 0:
            df_soulseek = (
                df_soulseek
                .pipe(self._clean_soulseek_df)
                .pipe(self._compare_names, artist_track_name, track_name)
                .pipe(self._compare_durations, duration_ms)
                .assign(similarity_score=lambda xdf: (xdf['name_similarity'] + xdf['duration_similarity'] * 2) / 3)
                .pipe(self._keep_only_high_similarity_scores, .9)
            )

        if df_soulseek.shape[0] == 0:
            file_name = self._download_track(df_soulseek)
            df_download.loc[i, 'track_file_name'] = file_name
            df_download.loc[i, 'datetime_downloaded'] = datetime.datetime.now()

        return df_download

    @staticmethod
    def check_completed_downloads(xdf):
        downloaded_tracks = os.listdir(Config.training_tracks_dir)
        xdf = xdf.assign(download_completed=lambda xxdf: xxdf['track_file_name'].isin(downloaded_tracks))
        return xdf

    def run(self):

        still_tracks_to_download = True
        while still_tracks_to_download:

            self.df = load(Config.training_tracklist_path)

            self.add_download_status_columns()

            df_download, df_attempted = self.split_downloaded_attempted_tracks()

            df_download = self.soulseek_interaction(df_download, 0)

            df = pd.concat([df_attempted, df_download], ignore_index=True)

            df = df.pipe(self.check_completed_downloads)

            dump(df, Config.training_tracklist_path)

            n_tracks_to_go = df_download.shape[0]

            still_tracks_to_download = n_tracks_to_go > 0

            print(f'{n_tracks_to_go} tracks to go', end='\r')



