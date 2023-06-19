import pandas as pd
from joblib import load, dump
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from unidecode import unidecode
import re
import time
import requests
import librosa
from spafe.features.gfcc import gfcc
from entropy_estimators import continuous
from scipy.stats.mstats import gmean
from scipy.stats import kurtosis, skew, beta
import subprocess
import shutil
import warnings
import os
from contextlib import contextmanager
import stat
import json


class Credentials:

    def __init__(self, directory, filename, api):
        self.file_location = f'{directory}\\{filename}'
        self.api = api

        self.credentials = None

    def get(self):
        with open(self.file_location, 'rb') as f:
            self.credentials = json.load(f)[self.api]


class MyMusic:

    def __init__(self, directory, filename):
        self.file_location = f'{directory}\\{filename}'

        self.data = None

    def get(self):
        self._load()

    def _load(self):
        self.data = load(self.file_location)


class RekordboxMusic:

    def __init__(self,
                 directory,
                 filename):

        self.file_location = f'{directory}\\{filename}'

        self.sp = None
        self.driver = None
        self.youtube = None
        self.df = None
        self.data = None

    def get(self):
        self._load_txt()
        self._remove_duplicated_filenames()
        self._change_column_name()
        self._fill_missing()
        self._add_genre_bool_columns()
        self._quantize_rating()
        self._quantize_duration()
        self._quantize_bpm()
        self._categorize_origin_type()
        self._drop_rownumber_column()
        self._to_records()

    def _load_txt(self):
        self.df = pd.read_csv(self.file_location,
                              sep=None, header=0,
                              encoding='utf-16',
                              engine='python')

    def _remove_duplicated_filenames(self):
        self.df = self.df.loc[~self.df.duplicated(subset='File Name'), :]
        self.df = self.df.reset_index(drop=True)

    def _change_column_name(self):
        self.df = self.df.rename(columns={'#': 'row'})

    def _fill_missing(self):
        self.df['row'] = list(range(1, (self.df.shape[0] + 1)))
        self.df['Composer'] = self.df['Composer'].fillna('')
        self.df['Album'] = self.df['Album'].fillna('')
        self.df['Label'] = self.df['Label'].fillna('')

    def _add_genre_bool_columns(self):
        genres = ['Afro Disco', 'Balearic', 'Cosmic', 'Disco', 'Italo Disco', 'Nu Disco',
                  'Acid House', 'Deep House', 'House', 'Indie', 'Techno', 'Nostalgia', 'Old Deep House',
                  'Offbeat']
        self.df['Comments'] = self.df['Comments']. \
            str.lstrip(' /* '). \
            str.rstrip(' */'). \
            str.split(' / ')
        for g in genres:
            m = []
            for i in range(len(self.df['Comments'])):
                m.append(g in self.df['Comments'][i])

            self.df[g.replace(' ', '_')] = m

        del self.df['Comments']

    def _quantize_rating(self):
        self.df['Rating'] = self.df['Rating'].str.rstrip().str.len()

    def _quantize_duration(self):
        minutes = self.df['Time'].str.rpartition(":")[0].astype(int) * 60
        seconds = self.df['Time'].str.rpartition(":")[2].astype(int)
        self.df['rb_duration'] = (minutes + seconds) * 1000

    def _quantize_bpm(self):
        if self.df.dtypes['BPM'] == str:
            self.df['BPM'] = self.df['BPM'].str.replace(',', '.').astype(float)

    def _categorize_origin_type(self):
        self.df['track_kind'] = 'original'
        self.df.loc[~self.df['Label'].isin(['', ' ']), 'track_kind'] = 'remix'
        self.df.loc[~self.df['Album'].str.lower().isin(
            ['', ' ', 'original', 'original mix']
        ), 'track_kind'] = 'version'

    def _drop_rownumber_column(self):
        self.df.drop('row', axis=1, inplace=True)

    def _to_records(self):
        self.data = self.df.to_dict('records')


class Disjoint:

    def __init__(self, data1,
                 data2,
                 feature='File Name'):
        self.data1 = data1
        self.data2 = data2
        self.feature = feature

        self.not_in_data1 = None
        self.not_in_data2 = None

    def not_in_data1(self):
        self.not_in_data1 = list(self._check_filenames(self.data2) -
                                 self._check_filenames(self.data1))

    def not_in_data2(self):
        self.not_in_data2 = list(self._check_filenames(self.data1) -
                                 self._check_filenames(self.data2))

    def _check_filenames(self, data):
        len_data = len(data)
        filenames_data = set([data[i][self.feature] for i in range(len_data)])

        return filenames_data


class SpotifyFeatures:

    def __init__(self, credentials):
        self.credentials = credentials

        self.sp = None

    def _make_spotify(self):
        cid = self.credentials['cid']
        secret = self.credentials['secret']
        ccm = SpotifyClientCredentials(client_id=cid,
                                       client_secret=secret)
        self.sp = spotipy.Spotify(client_credentials_manager=ccm)

