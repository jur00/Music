import json
import time
from datetime import datetime
import requests
import socket
import urllib3

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_credentials(kind='db', file='credentials.json'):
    with open(file, 'rb') as f:
        credentials = json.load(f)[kind]

    return credentials

def make_spotify():
    sp_credentials = get_credentials('sp')
    cid = sp_credentials['cid']
    secret = sp_credentials['secret']
    ccm = SpotifyClientCredentials(client_id=cid,
                                   client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=ccm)

    return sp

def make_youtube():
    api_key = get_credentials('yt')['api_key']
    youtube = build('youtube', 'v3', developerKey=api_key)
    browser_option = webdriver.ChromeOptions()
    browser_option.add_argument('headless')
    browser_option.add_argument('log-level = 2')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_option)

    return youtube, driver

def change_list_to_string(val):
    return ' | '.join(val)

def overrule_spotify_errors(sp_func, verbose=0, empty=None):
    if empty is None:
        empty = {}
    output = empty
    conn_error = True
    sleep_counter = 0
    while conn_error & (sleep_counter < 300):
        try:
            output = sp_func
            conn_error = False
        except (requests.exceptions.ReadTimeout,
                urllib3.exceptions.ReadTimeoutError,
                socket.timeout) as error:
            sleep_counter += 1
            if verbose > 0:
                print('got an error, trying again...', end='\r')
            time.sleep(1)

    return output

class Progress:

    def __init__(self):
        self.start = time.time()

    def show(self, loop_space, current_loop):
        n = len(loop_space)
        counter = loop_space.index(current_loop) + 1
        fraction_done = counter / n

        progress_precentage = str(round(fraction_done * 100, 2)) + "%"
        now = time.time()
        eta = self.start + ((now - self.start) / fraction_done)
        time_completed = datetime.fromtimestamp(eta).strftime("%Y-%m-%d %H:%M:%S")

        time_until_completed = ((now - self.start) / fraction_done) - (now - self.start)

        time_until_completed_hours = int(time_until_completed / 3600)
        time_until_completed_minutes = int(((time_until_completed / 3600) - time_until_completed_hours) * 60)
        time_until_completed_seconds = int(time_until_completed % 60)
        average_compute_time = (now - self.start) / counter
        avg_compute_time_minutes = int(average_compute_time / 60)
        avg_compute_time_seconds = int(average_compute_time % 60)
        print(f'Done: {counter} / {n} ({progress_precentage}), ETA: {time_completed}, Time until completed: {time_until_completed_hours} hours, {time_until_completed_minutes} minutes, {time_until_completed_seconds} seconds, Average time per loop: {avg_compute_time_minutes} minutes and {avg_compute_time_seconds} seconds',
              end='\r')
