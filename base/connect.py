import json

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def load_credentials(path):
    with open(path, 'rb') as f:
        credentials = json.load(f)

    return credentials

class SpotifyConnect:

    def __init__(self, credentials):
        self.sp = self.__make_spotify(credentials)

    @staticmethod
    def __make_spotify(credentials):
        cid = credentials['cid']
        secret = credentials['secret']
        ccm = SpotifyClientCredentials(client_id=cid,
                                       client_secret=secret)
        sp = spotipy.Spotify(client_credentials_manager=ccm)

        return sp

class YoutubeConnect:

    def __init__(self, credentials):
        self.youtube, self.driver = self.__make_youtube(credentials)

    @staticmethod
    def __make_youtube(credentials):
        youtube = build('youtube', 'v3', developerKey=credentials['api_key'])
        browser_option = webdriver.ChromeOptions()
        browser_option.add_argument('--no-sandbox')
        browser_option.add_argument('--headless')
        browser_option.add_argument('--disable-dev-shm-usage')
        browser_option.add_argument('--log-level=3')
        browser_option.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(service=Service(ChromeDriverManager(version='114.0.5735.90').install()), options=browser_option)

        return youtube, driver