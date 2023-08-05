from unidecode import unidecode
import re
import time
import requests
import socket
import urllib3
import httplib2
from contextlib import contextmanager
import os
from datetime import datetime
import stat

import spotipy

def jaccard_similarity(test, real):
    intersection = set(test).intersection(set(real))
    union = set(test).union(set(real))
    return len(intersection) / len(union)

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def neutralize(string):
    return unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', string)).lower()

def overrule_connection_errors(func, verbose=1, empty=None):
    if empty is None:
        empty = {}
    output = empty
    conn_error = True
    sleep_counter = 0
    while conn_error & (sleep_counter < 300):
        try:
            output = func
            conn_error = False
        except (ConnectionError,
                ConnectionResetError,
                ConnectionRefusedError,
                ConnectionAbortedError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                urllib3.exceptions.ReadTimeoutError,
                urllib3.exceptions.ProtocolError,
                socket.timeout,
                spotipy.exceptions.SpotifyException,
                httplib2.error.ServerNotFoundError) as error:
            sleep_counter += 1
            if verbose > 0:
                print('got an error, trying again...', end='\r')
            time.sleep(1)

    return output, conn_error

def create_name_string(rb_data, i, name_parts):
    return neutralize(' '.join([v for k, v in rb_data[i].items() if k in name_parts]).strip())

def check_original(rb_data, i):
    return rb_data[i]['Album'] in ['', 'original mix']

def add_or_remove_original(comp_name):
    return re.sub('original mix', '', comp_name) if 'original mix' in comp_name else f'{comp_name} original mix'

@contextmanager
def set_dir(path):
    origin = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def on_rm_error(func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)

def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return ValueError('File Name does not exist.')

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

        average_compute_time = (now - self.start) / counter
        minutes = int(average_compute_time / 60)
        seconds = int(average_compute_time % 60)
        print('Done: {} / {} ({}), ETA: {}, Average time per loop: {} minutes and {} seconds'.
              format(counter, n, progress_precentage, time_completed, minutes, seconds),
              end='\r')
