import requests
import socket
import urllib3
import httplib2
import time
from functools import wraps
from datetime import datetime
from difflib import SequenceMatcher
import webbrowser
from scipy.stats import gaussian_kde
import numpy as np

import plotly
from plotly.io._base_renderers import BaseHTTPRequestHandler, HTTPServer

import spotipy

def retry(**fkwargs):
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            conn_error = None
            for attempt in range(fkwargs['n_attempts']):
                try:
                    conn_error = False
                    return func(*args, **kwargs), conn_error
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
                        httplib2.error.ServerNotFoundError):
                    conn_error = True
                    print(f"Connection error, trying again...{attempt+1} / {fkwargs['n_attempts']}", end='\r')
                    time.sleep(1)

            return fkwargs['empty_output'], conn_error
        return wrapper
    return _decorator

def string_similarity(a, b):
    """
    Retourne le ratio de ressemble entre deux strings
    :param a:
    :param b:
    :return:
    """
    return SequenceMatcher(None, a, b).ratio()

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

def jaccard_similarity(test, real):
    intersection = set(test).intersection(set(real))
    union = set(test).union(set(real))
    return len(intersection) / len(union)

class Progress:

    def __init__(self):
        self.start = time.time()

    def show(self, iteration, iterator):
        n = len(iterator)
        counter = iterator.index(iteration) + 1
        fraction_done = counter / n

        progress_percentage = str(round(fraction_done * 100, 2)) + "%"
        now = time.time()
        eta = self.start + ((now - self.start) / fraction_done)
        time_completed = datetime.fromtimestamp(eta).strftime("%Y-%m-%d %H:%M:%S")
        delta = datetime.fromtimestamp(eta) - datetime.fromtimestamp(now)
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_to_go = ''
        if days > 0:
            time_to_go += f'{days} days, '
        if hours > 0:
            time_to_go += f'{hours} hours, '
        if minutes > 0:
            time_to_go += f'{minutes} minutes and '

        time_to_go += f'{seconds} seconds'

        average_compute_time = (now - self.start) / counter
        avg_minutes = int(average_compute_time / 60)
        avg_seconds = int(average_compute_time % 60)
        print(f'Done: {counter} / {n} ({progress_percentage}), ETA: {time_completed} ({time_to_go}), '
              f'Average time per loop: {avg_minutes} minutes and {avg_seconds} seconds',
              end='\r')


def open_html_in_browser(html, using=None, new=0, autoraise=True):
    """
    Display html in a web browser without creating a temp file.
    Instantiates a trivial http server and uses the webbrowser module to
    open a URL to retrieve html from that server.
    Parameters
    ----------
    html: str
        HTML string to display
    using, new, autoraise:
        See docstrings in webbrowser.get and webbrowser.open
    """
    if isinstance(html, str):
        html = html.encode("utf8")

    browser = None

    if using is None:
        browser = webbrowser.get(None)
    else:
        if not isinstance(using, tuple):
            using = (using,)
        for browser_key in using:
            try:
                browser = webbrowser.get(browser_key)
                if browser is not None:
                    break
            except webbrowser.Error:
                pass

        if browser is None:
            raise ValueError("Can't locate a browser with key in " + str(using))

    class OneShotRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            bufferSize = 1024 * 1024
            for i in range(0, len(html), bufferSize):
                self.wfile.write(html[i : i + bufferSize])

        def log_message(self, format, *args):
            # Silence stderr logging
            pass

    server = HTTPServer(("127.0.0.1", 55002), OneShotRequestHandler)  # fixed port number
    browser.open(
        "http://127.0.0.1:%s" % server.server_port, new=new, autoraise=autoraise
    )

    server.handle_request()

def adaptive_linspace(data, num_points):
    kde = gaussian_kde(data)
    x_min, x_max = data.min(), data.max()
    x_range = np.linspace(x_min, x_max, 1000)
    density = kde(x_range)
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]

    num_points_wider = (num_points * 2) + 1
    uniform_cdf_points = np.linspace(0, 1, num_points_wider)[1::2]
    adaptive_points = np.interp(uniform_cdf_points, cdf, x_range)

    return adaptive_points