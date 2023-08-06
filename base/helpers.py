import requests
import socket
import urllib3
import httplib2
import time
from functools import wraps

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
