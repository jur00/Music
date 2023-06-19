from pickle import load, dump
import pandas as pd

with open('files/music_my.sav', 'rb') as f:
    df = pd.DataFrame(load(f))

