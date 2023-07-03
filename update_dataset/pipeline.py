import os

from update_dataset.engineering import (Credentials, MyMusic, RekordboxMusic, Disjoint,
                                        SpotifyFeatures, YoutubeFeatures, WaveFeatures,
                                        FeaturesImprovement)

tracks_dir = 'D:\\Data Science\\Lake\\music\\tracks_my\\'

credentials_sp = Credentials(directory='', filename='credentials.json', api='sp')
credentials_sp.get()
credentials_spotify = credentials_sp.credentials

credentials_yt = Credentials(directory='', filename='credentials.json', api='yt')
credentials_yt.get()
credentials_youtube = credentials_yt.credentials

os.chdir('D:\\Data Science\\Python zelfstudie\\Music')

mm = MyMusic(directory='files', filename='music_my.sav')
mm.get()

rm = RekordboxMusic(directory='files', filename='music_rekordbox.txt')
rm.get()

dj = Disjoint(rm.data, mm.data)
get_features_indexes = dj.get_indexes(type='not_in_data2')

sf = SpotifyFeatures(rb_data=rm.data, credentials=credentials_spotify)
yf = YoutubeFeatures(rb_data=rm.data, credentials=credentials_youtube)
wf = WaveFeatures(tracks_dir=tracks_dir, rb_data=rm.data)

all_features = {}
i = 1950

all_features.update(rm.data[i])

sf.get(i)
all_features.update(sf.spotify_features)

yf.get(i)
all_features.update(yf.youtube_features)

wf.get(i)
all_features.update(wf.wave_features)

fi = FeaturesImprovement(all_features)
fi.improve()
fi.af