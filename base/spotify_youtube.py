from base.helpers import retry

n_attempts = 50

# update_dataset
@retry(n_attempts=n_attempts, empty_output=[])
def get_spotify_audio_features(sp, sp_id):
    return sp.audio_features(sp_id)


@retry(n_attempts=n_attempts, empty_output={})
def search_spotify_tracks(sp, artist_track):
    return sp.search(q=artist_track, type="track", limit=50)


@retry(n_attempts=n_attempts, empty_output=None)
def get_youtube_link(driver, link):
    driver.get(link)


@retry(n_attempts=n_attempts, empty_output=[])
def find_youtube_elements(driver):
    return driver.find_elements('xpath', '//*[@id="video-title"]')


@retry(n_attempts=n_attempts, empty_output={})
def get_youtube_video_properties(youtube, yt_id):
    return youtube.videos().list(part='snippet,statistics,contentDetails', id=yt_id).execute()


# get_training_data
@retry(n_attempts=n_attempts, empty_output={})
def get_spotify_track(sp, sp_id):
    return sp.track(sp_id)


@retry(n_attempts=n_attempts, empty_output={})
def get_spotify_artist_genres(sp, artist_id):
    return sp.artist(artist_id)['genres']


@retry(n_attempts=n_attempts, empty_output=[])
def get_spotify_related_artists(sp, artist_id):
    related_artists = sp.artist_related_artists(artist_id)['artists']
    return [ra['id'] for ra in related_artists]

@retry(n_attempts=n_attempts, empty_output=[])
def get_spotify_recommendations(sp, track_id, limit):
    return sp.recommendations(seed_tracks=[track_id], limit=limit)['tracks']

@retry(n_attempts=n_attempts, empty_output=[])
def get_spotify_artist_top_tracks(sp, artist_id):
    return sp.artist_top_tracks(artist_id=artist_id)['tracks']