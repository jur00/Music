
my_genres = reduce(lambda x, y: x + y, df_artist_ids['genres'].to_list())
my_genres_unique = np.unique(my_genres)

related_artists = reduce(lambda x, y: x + y, df_artist_ids['related_artists'].to_list())
related_artists_unique = np.unique(related_artists)
related_artist_counts = Counter(related_artists)

[sp.artist(list(related_artist_counts.keys())[i])['name'] for i in range(100)]