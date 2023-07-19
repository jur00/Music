from pathlib import Path
from math import ceil

import pandas as pd
import numpy as np

from update_dataset.engineering import RekordboxMusic

# for testing purposes; creating small chunks of the rekordbox file, as if i start with only a small amount of tracks
# and gradually add tracks to the database

# initialize
file_dir = 'files'
rekordbox_music_fn = 'music_rekordbox.txt'
rb_path = Path(file_dir, rekordbox_music_fn)
rm = RekordboxMusic(rb_path)
data = rm.get()
df = pd.DataFrame(data)
df = df.sort_values(by='rb_duration')

# get indexes of shortest songs
shortest_idxs = list(df.index[:5])
shortest_idxs = [str(i+1) for i in shortest_idxs]
chunk_idxs = [shortest_idxs[:i] if i == 2 else shortest_idxs[1:i] for i in range(2, len(shortest_idxs))]
chunks = [chunk + ['#'] for chunk in chunk_idxs]

# read and write txt files
for i in range(len(chunks)):
    with open(rb_path, 'r', encoding='utf-16') as f:
        lines = []
        for line in f:
            if line.split('\t', maxsplit=1)[0] in chunks[i]:
                lines.append(line)

    rekordbox_music_chunk_fn = f"{rekordbox_music_fn.split('.')[0]}_chunk_{i}.{rekordbox_music_fn.split('.')[1]}"
    with open(Path(file_dir, rekordbox_music_chunk_fn), 'w', encoding='utf-16') as f:
        f.writelines(lines)
