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

# get exponential growing chunk indexes
chunk_idxs = [int(ceil(2.5 * 2**i)) for i in range(10)]
chunk_idxs[-1] = df.shape[0] - sum(chunk_idxs[:-1])
chunk_idxs = np.cumsum(chunk_idxs)

# set index being first at 1
df_index = df.index
df_index += 1

# split music_rekordbox file in more sub files
chunks = [list(df_index[:chunk_idx].astype(str)) + ['#'] for chunk_idx in chunk_idxs]
for i in range(len(chunks)):
    with open(rb_path, 'r', encoding='utf-16') as f:
        lines = []
        for line in f:
            if line.split('\t', maxsplit=1)[0] in chunks[i]:
                lines.append(line)

    rekordbox_music_chunk_fn = f"{rekordbox_music_fn.split('.')[0]}_chunk_{i}.{rekordbox_music_fn.split('.')[1]}"
    with open(Path(file_dir, rekordbox_music_chunk_fn), 'w', encoding='utf-16') as f:
        f.writelines(lines)
