"""
Use this file to set paths where big files will get stored.
"""
from pathlib import Path

"""
Somewhere you can keep a lot of data for a long time.
"""
storage_big = Path('/gws/nopw/j04/bas_climate/users/champs/')

"""
Somewhere you can write quickly.
"""
storage_scratch = Path('/work/scratch-pw/champs/')

for storage in (storage_big, storage_scratch):
    if not storage.exists():
        raise RuntimeError(f'specified storage location does not exist: {storage}')
