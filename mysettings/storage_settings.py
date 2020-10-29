"""
Use this file to set paths where big files will get stored.
"""
from pathlib import Path

"""
Somewhere you can keep a lot of data for a long time.
"""
storage = Path('/gws/nopw/j04/bas_climate/users/champs/RiceHeat_demo')

if not storage.exists():
    raise RuntimeError(f'specified storage location does not exist: {storage}')
