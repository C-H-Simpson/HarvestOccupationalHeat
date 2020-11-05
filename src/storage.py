import os
from pathlib import Path

try:
    storage = Path(os.environ['STORAGE'])
    assert(storage.exists())
except:
    raise FileNotFoundError('You must set a valid storage location, using the STORAGE environment variable')
