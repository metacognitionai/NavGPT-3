"""Observation frames written to a run's live directory."""
import os
import tempfile
from pathlib import Path


def save_frame(path, blob):
    """Write atomically, so a viewer reading latest.jpg never sees a partial file."""
    path = Path(path)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as fh:
            temp = fh.name
            fh.write(blob)
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    return str(path)


def read_frame(location):
    return Path(location).read_bytes()


def frame_exists(location):
    return bool(location) and Path(location).is_file()
