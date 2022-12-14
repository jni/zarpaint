from __future__ import annotations
import zarr
import os
import pathlib
from ._zarpaint import open_ts_meta


def zarr_tensorstore(path: str | pathlib.Path):
    if (str(path).endswith('.zarr') and os.path.isdir(path)
                and '.zarray' in os.listdir(path)):
        return lambda path: [(zarr.open(path), open_ts_meta(path), 'labels')]
