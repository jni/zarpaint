from __future__ import annotations

import os
import pathlib
from ._zarpaint import open_zarr, open_ts_meta


def zarr_tensorstore(path: str | pathlib.Path):
    if (str(path).endswith('.zarr') and os.path.isdir(path)
                and '.zarray' in os.listdir(path)):
        return lambda p: [(open_zarr(p), open_ts_meta(path), 'labels')]
