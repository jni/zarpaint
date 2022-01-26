from __future__ import annotations

import os
import pathlib
from ._zarpaint import open_tensorstore, open_ts_meta


def zarr_tensorstore(path: str | pathlib.Path):
    if (str(path).endswith('.zarr') and os.path.isdir(path)
                and '.zarray' in os.listdir(path)):
        return lambda path: [
                (open_tensorstore(path), open_ts_meta(path), 'labels')
                ]
