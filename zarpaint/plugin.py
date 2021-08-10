from __future__ import annotations

import os
import pathlib
from napari_plugin_engine import napari_hook_implementation
from ._zarpaint import create_labels, open_tensorstore, open_ts_meta
from ._dims_chooser import DimsSorter, set_axis_labels
from ._watershed import watershed_split
from ._add_3d_points import add_points_3d_with_alt_click


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return create_labels


@napari_hook_implementation(
    specname='napari_experimental_provide_dock_widget'
)
def dims_sorter():
    return DimsSorter


@napari_hook_implementation(
    specname='napari_experimental_provide_dock_widget'
)
def axis_labels():
    return set_axis_labels


@napari_hook_implementation(
    specname='napari_experimental_provide_dock_widget'
)
def watershed():
    return watershed_split


@napari_hook_implementation(
    specname='napari_get_reader'
)
def zarr_tensorstore(path: str | pathlib.Path):
    if (str(path).endswith('.zarr')
            and os.path.isdir(path) and '.zarray' in os.listdir(path)):
        return lambda path: [(open_tensorstore(path), open_ts_meta(path), 'labels')]


@napari_hook_implementation(
    specname='napari_experimental_provide_dock_widget'
)
def _add_points_callback():
    return add_points_3d_with_alt_click
