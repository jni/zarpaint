from __future__ import annotations

import os
import magicgui
import napari
import pathlib
from napari_plugin_engine import napari_hook_implementation
from ._zarpaint import create_labels, open_tensorstore, open_ts_meta
from ._dims_chooser import DimsSorter, set_axis_labels
from ._watershed import watershed_split
from ._add_3d_points import find_midpoint_of_first_segment


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

    @magicgui.magic_factory
    def add_points_3d_with_alt_click(
        labels: napari.layers.Labels,
        points: napari.layers.Points,
    ):
        pts_world2data = points._transforms[1:3].simplified.inverse

        @labels.mouse_drag_callbacks.append
        def click_callback(layer, event):
            if not (
                len(event.modifiers) == 1
                and event.modifiers[0].name == 'Alt'
            ):
                return
            world_click_coordinates = find_midpoint_of_first_segment(
                layer, event
            )
            pts_coordinates = pts_world2data(world_click_coordinates)
            points.add(pts_coordinates)
    return add_points_3d_with_alt_click
