from __future__ import annotations

import ast
import os
import yaml

from magicgui import magic_factory
import dask.array as da
import napari
import numpy as np
import pathlib
from pathlib import Path
import tensorstore as ts
import zarr
import toolz as tz


@tz.curry
def _set_default_labels_path(widget, source_image_event):
    source_image = source_image_event.value
    if (hasattr(source_image, 'source')  # napari <0.4.8
            and source_image.source.path is not None):
        source_path = pathlib.Path(source_image.source.path)
        if source_path.suffix != '.zarr':
            labels_path = source_path.with_suffix('.zarr')
        else:
            labels_path = source_path.with_suffix('.labels.zarr')
        widget.labels_file.value = labels_path


def _on_create_labels_init(widget):
    widget.source_image.changed.connect(_set_default_labels_path(widget))


def create_ts_meta(labels_file: pathlib.Path, metadata):
    """Create bespoke metadata yaml file within zarr array."""
    fn = os.path.join(labels_file, '.naparimeta.yml')
    with open(fn, 'w') as fout:
        for key, val in metadata.items():
            if type(val) == np.ndarray:
                if np.issubdtype(val.dtype, np.floating):
                    metadata[key] = list(map(float, val))
                else:
                    metadata[key] = list(map(int, val))
        yaml.dump(metadata, fout)


def open_ts_meta(labels_file: pathlib.Path) -> dict:
    """Open bespoke metadata yaml file within zarr array, if present."""
    fn = os.path.join(labels_file, '.naparimeta.yml')
    meta = {}
    if os.path.exists(fn):
        with open(fn, 'r') as fin:
            meta = yaml.safe_load(fin)
    return meta


def open_tensorstore(labels_file: pathlib.Path, *, shape=None, chunks=None):
    if not os.path.exists(labels_file):
        zarr.open(
                str(labels_file),
                mode='w',
                shape=shape,
                dtype=np.uint32,
                chunks=chunks,
                )
    # read some of the metadata for tensorstore driver from file
    labels_temp = zarr.open(str(labels_file), mode='r')
    metadata = {
            'dtype': labels_temp.dtype.str,
            'order': labels_temp.order,
            'shape': labels_temp.shape,
            }

    dir, name = os.path.split(labels_file)
    labels_ts_spec = {
            'driver': 'zarr',
            'kvstore': {
                    'driver': 'file',
                    'path': dir,
                    },
            'path': name,
            'metadata': metadata,
            }
    data = ts.open(labels_ts_spec, create=False, open=True).result()
    return data


@magic_factory(
        labels_file={'mode': 'w'},
        widget_init=_on_create_labels_init,
        )
def create_labels(
        source_image: napari.layers.Image,
        labels_file: pathlib.Path,
        chunks='',
        ) -> napari.types.LayerDataTuple:
    """Create/load a zarr array as a labels layer based on image layer.

    Parameters
    ----------
    source_image : Image layer
        The image that we are segmenting.
    labels_file : pathlib.Path
        The path to the zarr file to be created.
    chunks : str, optional
        A string that can be evaluated as a tuple of ints specifying the chunk
        size for the zarr file. If empty, they will be (128, 128) along the
        last dimensions and (1) along any remaining dimensions. This argument
        has no effect if the file already exists.
    """
    if chunks:
        chunks_str = chunks
        chunks = ast.literal_eval(chunks)
        if (type(chunks) is not tuple
                or not all(isinstance(val, int) for val in chunks)):
            raise ValueError(
                'chunks should be a tuple of ints, e.g. "(1, 1, 512, 512)", '
                f'got {chunks_str}'
                )
    else:  # use default
        chunks = (1,) * (source_image.ndim - 2) + (128, 128)

    layer_data = open_tensorstore(
            labels_file,
            shape=source_image.data.shape,
            chunks=chunks,
            )
    layer_type = 'labels'
    layer_metadata = {
            'scale': source_image.scale,
            'translate': source_image.translate,
            }
    create_ts_meta(labels_file, layer_metadata)
    return layer_data, layer_metadata, layer_type


class LabelCorrector:
    def __init__(self,
                 image_file,
                 labels_file,
                 time_index,
                 scale=(1,1,4),
                 c=2,
                 t=None):
        """
        Correct labels to create a ground truth with five opperations,
        each of which correspond to the following number key.
            (1) toggel selection of points - to seed watershed to split
                labels.
            (2) watershed - recompute watershed for joined labels based on
                chosen points.
            (3) toggle pick label colour - choose the colour of the label
                with which to paint.
            (4) toggle label fill mode - for merging labels
            (5) toggel paint mode - paint new label.

        Output can be saved to a new file by using Control-s. Note that this is
        not necessary if using tensorstore, as annotations will be written to
        file. However, this is still useful for generating separate copies of
        GT data (if completing annotations of whole frames). If time_index is
        None, the save operation will make a copy of the currently displayed
        frame (dim = t) and the file will have a suffix of the form
        "t<idx>_GT.zarr".

        If the labels_file parameter is given a dict, this will be assumed
        to be JSON-like spec for tensorstore.
        - ASSUMES IMAGES ARE (c, t, z, y, x)
        - ASSUMES LABELS ARE (t, z, y, x)
        - ASSUMES GT FILES ARE (z, y, x)

        Parameters
        ----------
        image_file: str
            Path to the image data (.zarr)
        labels_file: str or dict
            str: Path to the labels data (.zarr)
            dict: tensorstore spec for opening the file
                (also .zarr)
        time_index: int or None
                None: all frames
                int: one frame
                TODO: slice: selection of frames
                    (must reindex ts when slicing or napari has issues)
        scale: tuple of int
            scale of image and labels for napari
        c: int
            the channel to read from the image
        t: int or none
            position of the time index in the labels file
            (if this exists). If ndim > 3 and None specified
            the format will be assumed to be (c*, t, z, y, x)

        Attributes
        ----------
        tensorstore: bool
            Should the lables be accessed via tensorstore?
            This is set to True when the inputted labels_file
            is a dict (JSON-esque spec for ts.open)
        gt_file: bool
            Is this a file ending in the suffix _GT?
            If so it is assumed to be the saved output of another
            annotation session
        time_index: slice, int, or None
            Gives the point/s in time to be selected
        labels: ndarray or tensorstore.TensorStore
        image: dask array
        scale: tuple of int
        """

        # switches
        # --------
        # tensorstore -> open zarr with tensorstore
        #   affects how we need to open data
        # gt_file -> already using a ground truth labels
        #   affects how we need to save and access the file
        #   i.e., indexes are thought to apply only to the image
        #   the ground truth is assumed to be the indexed image
        self.tensorstore = isinstance(labels_file, dict)
        self.gt_file = None # is reassigned to a bool within _get_path_info

        # Read/Write Info
        # ---------------
        self.labels_file = labels_file
        self.time_index = time_index
        self._save_path = self._get_path_info()

        # Lazy Data
        # ---------
        if time_index is None:
            self.time_index = slice(None) # ensure that the following two lines
                                          # work
        # Note: we assume a 5D array saved in ome-zarr order: tczyx
        self.image = da.from_zarr(image_file)[self.time_index, c]
        self.labels = self._open_labels()

        # Vis Info
        # --------
        self.viewer = None
        self.scale = scale
        self.ndim = len(self.image.shape)
        if self.ndim > 3 and t == None:
            t = -4
        self.t = t


    # Init helpers
    # ------------
    def _get_path_info(self):
        labels_file = self.labels_file
        # is the file storing a previously annotated and saved ground truth?
        if isinstance(labels_file, str):
            labels_path = labels_file
        elif self.tensorstore:
            # get the path str from spec
            labels_path = os.path.join(labels_file['kvstore']['path'],
                                        labels_file['path'])
        else:
            m = f'labels_file parameter must be dict or list not {type(labels_file)}'
            raise ValueError(m)
        self.gt_file = labels_path.endswith('_GT.zarr')
        save_path = self._get_save_path(labels_path)
        return save_path


    # Init helpers
    # ------------
    def _get_save_path(self, labels_path):
        if self.gt_file:
            # if working on a GT file save to the same name
            save_path = labels_path
        else:
            # otherwise use the name of the file sans extension
            # the self.save_path property will use the index on t
            # to generate the rest of the name.
            data_path = Path(labels_path)
            save_path = os.path.join(data_path.parents[0],
                                     data_path.stem)
        return save_path


    def _open_labels(self):
        labels_file = self.labels_file
        if self.tensorstore:
            # labels file should be the spec dict for tensorstore
            if not self.gt_file:
                # we need to apply the slice and so need to construct
                # the correct tuple of int / slices
                labels = labels[self.time_index]
        else:
            labels = zarr.open(labels_file, mode='r+')
            if not self.gt_file:
                labels = labels[self.time_index]
        return labels


    # CALL
    # ----
    def __call__(self):
        with napari.gui_qt():
            self.viewer =  napari.Viewer()
            self.viewer .add_image(
                                   self.image,
                                   name='Image',
                                   scale=self.scale
                                   )
            self.viewer .add_labels(
                                    self.labels,
                                    name='Labels',
                                    scale=self.scale
                                    )
            self.viewer .add_points(
                                    np.empty(
                                             (0, len(self.labels.shape)),
                                             dtype=float
                                             ),
                                    scale=self.scale,
                                    size=2)
            self.viewer .bind_key('1', self._points)
            self.viewer .bind_key('2', self._watershed)
            self.viewer .bind_key('3', self._select_colour)
            self.viewer .bind_key('4', self._fill)
            self.viewer .bind_key('5', self._paint)
            self.viewer .bind_key('Shift-s', self._save)


    @property
    def save_path(self):
        if self.gt_file:
            # no need to find the save path suffix, it is ex
            return self._save_path
        else:
            if isinstance(self.time_index, int):
                time = self.time_index
            elif self.ndim > 3:
                time = self.viewer.dims.current_step[self.t]
            else:
                time = 'unknown'
            suffix = f"_t{time}_GT.zarr"
            return self._save_path + suffix


    # Call helpers
    # ------------
    def _points(self, viewer):
        """
        Switch to points layer to split a label
        """
        if viewer.layers['Points'].mode != 'add':
            viewer.layers['Points'].mode = 'add'
        else:
            viewer.layers['Points'].mode = 'pan_zoom'




    def _select_colour(self, viewer):
        """
        Select colour for painting
        """
        if viewer.layers['Labels'].mode != 'pick':
            viewer.layers['Labels'].mode = 'pick'
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'


    def _fill(self, viewer):
        """
        Switch napari labels layer to fill mode
        """
        if viewer.layers['Labels'].mode != 'fill':
            viewer.layers['Labels'].mode = 'fill'
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'


    def _paint(self, viewer):
        """
        Switch napari labels layer to paint mode
        """
        if viewer.layers['Labels'].mode != 'paint':
            viewer.layers['Labels'].mode = 'paint'
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'


    def _save(self, viewer):
        """
        Save the annotated time_index as a zarr file
        """
        array = viewer.layers['Labels'].data
        if len(array.shape) > 3: # save the current frame
            time = self.viewer.dims.current_step[self.t]
            idx = [slice(None)] * self.ndim
            idx[self.t] = time
            idx = tuple(idx)
            array = array[idx]
        zarr.save_array(self.save_path, np.array(array))
        print("Labels saved at:")
        print(self.save_path)


# Split Objects
# -------------