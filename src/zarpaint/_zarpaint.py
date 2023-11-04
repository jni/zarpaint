import ast
import os
import warnings
import yaml

from magicgui import magic_factory
import napari
import numpy as np
import pathlib
try:
    import tensorstore as ts
    tensorstore_available = True
except ModuleNotFoundError:
    tensorstore_available = False
    have_warned = False
import zarr
import toolz as tz


@tz.curry
def _set_default_labels_path(widget, source_image):
    """Helper function to set the default output path next to source image.

    When the widget to create a labels layer is instantiated, it asks for
    a filename. This function sets the default to be next to the original
    file.
    """
    if source_image.source.path is not None:
        source_path = pathlib.Path(source_image.source.path)
        labels_path = source_path.with_suffix('.labels.zarr')
        widget.labels_file.value = labels_path


def _on_create_labels_init(widget):
    """Ensure changes to the source image change the default labels path."""
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


def open_zarr(labels_file: pathlib.Path, *, shape=None, chunks=None):
    """Open a zarr file, with tensorstore if available, with zarr otherwise.

    If the file doesn't exist, it is created.

    Parameters
    ----------
    labels_file : Path
        The output file name.
    shape : tuple of int
        The shape of the array.
    chunks : tuple of int
        The chunk size of the array.

    Returns
    -------
    data : ts.Array or zarr.Array
        The array loaded from file.
    """
    if not os.path.exists(labels_file):
        zarr.open(
                str(labels_file),
                mode='w',
                shape=shape,
                dtype=np.uint32,
                chunks=chunks,
                )
    # read some of the metadata for tensorstore driver from file
    labels_temp = zarr.open(str(labels_file), mode='a')
    metadata = {
            'dtype': labels_temp.dtype.str,
            'order': labels_temp.order,
            'shape': labels_temp.shape,
            }

    dir, name = os.path.split(labels_file)
    labels_ts_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': dir},
            'path': name,
            'metadata': metadata,
            }
    if tensorstore_available:
        data = ts.open(labels_ts_spec, create=False, open=True).result()
    else:
        global have_warned
        if not have_warned:
            warnings.warn(
                    'tensorstore not available, falling back to zarr.\n'
                    'Drawing with tensorstore is *much faster*. We recommend '
                    'you install tensorstore with '
                    '`python -m pip install tensorstore`.'
                    )
            have_warned = True
        data = labels_temp
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

    layer_data = open_zarr(
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