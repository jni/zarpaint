import os
import tempfile
import napari
import numpy as np
from numpy.testing import assert_array_equal
from zarpaint import create_labels


def test_create_labels():
    image = napari.layers.Image(
        np.random.random((10, 512, 512)),
        name='img',
        scale=[4, 1, 1],
        translate=[1000, -1000, -2000],
        )
    create_labels_ = create_labels()  # instance from magic_factory
    with tempfile.TemporaryDirectory() as tmpdir:
        pth = os.path.join(tmpdir, 'labels.zarr')
        arr, meta, layer_type = create_labels_(image, pth, str((1, 128, 128)))
        # note that arr.shape is an array if arr is tensorstore.Array
        assert tuple(arr.shape) == image.data.shape
        for key, val in meta.items():
            assert_array_equal(getattr(image, key), val)
        assert layer_type == 'labels'
        assert os.path.exists(pth)
        assert len(os.listdir(pth)) == 1  # .zarray
        arr[4, :256, :256] = 1  # touch 4 chunks
        assert len(os.listdir(pth)) == 5
