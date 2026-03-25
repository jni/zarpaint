import os
from pathlib import Path
import tempfile
import napari
import numpy as np
from numpy.testing import assert_array_equal
from zarpaint import create_labels, DimsSorter
from zarpaint.reader import zarr_tensorstore


def test_create_labels():
    image = napari.layers.Image(
            np.random.random((10, 512, 512)),
            name='img',
            scale=[4, 1, 1],
            translate=[1000, -1000, -2000],
            )
    create_labels_ = create_labels()  # instance from magic_factory
    with tempfile.TemporaryDirectory() as tmpdir:
        pth = Path(tmpdir) / 'labels.zarr'
        arr, meta, layer_type = create_labels_(image, pth, str((1, 128, 128)))
        # note that arr.shape is an array if arr is tensorstore.Array
        assert tuple(arr.shape) == image.data.shape
        for key, val in meta.items():
            assert_array_equal(getattr(image, key), val)
        assert layer_type == 'labels'
        assert pth.exists()
        assert len(list(pth.iterdir())) == 2  # .zarray/zarr.json, .naparimeta.yml
        arr[:4, :64, :64] = 1  # touch 4 chunks along axis 0
        if (pth / '.zarray').exists():  # zarr v2
            assert len(list(pth.iterdir())) == 6
        else:  # zarr v3
            assert len(os.listdir(pth)) == 3
            assert len(os.listdir(os.path.join(pth, 'c'))) == 4


def test_dims_sorter(make_napari_viewer):
    viewer = make_napari_viewer(strict_qt=False)
    ndim = 5
    viewer.add_points(np.random.random((10, ndim)) * 512)
    sorter = DimsSorter(viewer)
    new_order = (1, 0, 2, 3, 4)
    viewer.dims.order = new_order
    assert [sorter.axes_list[i].axis for i in range(ndim)] == list(new_order)
    assert sorter.axes_list[0].dims is viewer.dims
    sorter.axes_list.move(4, 3)
    assert viewer.dims.order == (1, 0, 2, 4, 3)


def test_open_tensorstore():
    image = napari.layers.Image(
            np.random.random((10, 512, 512)),
            name='img',
            scale=[4, 1, 1],
            translate=[1000, -1000, -2000],
            )
    create_labels_ = create_labels()  # instance from magic_factory
    with tempfile.TemporaryDirectory() as tmpdir:
        pth = Path(tmpdir) / 'labels.zarr'
        _ = create_labels_(image, pth, str((1, 128, 128)))
        # note that arr.shape is an array if arr is tensorstore.Array
        reader = zarr_tensorstore(pth)
        arr, meta, layer_type = reader(pth)[0]  # 1st element of layer list
        assert tuple(arr.shape) == image.data.shape
        for key, val in meta.items():
            assert_array_equal(getattr(image, key), val)
        assert layer_type == 'labels'
        assert pth.exists()
        assert len(list(pth.iterdir())) == 2  # .zarray/zarr.json, .naparimeta.yml
        arr[:4, :64, :64] = 2  # touch 4 chunks along axis 0
        if (pth / '.zarray').exists():  # zarr v2
            assert len(list(pth.iterdir())) == 6
        else:  # zarr v3
            assert len(os.listdir(pth)) == 3
            assert len(os.listdir(os.path.join(pth, 'c'))) == 4
