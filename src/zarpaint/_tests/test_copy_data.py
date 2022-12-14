from napari.layers import Labels
from zarpaint import copy_data
import numpy as np
import tensorstore as ts
from zarpaint import open_tensorstore
import zarr


def test_copy_data(make_napari_viewer):
    viewer = make_napari_viewer()
    labels_layer1 = viewer.add_labels(
            np.random.randint(0, 2**23, size=(10, 20, 30))
            )
    labels_layer2 = viewer.add_labels(np.zeros((2, 10, 20, 30), dtype=int))
    viewer.dims.set_point(axis=0, value=1)
    widget = copy_data()
    widget(viewer, labels_layer1, labels_layer2)
    np.testing.assert_array_equal(labels_layer2.data[0], 0)
    np.testing.assert_array_equal(labels_layer2.data[1], labels_layer1.data)


def test_copy_data_tensorstore(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    labels_layer1 = viewer.add_labels(
            np.random.randint(0, 2**23, size=(10, 20, 30))
            )
    array2 = open_tensorstore(
            tmp_path / "example.zarr",
            shape=(2, 10, 20, 30),
            chunks=(1, 1, 20, 30)
            )
    labels_layer2 = viewer.add_labels(array2)
    viewer.dims.set_point(axis=0, value=1)
    widget = copy_data()
    widget(viewer, labels_layer1, labels_layer2)
    np.testing.assert_array_equal(labels_layer2.data[0], 0)
    np.testing.assert_array_equal(labels_layer2.data[1], labels_layer1.data)


def test_copy_data_zarr(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    labels_layer1 = viewer.add_labels(
            np.random.randint(0, 2**23, size=(10, 20, 30))
            )
    array2 = zarr.open(
            str(tmp_path / "example.zarr"),
            mode='w',
            shape=(2, 10, 20, 30),
            dtype=np.uint32,
            chunks=(1, 1, 20, 30),
            )
    labels_layer2 = viewer.add_labels(array2)
    viewer.dims.set_point(axis=0, value=1)
    widget = copy_data()
    widget(viewer, labels_layer1, labels_layer2)
    np.testing.assert_array_equal(labels_layer2.data[0], 0)
    np.testing.assert_array_equal(labels_layer2.data[1], labels_layer1.data)
