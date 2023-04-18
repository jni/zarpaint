import numpy as np
from zarpaint import _interpolate_labels
from napari.layers import Labels
from skimage.draw import ellipse, ellipsoid
from zarpaint._interpolate_labels import interpolate_between_slices
from unittest.mock import MagicMock
import pytest


def test_2d_slice_ellipse():
    label_id = 2
    arraysize = 512
    slice_index_1 = 0
    slice_index_2 = 10
    # First label slice
    coords_1 = ellipse(
            arraysize // 2,
            arraysize // 2,  # row, column (center coordinates)
            arraysize // 6,
            arraysize // 6,  # r_radius, c_radius (ellipse radii)
            shape=(arraysize, arraysize)
            )
    label_slice_1 = np.zeros((arraysize, arraysize)).astype(int)
    label_slice_1[coords_1] = label_id
    # Second label slice
    coords_2 = ellipse(
            arraysize // 2,
            arraysize // 2,  # row, column (center coordinates)
            arraysize // 5,
            arraysize // 3,  # r_radius, c_radius (ellipse radii)
            shape=(arraysize, arraysize)
            )
    label_slice_2 = np.zeros((arraysize, arraysize)).astype(int)
    label_slice_2[coords_2] = label_id
    # Create labels for interpolation
    labels = np.zeros((11, arraysize, arraysize)).astype(int)
    labels[slice_index_1] = label_slice_1
    labels[slice_index_2] = label_slice_2
    labels_layer = Labels(labels)

    # Check all intermediate label slices contain only zero
    for labels_slice in labels_layer.data[1:-1]:
        assert np.max(labels_slice) == 0
    # Check the expected number of non-zero pixels exist now
    assert np.count_nonzero(labels_layer.data[1:-1]) == 0

    # Interpolate intermediate slices
    interpolate_between_slices(
            labels_layer,
            slice_index_1,
            slice_index_2,
            label_id,
            interpolate_axis=0
            )

    # Check all intermediate label slices now contain non-zero pixels
    for labels_slice in labels_layer.data[1:-1]:
        assert np.max(labels_slice) == label_id
    # Check the expected number of non-zero pixels exist now
    assert np.count_nonzero(labels_layer.data[1:-1]) == 315045


def test_2d_slice_square():
    space = (100, 100, 100)

    data = np.zeros(shape=space, dtype="uint8")

    data[10, 10:20, 10:20] = 2
    data[20, 10:20, 10:20] = 2

    labels = Labels(data)

    _interpolate_labels.interpolate_between_slices(labels, 10, 20, 2, 0)

    np.testing.assert_allclose(labels.data[10:20, 10:20, 10:20], 2)


def test_interpolate_reversed_slices():
    space = (100, 100, 100)

    data = np.zeros(shape=space, dtype="uint8")

    data[10, 10:20, 10:20] = 2
    data[20, 10:20, 10:20] = 2

    labels = Labels(data)

    _interpolate_labels.interpolate_between_slices(labels, 20, 10, 2, 0)

    np.testing.assert_allclose(labels.data[10:20, 10:20, 10:20], 2)


def test_3d_slice_ellipsoid():
    label_id = 2
    arraysize = 100
    slice_index_1 = 0
    slice_index_2 = 4
    # First label slice
    ellipse_1 = ellipsoid(20, 35, 25) * label_id
    padding = np.array((arraysize, arraysize, arraysize)
                       ) - np.array(ellipse_1.shape)
    pad_width = [(i // 2, i//2 + 1) for i in padding]
    label_slice_1 = np.pad(ellipse_1, pad_width)
    # Second label slice
    ellipse_2 = ellipsoid(28, 33, 40) * label_id
    padding = np.array((arraysize, arraysize, arraysize)
                       ) - np.array(ellipse_2.shape)
    pad_width = [(i // 2, i//2 + 1) for i in padding]
    label_slice_2 = np.pad(ellipse_2, pad_width)
    # Create labels for interpolation
    labels = np.zeros((5, arraysize, arraysize, arraysize)).astype(int)
    labels[slice_index_1] = label_slice_1
    labels[slice_index_2] = label_slice_2
    labels_layer = Labels(labels)

    # Check all intermediate label slices contain only zero
    for labels_slice in labels_layer.data[1:-1]:
        assert np.max(labels_slice) == 0
    # Check the expected number of non-zero pixels exist now
    assert np.count_nonzero(labels_layer.data[1:-1]) == 0

    # Interpolate intermediate slices
    interpolate_between_slices(
            labels_layer,
            slice_index_1,
            slice_index_2,
            label_id,
            interpolate_axis=0
            )

    # Check all intermediate label slices now contain non-zero pixels
    for labels_slice in labels_layer.data[1:-1]:
        assert np.max(labels_slice) == label_id
    # Check the expected number of non-zero pixels exist now
    assert np.count_nonzero(labels_layer.data[1:-1]) == 297885


def test_3d_slice_cube():
    space = (100, 100, 100, 100)
    data = np.zeros(shape=space, dtype="uint8")

    data[20, 10:20, 10:20, 10:20] = 2
    data[10, 10:20, 10:20, 10:20] = 2

    labels = Labels(data)

    _interpolate_labels.interpolate_between_slices(labels, 10, 20, 2, 0)

    np.testing.assert_allclose(labels.data[10:20, 10:20, 10:20, 10:20], 2)


def test_labels_layer_combo_box(make_napari_viewer):
    viewer = make_napari_viewer()
    space = (100, 100, 100, 100)
    data = np.zeros(shape=space, dtype="uint8")

    data[20, 10:20, 10:20, 10:20] = 2
    data[10, 10:20, 10:20, 10:20] = 2

    viewer.add_labels(data, name="test data")
    viewer.add_image(data)
    interp_widget = _interpolate_labels.InterpolateSliceWidget(viewer)
    labels_layers_list = interp_widget.get_labels_layers(
            interp_widget.labels_combo
            )

    assert len(labels_layers_list) == 1
    assert labels_layers_list[0].name == "test data"

    viewer.layers.remove(labels_layers_list[0])
    labels_layers_list = interp_widget.get_labels_layers(
            interp_widget.labels_combo
            )
    assert len(labels_layers_list) == 0


def test_store_painted_slices(make_napari_viewer):
    viewer = make_napari_viewer()

    interp_widget = _interpolate_labels.InterpolateSliceWidget(viewer)
    event = MagicMock()

    interp_widget.store_painted_slices(event)
    assert len(interp_widget.painted_slice_history) == 0

    event.value = [(([50, 50, 50, 50, 50, 50, 50, 50,
                      50], [19, 20, 21, 19, 20, 21, 19, 20,
                            21], [9, 9, 9, 10, 10, 10, 11, 11,
                                  11]), [0, 0, 0, 0, 0, 0, 0, 0, 0], 2)]
    event_2 = MagicMock()
    event_2.value = [(([45, 45, 45, 45, 45, 45, 45, 45,
                        45], [19, 20, 21, 19, 20, 21, 19, 20,
                              21], [9, 9, 9, 10, 10, 10, 11, 11,
                                    11]), [0, 0, 0, 0, 0, 0, 0, 0, 0], 2)]
    interp_widget.store_painted_slices(event)
    interp_widget.store_painted_slices(event_2)
    assert interp_widget.painted_slice_history[2] == {50, 45}

    event_3 = MagicMock()
    event_3.value = [(([30, 30, 30, 30, 30, 30, 30, 30,
                        30], [19, 20, 21, 19, 20, 21, 19, 20,
                              21], [9, 9, 9, 10, 10, 10, 11, 11,
                                    11]), [0, 0, 0, 0, 0, 0, 0, 0, 0], 5)]
    interp_widget.store_painted_slices(event_3)
    assert interp_widget.painted_slice_history[5] == {30}


def test_distance_transform():
    shape = (3, 3)
    image = np.zeros(shape=shape, dtype="uint8")
    image[1, 1] = 3
    res = _interpolate_labels.signed_distance_transform(image)

    assert res[1, 1] == 1
    np.testing.assert_allclose(res[0, 0], -2**0.5)


def test_interpolate(make_napari_viewer):
    viewer = make_napari_viewer()

    space = (10, 10, 10)

    data = np.zeros(shape=space, dtype="uint8")

    paint_history = [[([2, 2, 2, 2, 2, 2, 2, 2,
                        2], [3, 4, 5, 3, 4, 5, 3, 4,
                             5], [3, 3, 3, 4, 4, 4, 5, 5, 5]),
                      [2, 2, 2, 2, 2, 2, 2, 2, 2], 2],
                     [([6, 6, 6, 6, 6, 6, 6, 6, 6], [
                             3, 4, 5, 3, 4, 5, 3, 4, 5
                             ], [3, 3, 3, 4, 4, 4, 5, 5, 5]),
                      [2, 2, 2, 2, 2, 2, 2, 2, 2], 2]]

    data[3, 3:6, 3:6] = 2
    data[6, 3:6, 3:6] = 2

    labels = Labels(data)
    viewer.add_layer(labels)
    interp_widget = _interpolate_labels.InterpolateSliceWidget(viewer)
    event = MagicMock()
    event.value = paint_history

    interp_widget.painted_slice_history[2] = [6, 3]
    interp_widget.interpolate_axis = 0
    interp_widget.selected_layer = viewer.layers[0]
    interp_widget.interpolate(event)

    np.testing.assert_allclose(
            viewer.layers[0].data[3], viewer.layers[0].data[4]
            )


def test_enter_interpolation_mode(make_napari_viewer):
    viewer = make_napari_viewer()
    labels = Labels(np.zeros(shape=(10, 10), dtype=np.uint8), name="Layer 1")
    labels_2 = Labels(np.zeros(shape=(10, 10), dtype=np.uint8), name="Layer 2")

    viewer.add_layer(labels)
    viewer.add_layer(labels_2)
    interp_widget = _interpolate_labels.InterpolateSliceWidget(viewer)
    interp_widget.labels_combo.value = labels_2
    interp_widget.enter_interpolation_mode(None)
    assert interp_widget.selected_layer == labels_2
    assert interp_widget.labels_combo.enabled == False
    assert interp_widget.interp_dim_combo.enabled == False


def test_enter_interpolation_mode_no_labels(make_napari_viewer):
    viewer = make_napari_viewer()
    interp_widget = _interpolate_labels.InterpolateSliceWidget(viewer)
    with pytest.raises(RuntimeError):
        interp_widget.enter_interpolation_mode(None)
