import numpy as np
from zarpaint import _interpolate_labels
from napari.layers import Labels
from skimage.draw import ellipse, ellipsoid
from zarpaint._interpolate_labels import interpolate_between_slices


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
            labels_layer, slice_index_1, slice_index_2, label_id, interp_dim=0
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
            labels_layer, slice_index_1, slice_index_2, label_id, interp_dim=0
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
