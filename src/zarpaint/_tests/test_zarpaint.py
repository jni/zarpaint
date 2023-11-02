from zarpaint._add_3d_points import get_ray_coordinates, get_data_ray, find_midpoint_of_first_segment, add_points_3d_with_alt_click
import numpy as np
from napari.layers import Labels

from vispy.util.keys import ALT


class MockMouseEvent():
    def __init__(self, position, view_direction) -> None:
        self.position = position
        self.view_direction = view_direction


def test_get_data_ray():
    shape = (3, 3, 3)
    data = np.linspace(1, 27, 27).reshape(shape)

    start = np.asarray((0, 0, 0))
    end = np.asarray((2, 2, 2))
    ray = get_data_ray(data, start, end)[1]

    expected = np.asarray([1., 14., 14., 27.])
    np.testing.assert_allclose(ray, expected)


def test_get_ray_coordinates():
    shape = (3, 3, 3)
    start = np.asarray((0, 0, 0))
    end = np.asarray((2, 2, 2))
    test_coords = np.asarray([[0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2]])
    coords = get_ray_coordinates(shape, start, end)
    np.testing.assert_allclose(coords, test_coords)

    end = np.asarray((0, 2, 0))
    test_coords = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]])
    coords = get_ray_coordinates(shape, start, end)
    np.testing.assert_allclose(coords, test_coords)

    # Testing when end point is "out of bounds"
    end = np.asarray((4, 4, 4))
    test_coords = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2],
                            [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    coords = get_ray_coordinates(shape, start, end)
    np.testing.assert_allclose(coords, test_coords)


def test_midpoint_2d_empty_ray(make_napari_viewer):
    viewer = make_napari_viewer()
    dims = viewer.dims

    mock_data = np.zeros(shape=(5, 5), dtype="uint8")
    layer_data = Labels(mock_data)
    viewer.add_layer(layer_data)

    position = (0, 0)
    view_direction = [1, 0]
    mouse_event = MockMouseEvent(position, view_direction)

    result = find_midpoint_of_first_segment(layer_data, dims, mouse_event)
    assert result == (0, 0)


def test_midpoint_2d_nonempty_ray(make_napari_viewer):
    viewer = make_napari_viewer()
    dims = viewer.dims

    mock_data = np.ones(shape=(5, 5), dtype="uint8")
    layer_data = Labels(mock_data)
    viewer.add_layer(layer_data)

    position = (3, 0)
    view_direction = [1, 0]
    mouse_event = MockMouseEvent(position, view_direction)

    result = find_midpoint_of_first_segment(layer_data, dims, mouse_event)
    assert result == (3, 0)


def test_midpoint_3d_empty_ray(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    dims = viewer.dims

    mock_data = np.zeros(shape=(5, 5, 5), dtype="uint8")
    layer_data = Labels(mock_data)
    mouse_event = MockMouseEvent((2, 2, 0), [1, 0, 0])

    viewer.add_layer(layer_data)

    result = find_midpoint_of_first_segment(layer_data, dims, mouse_event)
    assert result is None

    mock_data[1:4, 1:4, 1:4] = 1
    layer_data = Labels(mock_data)
    mouse_event = MockMouseEvent((2, 2, 0), [1, 0, 0])

    viewer.add_layer(layer_data)

    result = find_midpoint_of_first_segment(layer_data, dims, mouse_event)
    assert result is None


def test_midpoint_3d_nonempty_ray(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    dims = viewer.dims

    mock_data = np.zeros(shape=(5, 5, 5), dtype="uint8")
    mock_data[1:4, 1:4, 1:4] = 1
    layer_data = Labels(mock_data)
    viewer.add_layer(layer_data)

    mouse_event = MockMouseEvent((2, 2, 0), [0, 1, 1])
    result = find_midpoint_of_first_segment(layer_data, dims, mouse_event)
    np.testing.assert_allclose(result, [2., 3.5, 1.5])


def test_add_point_3d_alt_click(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3

    mock_data = np.zeros(shape=(5, 5, 5), dtype="uint8")
    mock_data[1:4, 1:4, 1:4] = 1
    label_layer = Labels(mock_data)
    viewer.add_layer(label_layer)

    points_layer = viewer.add_points([], ndim=3)
    viewer.layers.selection.active = label_layer

    point_widget = add_points_3d_with_alt_click()
    point_widget(viewer, label_layer, points_layer)

    view = viewer.window._qt_viewer
    click_coordinates = (view.canvas.size[0] / 2, view.canvas.size[1] / 2, 0)
    view.canvas.events.mouse_press(
            pos=click_coordinates, modifiers=(ALT,), button=0
            )

    assert len(points_layer.data)
