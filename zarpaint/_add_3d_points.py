import magicgui
import napari
import numpy as np


def get_ray_coordinates(shape, start_point, end_point):
    length = np.linalg.norm(end_point - start_point)
    length_int = np.round(length).astype(int)
    coords = np.linspace(start_point, end_point, length_int + 1, endpoint=True)
    clipped_coords = np.clip(
        np.round(coords), 0, np.asarray(shape) - 1
    ).astype(int)
    return clipped_coords


def get_data_ray(data, start_point, end_point):
    """Coordinate of the first nonzero element between start and end points.

    Parameters
    ----------
    data : nD array, shape (N1, N2, ..., ND)
        A data volume.
    start_point : array, shape (D,)
        The start coordinate to check.
    end_point : array, shape (D,)
        The end coordinate to check.

    Returns
    -------
    coords : array, shape (M, D), where M = norm(end_point - start_point)
        The integer coordinates of each point of the ray.
    ray : array, shape (M,), 
        The data at ``coords``.
    """
    clipped_coords = get_ray_coordinates(data.shape, start_point, end_point)
    ray = data[tuple(clipped_coords.T)]
    return clipped_coords, ray


def find_midpoint_of_first_segment(layer, event):
    """Return the world coordinate of a Labels layer mouse event in 2D or 3D.

    In 2D, this is just the event's position.

    In 3D, a ray is cast in data coordinates, and the midpoint coordinate of
    the first nonzero blob along that ray is returned, after being transformed
    back to world coordinates. If the ray only contains zeros, None is
    returned.

    Parameters
    ----------
    layer : napari.layers.Labels
        The Labels layer.
    event : vispy MouseEvent
        The mouse event, containing position and view direction attributes.

    Returns
    -------
    coordinates : array of int
        The world coordinates for the mouse event.
    """
    ndim = len(layer._dims_displayed)
    if ndim == 2:
        coordinates = event.position
    else:  # 3d
        start, end = layer.get_ray_intersections(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=layer._dims_displayed,
            world=True,
        )
        coordinates, ray = get_data_ray(layer.data, start, end)
        ray_padded = np.pad(
            ray, pad_width=1, mode='constant', constant_values=0
        )
        diff_coords = np.flatnonzero(np.diff(ray_padded))
        if len(diff_coords) > 1:
            blob_start_end = coordinates[diff_coords[:2]]
            blob_mid = np.mean(blob_start_end, axis=0)
            data2world = layer._transforms[1:3].simplified
            coordinates = data2world(blob_mid)
        else:
            coordinates = None
    return coordinates


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
        if world_click_coordinates is not None:
            pts_coordinates = pts_world2data(world_click_coordinates)
            points.add(pts_coordinates)
