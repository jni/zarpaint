import numpy as np
from scipy import ndimage as ndi
from magicgui import magic_factory
from skimage.morphology import octahedron
from skimage.segmentation import watershed


@magic_factory(
    call_button='Split',
    viewer={'visible': False},
    ndim={'min': 2, 'max': 3},
)
def watershed_split(
        viewer: 'napari.viewer.Viewer',
        labels: 'napari.layers.Labels',
        points: 'napari.layers.Points',
        ndim: int = 3,
        ):
    """Execute watershed to split labels based on provided points.

    Only labels containing points will be modified. The operation will be
    performed in-place on the input Labels layer.

    Parameters
    ----------
    viewer : napari.Viewer
        The current viewer displaying the data.
    labels : napari Labels layer
        The layer containing the segmentation. This will be used both for
        input and output.
    points : napari Points layer
        The points marking the labels to be split.
    ndim : int in {2, 3}
        The number of dimensions for the watershed operation.
    """
    coord = viewer.dims.current_step
    slice_idx = coord[:-ndim]
    # find the labels corresponding to the current points in the points layer
    labels_sliced = np.asarray(labels.data[slice_idx])
    points_slicer = np.ones(points.data.shape[0], dtype=bool)
    for dim, idx in enumerate(slice_idx):
        points_slicer &= points.data[:, dim] == idx
    points_sliced = np.round(
            points.data[points_slicer][:, -ndim:]
            ).astype(int)
    image = np.ones_like(labels_sliced, dtype=np.uint8)
    labels_split = _watershed_split(
            image,
            labels_sliced,
            points_sliced,
            compactness=200,
            connectivity_octahedron=7,
            )
    labels.data[slice_idx] = labels_split
    points.data = np.empty((0, viewer.dims.ndim), dtype=float)
    labels.refresh()


def _watershed_split(
        image,
        labels,
        points,
        compactness=200,
        connectivity_octahedron=7
        ):
    """
    Split labels with using points as markers for watershed
    """
    connectivity = octahedron(connectivity_octahedron)
    points = np.round(points).astype(int)
    coords = tuple([points[:, i] for i in range(points.shape[1])])
    p_lab = labels[coords]
    p_lab = np.unique(p_lab)
    p_lab = p_lab[p_lab != 0]
    # generate a mask corresponding to the labels that need to be split
    mask = np.zeros(labels.shape, dtype=bool)
    for lab in p_lab:
        where = labels == lab
        mask = mask + where
    # split the labels using the points (in the masked image)
    markers_bool = np.zeros(labels.shape, dtype=bool)
    markers_bool[coords] = True
    markers, _ = ndi.label(markers_bool, output=labels.dtype)
    new_labels = watershed(
            image,
            markers=markers,
            mask=mask,
            compactness=compactness,
            connectivity=connectivity,
            )
    labels[mask] = new_labels[mask] + labels.max()
    return labels
