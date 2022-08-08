from collections import defaultdict
import warnings
from magicgui.widgets import Container, ComboBox, PushButton
import napari
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import distance_transform_edt


def distance_transform(image):
    """Distance transform for a boolean image.
    
    Returns positive values inside the object,
    and negative values outside.
    """
    image = image.astype(bool)
    edt = distance_transform_edt(image) - distance_transform_edt(~image)
    return edt


def point_and_values(image_1, image_2, interp_dim=0):
    edt_1 = distance_transform(image_1)
    edt_2 = distance_transform(image_2)
    values = np.stack([edt_1, edt_2], axis=interp_dim)
    points = tuple([np.arange(i) for i in values.shape])
    return points, values


def xi_coords(shape, percent=0.5, interp_dim=0):
    slices = [slice(0, i) for i in shape]
    xi = np.moveaxis(np.mgrid[slices], 0,
                     -1).reshape(np.prod(shape), len(shape)).astype('float')
    xi = np.insert(xi, interp_dim, percent, axis=1)
    return xi


def slice_iterator(slice_index_1, slice_index_2):
    intermediate_slices = np.arange(slice_index_1 + 1, slice_index_2)
    n_slices = slice_index_2 - slice_index_1 + 1  # inclusive
    stepsize = 1 / n_slices
    intermediate_percentages = np.arange(0 + stepsize, 1, stepsize)
    return zip(intermediate_slices, intermediate_percentages)


def interpolated_slice(percent, points, values, interp_dim=0, method='linear'):
    img_shape = list(values.shape)
    del img_shape[interp_dim]

    xi = xi_coords(img_shape, percent=percent, interp_dim=interp_dim)
    interpolated_img = interpn(points, values, xi, method=method)
    interpolated_img = np.reshape(interpolated_img, img_shape) > 0
    return interpolated_img


class InterpolateSliceWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.labels_combo = ComboBox(
                name='Labels Layer', choices=self.get_labels_layers
                )
        self.start_interpolation = PushButton(name='Start Interpolation')
        self.interpolate_btn = PushButton(name='Interpolate')
        self.start_interpolation.clicked.connect(self.enter_interpolation)
        self.extend([
                self.labels_combo, self.start_interpolation,
                self.interpolate_btn
                ])
        self.interpolate_btn.hide()
        self.painted_slice_history = defaultdict(set)
        self.interp_dim = None

    def get_labels_layers(self, combo):
        return [
                layer for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Labels)
                ]

    def paint_callback(self, event):
        last_label_history_item = event.value
        real_item = []
        # filter empty history atoms from item
        for atom in last_label_history_item:
            all_coords = list(atom[0])
            if any([len(arr) for arr in all_coords]):
                real_item.append(atom)
        if not real_item:
            return

        last_label_history_item = real_item
        last_label_coords = last_label_history_item[0][
                0
                ]  # first history atom is a tuple, first element of atom is coords
        # here we can determine both the slice index that was painted *and* the interp dim
        # it wil be the array in last_label_coords that has only one unique element in it
        # the interp dim will be the index of that array in the tuple
        if not last_label_coords:
            return

        unique_coords = list(map(np.unique, last_label_coords))
        if self.interp_dim is None:
            self._infer_interp_dim(unique_coords)

        last_slice_painted = unique_coords[self.interp_dim][0]

        label_id = last_label_history_item[-1][-1]

        self.painted_slice_history[label_id].add(last_slice_painted)

    def _infer_interp_dim(self, unique_coords):
        interp_dim = None
        for i in range(len(unique_coords)):
            if len(unique_coords[i]) == 1:
                interp_dim = i
                break
        if interp_dim == None:
            warnings.warn(
                    "Couldn't determine axis for interpolation. Using axis 0 by default."
                    )
            self.interp_dim = 0
        else:
            self.interp_dim = interp_dim

    def enter_interpolation(self, event):
        # TODO: we wanna connect some callbacks that track for us the painted labels
        # grey out the combo box
        self.selected_layer = self.viewer.layers[
                self.labels_combo.current_choice]

        self.selected_layer.events.paint.connect(self.paint_callback)

        self.start_interpolation.hide()
        self.interpolate_btn.show()

        self.interpolate_btn.clicked.connect(self.interpolate)

    def interpolate(self, event):

        assert self.interp_dim is not None, 'Cannot interpolate without knowing dimension'

        for label_id, slices_painted in self.painted_slice_history.items():
            slices_painted = list(sorted(slices_painted))
            if len(slices_painted) > 1:
                for i in range(1, len(slices_painted)):
                    interpolate_between_slices(
                            self.selected_layer, slices_painted[i - 1],
                            slices_painted[i], label_id, self.interp_dim
                            )

        self.selected_layer.events.paint.disconnect(self.paint_callback)
        self.painted_slice_history.clear()
        self.interp_dim = None
        # TODO: multiple slices, multiple labels, stitching history items so we don't have to pass in the whole layer


def interpolate_between_slices(
        label_layer: "napari.layers.Labels",
        slice_index_1: int,
        slice_index_2: int,
        label_id: int = 1,
        interp_dim: int = 0
        ):

    if slice_index_1 > slice_index_2:
        slice_index_1, slice_index_2 = slice_index_2, slice_index_1
    layer_data = np.asarray(label_layer.data)
    slice_1 = np.take(layer_data, slice_index_1, axis=interp_dim)
    slice_2 = np.take(layer_data, slice_index_2, axis=interp_dim)

    slice_1 = np.where(slice_1 == label_id, 1, 0)
    slice_2 = np.where(slice_2 == label_id, 1, 0)

    points, values = point_and_values(slice_1, slice_2, interp_dim)

    for slice_number, percentage in slice_iterator(slice_index_1,
                                                   slice_index_2):
        interpolated_img = interpolated_slice(
                percentage,
                points,
                values,
                interp_dim=interp_dim,
                method='linear'
                )
        indices = [slice(None) for _ in range(label_layer.data.ndim)]
        indices[interp_dim] = slice_number
        indices = tuple(indices)
        label_layer.data[indices][
                interpolated_img
                ] = label_id  #use labels data_setitem (from the paint event PR)
    label_layer.refresh()  # will update the current view
