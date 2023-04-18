from collections import defaultdict
from magicgui.widgets import Container, ComboBox, PushButton
import napari
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import distance_transform_edt


def signed_distance_transform(image):
    """apply distance transform and return new image

    Parameters
    ----------
    image : np.array
        the label layer data

    Returns
    -------
    np.array
        label layer data with distance transform applied
    """
    image = image.astype(bool)
    edt = distance_transform_edt(image) - distance_transform_edt(~image)
    return edt


def point_and_values(image_1, image_2, interp_axis=0):
    """apply distance transforms to the 2 images to interpolate.

    Apply distance transforms to the 2 images that are going to be 
    interpolated between. Return a tuple of points and values where
    points represent  

    Parameters
    ----------
    image_1 : np.array
        The first slice to interpolate from
    image_2 : _type_
        The second slice to interpolate to
    interp_axis : int, optional
        The dimension along which to interpolate, by default 0

    Returns
    -------
    tuple of numpy array
        tuple og distance transform data and corresponding location
    """
    edt_1 = signed_distance_transform(image_1)
    edt_2 = signed_distance_transform(image_2)
    values = np.stack([edt_1, edt_2], axis=interp_axis)
    points = tuple([np.arange(i) for i in values.shape])
    return points, values


def xi_coords(shape, percent=0.5, interp_axis=0):
    """ 
    create array of coordinates to interpolate between
    Parameters
    ----------
    shape : tuple
        Shape of the slice
    percent : float, optional
        Value to populate the xi array, by default 0.5
    interp_axis : int, optional
        The axis to interpolate along , by default 0

    Returns
    -------
    numpy array
        Coordinate denoting the area of the slice to interpolate along
    """
    slices = [slice(0, i) for i in shape]
    xi = np.moveaxis(np.mgrid[slices], 0,
                     -1).reshape(np.prod(shape), len(shape)).astype('float')
    xi = np.insert(xi, interp_axis, percent, axis=1)
    return xi


def slice_iterator(slice_index_1, slice_index_2):
    """create an iterable across the range of slices to be interpolated

    Parameters
    ----------
    slice_index_1 : int
        number of one bound of the slice range
    slice_index_2 : _type_
        the opposite bound of the slice range

    Returns
    -------
    zip of 2 numpy arrays
        tuple of slice indicies, tuple of percentages to give xi coords
    """
    intermediate_slices = np.arange(slice_index_1 + 1, slice_index_2)
    n_slices = slice_index_2 - slice_index_1 + 1
    stepsize = 1 / n_slices
    intermediate_percentages = np.arange(0 + stepsize, 1, stepsize)
    return zip(intermediate_slices, intermediate_percentages)


def interpolated_slice(
        percent, points, values, interp_axis=0, method='linear'
        ):
    """Create the dtata for one of the interpolated slices

    Parameters
    ----------
    percent : array_like
        A value to populate the xi array
    points : tuple of ndarray of float
        The points of the slice on which to paint
    values : array_like
        Data to draw on the slice
    interp_axis : int, optional
        The axis along which to interpolate, by default 0
    method : str, optional
        Interpolation method, by default 'linear'

    Returns
    -------
    np array 
        A slice with interpolated data drawn on
    """
    # TODO: check return type
    img_shape = list(values.shape)
    del img_shape[interp_axis]

    xi = xi_coords(img_shape, percent=percent, interp_axis=interp_axis)
    interpolated_img = interpn(points, values, xi, method=method)
    interpolated_img = np.reshape(interpolated_img, img_shape) > 0
    return interpolated_img


class InterpolateSliceWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Widget for handling the interpolate slice gui and event callbacks

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            napari viewer to add the widget to
        """
        super().__init__()
        self.viewer = viewer
        self.painted_slice_history = defaultdict(set)

        self.labels_combo = ComboBox(
                name='Labels Layer', choices=self.get_labels_layers
                )
        self.interp_dim_combo = ComboBox(
                name="Interpolation Dimension",
                choices=self.update_dim_choices
                )

        self.start_interpolation_btn = PushButton(name='Start Interpolation')
        self.interpolate_btn = PushButton(name='Interpolate')
        self.start_interpolation_btn.clicked.connect(
                self.enter_interpolation_mode
                )
        self.extend([
                self.labels_combo, self.interp_dim_combo,
                self.start_interpolation_btn, self.interpolate_btn
                ])
        self.interpolate_btn.hide()

        self.interpolate_axis = 0
        self.selected_layer = None

    def update_dim_choices(self, interp_dim_combo):
        layer_name = self.labels_combo.current_choice
        if not layer_name:
            return []
        return list(range(self.viewer.layers[layer_name].data.ndim))

    def get_labels_layers(self, combo):
        """Returns a list of existing labels to display

        Parameters
        ----------
        combo : magicgui ComboBox
            A dropdown to dispaly the layers

        Returns
        -------
        list[napari.layer.label]
            A list of curently existing layers
        """
        return [
                layer for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Labels)
                ]

    def store_painted_slices(self, event):
        """Identify slices that have been painetd on

        Parameters
        ----------
        event : Event
            napari paint event
        """
        last_label_history_item = event.value
        real_item = []
        # filter empty history atoms from item
        for atom in last_label_history_item:
            all_coords = list(atom[0])
            if any([len(arr) for arr in all_coords]):
                real_item.append(atom)
        if not real_item:
            return

        # item is list of atoms. atom is (tuple of e.g. (y, x) painted coords, array of original label, new label)
        last_label_history_item = real_item
        last_label_coords = last_label_history_item[0][0]

        unique_coords = list(map(np.unique, last_label_coords))

        last_slice_painted = unique_coords[self.interpolate_axis][0]

        label_id = last_label_history_item[-1][-1]

        self.painted_slice_history[label_id].add(last_slice_painted)

    def enter_interpolation_mode(self, event):
        """Connect the paint callback and change button text

        Parameters
        ----------
        event : Event
            Event spawned by button click
        """
        if not self.labels_combo.current_choice:
            raise RuntimeError("No labels layer selected.")

        self.selected_layer = self.viewer.layers[
                self.labels_combo.current_choice]

        self.selected_layer.events.paint.connect(self.store_painted_slices)

        self.start_interpolation_btn.hide()
        self.interpolate_btn.show()

        self.interpolate_btn.clicked.connect(self.interpolate)
        self.interpolate_axis = self.interp_dim_combo.get_value()
        self.labels_combo.enabled = False
        self.interp_dim_combo.enabled = False

    def interpolate(self, event):
        """For each label_id, iterate over each slice that has been painted on
         and perform pairwise (i, i+1) interpolation on each pair.

        Parameters
        ----------
        event : Event
            Object created upon clicking "interpolate" in the widget
        """

        for label_id, slices_painted in self.painted_slice_history.items():
            slices_painted = list(sorted(slices_painted))
            if len(slices_painted) > 1:
                for i in range(1, len(slices_painted)):
                    interpolate_between_slices(
                            self.selected_layer, slices_painted[i - 1],
                            slices_painted[i], label_id, self.interpolate_axis
                            )

        self.reset()

    def reset(self):
        """Reset button text and clear paint event history
        """
        self.selected_layer.events.paint.disconnect(self.store_painted_slices)
        self.painted_slice_history.clear()
        self.interpolate_axis = None
        self.interpolate_btn.clicked.disconnect(self.interpolate)

        self.interpolate_btn.hide()
        self.start_interpolation_btn.show()
        self.labels_combo.enabled = True
        self.interp_dim_combo.enabled = True


def interpolate_between_slices(
        label_layer: "napari.layers.Labels",
        slice_index_1: int,
        slice_index_2: int,
        label_id: int = 1,
        interpolate_axis: int = 0
        ):
    """Compute and draw interpolation between 2 label annotations.

    Parameters
    ----------
    label_layer : napari.layers.Labels
        The label layer to draw on 
    slice_index_1 : int
        slice containing the first label annotation
    slice_index_2 : int
        slice containing the second label anotation
        interpolation occurs between slice_index_1 slice_index_2
    label_id : int, optional
        the id of the annotation that is to be painted, by default 1
    interpolate_axis : int, optional
        the dimension/axis to interpolate across, by default 0
    """

    if slice_index_1 > slice_index_2:
        slice_index_1, slice_index_2 = slice_index_2, slice_index_1
    slice_1 = np.take(label_layer.data, slice_index_1, axis=interpolate_axis)
    slice_2 = np.take(label_layer.data, slice_index_2, axis=interpolate_axis)

    slice_1 = np.where(slice_1 == label_id, 1, 0)
    slice_2 = np.where(slice_2 == label_id, 1, 0)

    points, values = point_and_values(slice_1, slice_2, interpolate_axis)

    for slice_number, percentage in slice_iterator(slice_index_1,
                                                   slice_index_2):
        interpolated_img = interpolated_slice(
                percentage,
                points,
                values,
                interp_axis=interpolate_axis,
                method='linear'
                )
        indices = [slice(None) for _ in range(label_layer.data.ndim)]
        indices[interpolate_axis] = slice_number
        indices = tuple(indices)
        label_layer.data[indices][interpolated_img] = label_id
    label_layer.refresh()
