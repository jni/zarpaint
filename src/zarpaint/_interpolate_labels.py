from cProfile import label
from magicgui import magic_factory
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
    xi = np.moveaxis(np.mgrid[slices], 0, -1).reshape(np.prod(shape), len(shape)).astype('float')
    xi = np.insert(xi, interp_dim, percent, axis=1)
    return xi


def slice_iterator(slice_index_1, slice_index_2):
    intermediate_slices = np.arange(slice_index_1 + 1, slice_index_2)
    n_slices = slice_index_2 - slice_index_1 + 1  # inclusive
    stepsize = 1 / n_slices
    intermediate_percentages = np.arange(0 + stepsize, 1, stepsize)
    return zip(intermediate_slices, intermediate_percentages)


def interpolated_slice(percent, points, values, interp_dim=0, method='linear'):
    # Find the original image shape
    img_shape = list(values.shape)
    del img_shape[interp_dim]
    # Calculate the interpolated slice
    xi = xi_coords(img_shape, percent=percent, interp_dim=interp_dim)
    interpolated_img = interpn(points, values, xi, method=method)
    interpolated_img = np.reshape(interpolated_img, img_shape) > 0
    return interpolated_img
    

## Second draft, writes directly into tensorstore zarr array
@magic_factory(
        call_button='Interpolate'
        )
def interpolate_between_slices(label_layer: "napari.layers.Labels", slice_index_1: int, slice_index_2: int, label_id: int =1, interp_dim: int =0):
    if slice_index_1 > slice_index_2:
        slice_index_1, slice_index_2 = slice_index_2, slice_index_1
    layer_data = np.asarray(label_layer.data)
    slice_1 = np.take(layer_data, slice_index_1, axis=interp_dim)
    slice_2 = np.take(layer_data, slice_index_2, axis=interp_dim)
    # slice_1 = np.asarray(label_layer.data[slice_index_1])
    # slice_2 = np.asarray(label_layer.data[slice_index_2])

    #TODO: possible extension, handle all label ids separately     
    slice_1 = slice_1.astype(bool)
    slice_2 = slice_2.astype(bool)
    # interp_dim should just be the slider "dimension" right?
    points, values = point_and_values(slice_1, slice_2, interp_dim)
    #TODO: Thread this?   
    for slice_number, percentage in slice_iterator(slice_index_1, slice_index_2):
        interpolated_img = interpolated_slice(percentage, points, values, interp_dim=interp_dim, method='linear')
        indices = [slice(None) for _ in range(label_layer.data.ndim)]
        indices[interp_dim] = slice_number
        indices = tuple(indices)
        label_layer.data[indices][interpolated_img] = label_id
    label_layer.refresh()  # will update the current view

# interpolate_between_slices(label_layer, image_1, image_2, slice_index_1, slice_index_2)
# print("Done!")
# print("Please scroll through napari to see the interpolated label slices")
