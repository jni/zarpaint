import os
import napari
import dask.array as da
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import octahedron
from skimage.segmentation import watershed
import tensorstore as ts
from tensorstore import TensorStore
import zarr


# Monkey patch for ts.copy() - napari uses this but this should be resolved
# soon.
# Prevents errors when painting with napari - undo history uses data.copy()
TensorStore.copy = TensorStore.__array__


def correct_labels(image_file, labels_file, time_index, scale=(1,1,4), c=2):
    """
    Correct labels to create a ground truth with five opperations,
    each of which correspond to the following number key.
        (1) toggel selection of points - to seed watershed to split
            labels.
        (2) watershed - recompute watershed for joined labels based on
            chosen points.
        (3) toggle pick label colour - choose the colour of the label
            with which to paint.
        (4) toggle label fill mode - for merging labels
        (5) toggel paint mode - paint new label.

    Output can be saved to a new file by using Control-s.

    NOTE: 4D annotation: if the zarr is chunked across time (for reading
    frames: e.g., 1, None, None, None) then rolling the dims might
    cause big issues, as there is no way to prevent the t dim from being rolled
    as you progress through the permutations.
    I.e., more data than RAM -> blow up napari -> swear @ computer
        -> kill script (easier to restart to look at another view)

    Parameters
    ----------
    image_file: str
        Path to the image data (.zarr)
    labels_file: str or dict
        str: Path to the labels data (.zarr)
        dict: tensorstore spec for opening the file
            (also .zarr)
    time_index: int or None
            None: all frames
            int: one frame
            TODO: slice: selection of frames
                (must reindex ts when slicing or napari has issues)
    scale: tuple of int
        scale of image and labels for napari
    c: int
        the channel to read from the image
    t: int or none
        position of the time index in the labels file
        (if this exists). If ndim > 3 and None specified
        the format will be assumed to be (c*, t, z, y, x)
    """
    lets_annotate = LabelCorrector(
                                   image_file,
                                   labels_file,
                                   time_index,
                                   scale=scale,
                                   c=c
                                   )
    lets_annotate()



class LabelCorrector:
    def __init__(self,
                 image_file,
                 labels_file,
                 time_index,
                 scale=(1,1,4),
                 c=2,
                 t=None):
        """
        Correct labels to create a ground truth with five opperations,
        each of which correspond to the following number key.
            (1) toggel selection of points - to seed watershed to split
                labels.
            (2) watershed - recompute watershed for joined labels based on
                chosen points.
            (3) toggle pick label colour - choose the colour of the label
                with which to paint.
            (4) toggle label fill mode - for merging labels
            (5) toggel paint mode - paint new label.

        Output can be saved to a new file by using Control-s. Note that this is
        not necessary if using tensorstore, as annotations will be written to
        file. However, this is still useful for generating separate copies of
        GT data (if completing annotations of whole frames). If time_index is
        None, the save operation will make a copy of the currently displayed
        frame (dim = t) and the file will have a suffix of the form
        "t<idx>_GT.zarr".

        If the labels_file parameter is given a dict, this will be assumed
        to be JSON-like spec for tensorstore.
        - ASSUMES IMAGES ARE (c, t, z, y, x)
        - ASSUMES LABELS ARE (t, z, y, x)
        - ASSUMES GT FILES ARE (z, y, x)

        Parameters
        ----------
        image_file: str
            Path to the image data (.zarr)
        labels_file: str or dict
            str: Path to the labels data (.zarr)
            dict: tensorstore spec for opening the file
                (also .zarr)
        time_index: int or None
                None: all frames
                int: one frame
                TODO: slice: selection of frames
                    (must reindex ts when slicing or napari has issues)
        scale: tuple of int
            scale of image and labels for napari
        c: int
            the channel to read from the image
        t: int or none
            position of the time index in the labels file
            (if this exists). If ndim > 3 and None specified
            the format will be assumed to be (c*, t, z, y, x)

        Attributes
        ----------
        tensorstore: bool
            Should the lables be accessed via tensorstore?
            This is set to True when the inputted labels_file
            is a dict (JSON-esque spec for ts.open)
        gt_file: bool
            Is this a file ending in the suffix _GT?
            If so it is assumed to be the saved output of another
            annotation session
        time_index: slice, int, or None
            Gives the point/s in time to be selected
        labels: ndarray or tensorstore.TensorStore
        image: dask array
        scale: tuple of int
        """

        # switches
        # --------
        # tensorstore -> open zarr with tensorstore
        #   affects how we need to open data
        # gt_file -> already using a ground truth labels
        #   affects how we need to save and access the file
        #   i.e., indexes are thought to apply only to the image
        #   the ground truth is assumed to be the indexed image
        self.tensorstore = isinstance(labels_file, dict)
        self.gt_file = None # is reassigned to a bool within _get_path_info

        # Read/Write Info
        # ---------------
        self.labels_file = labels_file
        self.time_index = time_index
        self._save_path = self._get_path_info()

        # Lazy Data
        # ---------
        if time_index is None:
            self.time_index = slice(None) # ensure that the following two lines
                                          # work
        # Note: we assume a 5D array saved in ome-zarr order: tczyx
        self.image = da.from_zarr(image_file)[self.time_index, c]
        self.labels = self._open_labels()

        # Vis Info
        # --------
        self.viewer = None
        self.scale = scale
        self.ndim = len(self.image.shape)
        if self.ndim > 3 and t == None:
            t = -4
        self.t = t


    # Init helpers
    # ------------
    def _get_path_info(self):
        labels_file = self.labels_file
        # is the file storing a previously annotated and saved ground truth?
        if isinstance(labels_file, str):
            labels_path = labels_file
        elif self.tensorstore:
            # get the path str from spec
            labels_path = os.path.join(labels_file['kvstore']['path'],
                                        labels_file['path'])
        else:
            m = f'labels_file parameter must be dict or list not {type(labels_file)}'
            raise ValueError(m)
        self.gt_file = labels_path.endswith('_GT.zarr')
        save_path = self._get_save_path(labels_path)
        return save_path


    # Init helpers
    # ------------
    def _get_save_path(self, labels_path):
        if self.gt_file:
            # if working on a GT file save to the same name
            save_path = labels_path
        else:
            # otherwise use the name of the file sans extension
            # the self.save_path property will use the index on t
            # to generate the rest of the name.
            data_path = Path(labels_path)
            save_path = os.path.join(data_path.parents[0],
                                     data_path.stem)
        return save_path


    def _open_labels(self):
        labels_file = self.labels_file
        if self.tensorstore:
            # labels file should be the spec dict for tensorstore
            labels = ts.open(labels_file, create=False, open=True).result()
            if not self.gt_file:
                # we need to apply the slice and so need to construct
                # the correct tuple of int / slices
                labels = labels[self.time_index]
        else:
            labels = zarr.open(labels_file, mode='r+')
            if not self.gt_file:
                labels = labels[self.time_index]
        return labels


    # CALL
    # ----
    def __call__(self):
        with napari.gui_qt():
            self.viewer =  napari.Viewer()
            self.viewer .add_image(
                                   self.image,
                                   name='Image',
                                   scale=self.scale
                                   )
            self.viewer .add_labels(
                                    self.labels,
                                    name='Labels',
                                    scale=self.scale
                                    )
            self.viewer .add_points(
                                    np.empty(
                                             (0, len(self.labels.shape)),
                                             dtype=float
                                             ),
                                    scale=self.scale,
                                    size=2)
            self.viewer .bind_key('1', self._points)
            self.viewer .bind_key('2', self._watershed)
            self.viewer .bind_key('3', self._select_colour)
            self.viewer .bind_key('4', self._fill)
            self.viewer .bind_key('5', self._paint)
            self.viewer .bind_key('Shift-s', self._save)


    @property
    def save_path(self):
        if self.gt_file:
            # no need to find the save path suffix, it is ex
            return self._save_path
        else:
            if isinstance(self.time_index, int):
                time = self.time_index
            elif self.ndim > 3:
                time = self.viewer.dims.current_step[self.t]
            else:
                time = 'unknown'
            suffix = f"_t{time}_GT.zarr"
            return self._save_path + suffix


    # Call helpers
    # ------------
    def _points(self, viewer):
        """
        Switch to points layer to split a label
        """
        if viewer.layers['Points'].mode != 'add':
            viewer.layers['Points'].mode = 'add'
        else:
            viewer.layers['Points'].mode = 'pan_zoom'


    def _watershed(self, viewer):
        """
        Execute watershed to split labels based on provided points.
        Uses single time frame as data.
        """
        # find the labels corresponding to the current points in the points layer
        labels = viewer.layers['Labels'].data
        image = viewer.layers['Image'].data
        points = viewer.layers['Points'].data
        if self.ndim > 3: # dont read in more than one 3d frame at a time
            idx = viewer.dims.current_step[self.t]
            points = points[np.where(points[:, self.t] == idx)]
            points = np.delete(points, self.t, axis=1)
        else:
            idx = slice(None)
        labels = np.array(labels[idx])
        image = np.array(image[idx])
        points = np.round(points).astype(int)
        labels = watershed_split(
                                 image,
                                 labels,
                                 points,
                                 compactness=200,
                                 connectivity_octahedron=7
                                 )
        viewer.layers['Labels'].data[idx] = labels
        viewer.layers['Points'].data = np.empty((0, self.ndim), dtype=float)
        viewer.layers['Labels'].refresh()


    def _select_colour(self, viewer):
        """
        Select colour for painting
        """
        if viewer.layers['Labels'].mode != 'pick':
            viewer.layers['Labels'].mode = 'pick'
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'


    def _fill(self, viewer):
        """
        Switch napari labels layer to fill mode
        """
        if viewer.layers['Labels'].mode != 'fill':
            viewer.layers['Labels'].mode = 'fill'
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'


    def _paint(self, viewer):
        """
        Switch napari labels layer to paint mode
        """
        if viewer.layers['Labels'].mode != 'paint':
            viewer.layers['Labels'].mode = 'paint'
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'


    def _save(self, viewer):
        """
        Save the annotated time_index as a zarr file
        """
        array = viewer.layers['Labels'].data
        if len(array.shape) > 3: # save the current frame
            time = self.viewer.dims.current_step[self.t]
            idx = [slice(None)] * self.ndim
            idx[self.t] = time
            idx = tuple(idx)
            array = array[idx]
        zarr.save_array(self.save_path, np.array(array))
        print("Labels saved at:")
        print(self.save_path)


# Split Objects
# -------------
def watershed_split(
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
    markers = np.zeros(labels.shape, dtype=bool)
    markers[coords] = True
    markers = ndi.label(markers)
    markers = markers[0]
    new_labels = watershed(
                           image,
                           markers=markers,
                           mask=mask,
                           compactness=compactness,
                           connectivity=connectivity
                           )
    new_labels[new_labels != 0] += labels.max()
    # assign new values to the original labels
    labels = np.where(mask, new_labels, labels)
    return labels


# Execute
# -------
if __name__ == '__main__':
    path = '/Users/amcg0011/Data/pia-tracking/191113_IVMTR26_Inj3_cang_exp3.zarr'
    #labs = '/Users/amcg0011/Data/pia-tracking/191113_IVMTR26_Inj3_cang_exp3_labels_t74_GT.zarr'
    labs = {
      'driver': 'zarr',
      'kvstore': {
        'driver': 'file',
        'path': '/Users/amcg0011/Data/pia-tracking/',
      },
      'path': '191113_IVMTR26_Inj3_cang_exp3_labels.zarr',
      'metadata': {
        'dtype': '<i4',
        'shape': [195, 512, 512, 33],
        'order': 'C'
      },
    }
    correct_labels(path, labs, None, scale=(1, 1, 1, 4))
