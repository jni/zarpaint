import numpy as np
from zarpaint import watershed_split

def test_watershed_split_2d(make_napari_viewer):
    data = np.zeros(shape=(18,18), dtype="uint8")
    # create 2 squares with one corner overlapping
    data[1:10, 1:10] = 1
    data[8:17, 8:17] = 1
    print(data)

    # palce points in the centre of the 2 squares
    points = np.asarray([[5,5], [12,12]])

    expected = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype="uint8")

    viewer = make_napari_viewer()
    viewer.add_labels(data)
    viewer.add_points(points)
    watershed_widget = watershed_split()
    watershed_widget(viewer, viewer.layers[0], viewer.layers[1])
    np.testing.assert_allclose(viewer.layers[0].data, expected)




    
def test_watershed_split_3d(make_napari_viewer):
    data = np.zeros(shape=(9,9,9), dtype="uint8")
    data[4:,4:,4:] = 1
    data[:5,:5,:5] = 1
    viewer = make_napari_viewer()
    viewer.add_labels(data)

    points = np.asarray([[4,2,2], [4,6,6]])
    
    viewer.add_points(points)

    watershed_widget = watershed_split()
    watershed_widget(viewer, viewer.layers[0], viewer.layers[1])

    assert np.all(data[:5,:5,:5] == 2)
    assert np.all(data[5:,5:,5:] == 3)
