import numpy as np


def slice_points(points_layer, dims, ndim):
    tf = points_layer._transforms[1:3].simplified.inverse
    steps = tf.scale[:-ndim] * np.array(dims.range)[:-ndim, 2]
    pt = tf(np.array(dims.point))[:-ndim]
    data = points_layer.data
    sel = np.ones(data.shape[0], dtype=bool)
    for i, (p, st) in enumerate(zip(pt, steps)):
        sel &= p - st/2 <= data[:, i]
        sel &= data[:, i] < p + st/2
    return data[sel]
