try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._zarpaint import create_labels, open_tensorstore
from ._dims_chooser import DimsSorter, set_axis_labels
from ._watershed import watershed_split
from ._add_3d_points import add_points_3d_with_alt_click
from ._copy_data import copy_data

__all__ = [
        'create_labels',
        'open_tensorstore',
        'DimsSorter',
        'set_axis_labels',
        'watershed_split',
        'add_points_3d_with_alt_click',
        'copy_data',
        ]
