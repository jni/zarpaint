from __future__ import annotations

from magicgui import magic_factory
import numpy as np
import napari
from qtpy.QtWidgets import QStyledItemDelegate
from qtpy.QtGui import QBrush
from napari.utils.events import SelectableEventedList
from qtpy.QtWidgets import QWidget, QHBoxLayout


class AxisModel:
    """View of an axis within a dims model keeping track of axis names."""

    def __init__(self, dims: napari.components.Dims, axis: int):
        self.dims = dims
        self.axis = axis

    def __hash__(self):
        return id(self)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.dims.axis_labels[self.axis]

    def __eq__(self, other: int | str):
        if isinstance(other, int):
            return self.axis == other
        else:
            return repr(self) == other


# this class is currently unused — needed to paint visible dims differently
class DimsDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if index.row() > 3:
            brush = QBrush()
            brush.setColor('red')
            option.backgroundBrush = brush
        super().paint(painter, option, index)


def set_dims_order(dims, order):
    if type(order[0]) == AxisModel:
        order = [a.axis for a in order]
    dims.order = order


def _array_in_range(arr, low, high):
    return (arr >= low) & (arr < high)


def move_indices(axes_list, order):
    with axes_list.events.blocker_all():
        axes = [a.axis for a in axes_list]
        if tuple(axes_list) == tuple(order):
            return
        ax_to_existing_position = {a: ix for ix, a in enumerate(axes)}
        move_list = np.asarray([
            (ax_to_existing_position[order[i]], i)
            for i in range(len(order))
        ])
        for src, dst in move_list:
            axes_list.move(src, dst)
            move_list[_array_in_range(move_list[:, 0], dst, src)] += 1
        # remove the elements from the back if order has changed length
        while len(axes_list) > len(order):
            axes_list.pop()


class DimsSorter(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent=parent)
        from napari._qt.containers import QtListView
        dims = napari_viewer.dims
        root = SelectableEventedList(
            [AxisModel(dims, i) for i in range(dims.ndim)]
        )
        root.events.reordered.connect(
            lambda event, dims=dims: set_dims_order(dims, event.value)
        )
        dims.events.order.connect(
            lambda event, axes_list=root: move_indices(axes_list, event.value)
        )
        view = QtListView(root)
        self.axes_list = root
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(view)


@magic_factory(
    call_button='set axis labels',
    viewer={'visible': False},
)
def set_axis_labels(viewer: napari.Viewer, axes=''):
    if type(axes) == str and len(axes) == viewer.dims.ndim:
        viewer.dims.axis_labels = list(axes)


if __name__ == '__main__':
    from napari._qt.containers import QtListView
    from napari.components import Dims

    dims = Dims(ndim=5, ndisplay=2, last_used=0, axis_labels=list('tzcyx'))
    # create a python model
    root = SelectableEventedList([AxisModel(dims, i) for i in range(5)])
    root.events.reordered.connect(
        lambda event, dims=dims: set_dims_order(dims, event.value)
    )
    # note: we don't yet handle expanding the order
    dims.events.order.connect(
        lambda event, axes_list=root: move_indices(axes_list, event.value)
    )

    # create Qt views onto the python models
    view = QtListView(root)

    w = QWidget()
    w.setLayout(QHBoxLayout())
    w.layout().addWidget(view)
    w.show()

    dims.order = (4, 3, 1, 0, 2)  # xyztc

    napari.run()
