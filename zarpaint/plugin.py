from napari_plugin_engine import napari_hook_implementation
from ._zarpaint import create_labels


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return create_labels