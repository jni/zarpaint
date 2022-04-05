from magicgui import magic_factory


@magic_factory
def copy_data(
        napari_viewer: 'napari.viewer.Viewer',
        source_layer: 'napari.layers.Layer',
        target_layer: 'napari.layers.Layer',
        ):
    src_data = source_layer.data
    dst_data = target_layer.data

    ndim_src = src_data.ndim
    ndim_dst = dst_data.ndim
    slice_ = napari_viewer.dims.current_step
    slicing = slice_[:ndim_dst - ndim_src]
    dst_data[slicing] = src_data