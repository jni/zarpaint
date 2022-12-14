from typing import List, Tuple
from zarpaint.reader import zarr_tensorstore
import zarr
import numpy as np


def test_reader_pass(tmp_path):
    """
    Test that the reader is able to open a test file
    """
    new_path = tmp_path / "example.test"
    test_file = zarr_tensorstore(new_path)
    assert test_file is None


def test_reader_return_callable(tmp_path):
    """
    Test the the reader returns a valid funciton when opening a file
    """
    example_zarr_folder = tmp_path / 'example.zarr'
    z1 = zarr.open_array(
            example_zarr_folder,
            mode='w',
            shape=(10000, 10000),
            chunks=(1000, 1000),
            dtype='i4',
            fill_value=0
            )
    res = zarr_tensorstore(example_zarr_folder)
    assert callable(res)


def test_reader_can_read_and_write_to_file(tmp_path):
    """
    Creates a zarr file, writes random data to it, then saves the file. Once saved, the file is then 
    reopened and the data is compared.
    """
    example_zarr_folder = tmp_path / 'example.zarr'
    z1 = zarr.open_array(
            example_zarr_folder, mode='w', shape=(100, 100), chunks=(100, 100)
            )
    z1[:] = np.random.rand(100, 100)

    reader_func = zarr_tensorstore(example_zarr_folder)

    layers = reader_func(example_zarr_folder)
    assert isinstance(layers, List)
    assert len(layers) == 1

    layer_info = layers[0]
    assert isinstance(layer_info, Tuple)
    np.testing.assert_allclose(np.asarray(layer_info[0]), z1)
