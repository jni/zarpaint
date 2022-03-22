from zarpaint.reader import zarr_tensorstore
import zarr
import numpy as np

def test_reader_pass(tmp_path):
    new_path = tmp_path / "example.test"
    test_file = zarr_tensorstore(new_path)
    assert test_file is None

def test_reader_return_callable(tmp_path):
    example_zarr_folder = tmp_path / 'example.zarr'
    z1 = zarr.open_array(example_zarr_folder, mode='w', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4', fill_value=0)
    res = zarr_tensorstore(example_zarr_folder)
    assert callable(res)
    
def test_reader_returns_valid_function(tmp_path):
    example_zarr_folder = tmp_path / 'example.zarr'
    z1 = zarr.open_array(example_zarr_folder, mode='w', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4', fill_value=0)

    #TODO: actually generate random test data (np.random.random or something) as the same shape
    # z1[:] = my_random_array[:]

    res = zarr_tensorstore(example_zarr_folder)

    z2 = res(example_zarr_folder)
    
    print(z1)
    print(z2)
    assert z2 is not None 

    assert np.testing.assert_allclose(z1, z2)

