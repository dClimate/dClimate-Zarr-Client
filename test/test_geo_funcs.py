import pathlib

import dclimate_zarr_client.ipfs_zarr_retrieval as ipfs_zarr_retrieval
import pytest
import xarray as xr
from dclimate_zarr_client.dclimate_zarr_errors import SelectionTooLargeError

# Lower point limit so tests can trigger error
ipfs_zarr_retrieval.POINT_LIMIT = 10000


@pytest.fixture
def input_ds():
    return xr.open_zarr(pathlib.Path(__file__).parent / "etc" / "retrieval_test.zarr")


def test_get_single_point(input_ds):
    point = ipfs_zarr_retrieval.get_single_point(input_ds, 40, -120)
    assert point.shape == (168,)


def test_get_points_in_circle(input_ds):
    ds = ipfs_zarr_retrieval.get_points_in_circle(input_ds, 40, -120, 50)
    assert float(ds.latitude.min()) == 39.75
    assert float(ds.latitude.max()) == 40.25
    assert float(ds.longitude.min()) == -120.5
    assert float(ds.longitude.max()) == -119.5


def test_get_points_in_circle_too_many_points(input_ds):
    with pytest.raises(SelectionTooLargeError):
        ipfs_zarr_retrieval.get_points_in_circle(input_ds, 40, -120, 100)
