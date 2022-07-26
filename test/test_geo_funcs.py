import pathlib

import dclimate_zarr_client.geo_utils as geo_utils
import pytest
import xarray as xr


@pytest.fixture
def input_ds():
    return xr.open_zarr(pathlib.Path(__file__).parent / "etc" / "retrieval_test.zarr")


def test_get_single_point(input_ds):
    point = geo_utils.get_single_point(input_ds, 40, -120)
    assert point.shape == (168,)


def test_get_points_in_circle(input_ds):
    ds = geo_utils.get_points_in_circle(input_ds, 40, -120, 50)
    assert float(ds.latitude.min()) == 39.75
    assert float(ds.latitude.max()) == 40.25
    assert float(ds.longitude.min()) == -120.5
    assert float(ds.longitude.max()) == -119.5
