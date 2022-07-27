import pathlib

import dclimate_zarr_client.geo_utils as geo_utils
import pytest
import xarray as xr
import geopandas as gpd


@pytest.fixture
def input_ds():
    return xr.open_zarr(pathlib.Path(__file__).parent / "etc" / "retrieval_test.zarr")

@pytest.fixture
def polygon_mask():
    shp = gpd.read_file(pathlib.Path(__file__).parent / "etc" / "northern_ca_counties.shp")
    return shp.geometry.values
    

def test_get_single_point(input_ds):
    point = geo_utils.get_single_point(input_ds, 40, -120)
    assert point.shape == (168,)


def test_get_points_in_circle(input_ds):
    ds = geo_utils.get_points_in_circle(input_ds, 40, -120, 50)
    assert float(ds.latitude.min()) == 39.75
    assert float(ds.latitude.max()) == 40.25
    assert float(ds.longitude.min()) == -120.5
    assert float(ds.longitude.max()) == -119.5


def test_representative_point(input_ds, polygon_mask):
    rep_pt_ds = geo_utils.reduce_polygon_to_point(input_ds, polygon_mask=polygon_mask)
    assert rep_pt_ds["u100"].values[0] == 1.9847412109375


def test_rolling_aggregation(input_ds):
    mean_vals = geo_utils.rolling_aggregation(input_ds, 5, "mean")
    assert mean_vals["u100"].values[0][0][0] == 7.912628173828125
    max_vals = geo_utils.rolling_aggregation(input_ds, 5, "max")
    assert max_vals["u100"].values[0][0][0] == 8.5950927734375
    std_vals = geo_utils.rolling_aggregation(input_ds, 5, "std")
    assert std_vals["u100"].values[0][0][0] == 0.5272848606109619


def test_temporal_aggregation(input_ds):
    # [0][0][0] returns the first value for latitude 45.0, longitue -140.0 
    daily_maxs = geo_utils.temporal_aggregation(input_ds, time_period="day", agg_method="max", time_unit = 2)
    assert float(daily_maxs["u100"].values[0][0][0]) == 8.5950927734375
    monthly_means = geo_utils.temporal_aggregation(input_ds, time_period="month", agg_method="mean")
    assert float(monthly_means["u100"].values[0][0][0]) == -0.19848433136940002
    yearly_std = geo_utils.temporal_aggregation(input_ds, time_period="year", agg_method="std")
    assert float(yearly_std["u100"].values[0][0][0]) == 6.490322113037109