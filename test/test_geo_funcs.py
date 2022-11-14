import pytest

import dclimate_zarr_client.geo_utils as geo_utils
from dclimate_zarr_client.dclimate_zarr_errors import NoDataFoundError


def test_get_single_point(input_ds):
    """
    Test the single point geographic extraction method
    """
    point = geo_utils.get_single_point(input_ds, 40, -120)
    assert point["u100"].values.shape == (168,)


def test_get_single_point_misaligned_error(input_ds):
    with pytest.raises(NoDataFoundError):
        geo_utils.get_single_point(input_ds, 40, -120.1, snap_to_grid=False)


def test_get_points_in_circle(input_ds):
    """
    Test the circular points geographic extraction method
    """
    ds = geo_utils.get_points_in_circle(input_ds, 40, -120, 50)
    assert float(ds.latitude.min()) == 39.75
    assert float(ds.latitude.max()) == 40.25
    assert float(ds.longitude.min()) == -120.5
    assert float(ds.longitude.max()) == -119.5


def test_representative_point(input_ds, polygons_mask):
    """
    Test the representative point (approximate centroid guaranteed to be within a polygon) geographic extraction method
    """
    rep_pt_ds = geo_utils.reduce_polygon_to_point(input_ds, polygons_mask=polygons_mask)
    assert rep_pt_ds["u100"].values[0] == pytest.approx(0.9564208984375)


def test_rolling_aggregation(input_ds):
    """
    Test various descriptive statistics methods with the rolling aggregation approach
    """
    mean_vals = geo_utils.rolling_aggregation(input_ds, 5, "mean")
    max_vals = geo_utils.rolling_aggregation(input_ds, 5, "max")
    std_vals = geo_utils.rolling_aggregation(input_ds, 5, "std")
    assert mean_vals["u100"].values[0][0][0] == pytest.approx(7.912628173828125)
    assert max_vals["u100"].values[0][0][0] == pytest.approx(8.5950927734375)
    assert std_vals["u100"].values[0][0][0] == pytest.approx(0.5272848606109619)


def test_temporal_aggregation(input_ds):
    """
    Test various descriptive statistics methods with the temporal (resampling) aggregation approach
    """
    # [0][0][0] returns the first value for latitude 45.0, longitue -140.0
    daily_maxs = geo_utils.temporal_aggregation(
        input_ds, time_period="day", agg_method="max", time_unit=2
    )
    monthly_means = geo_utils.temporal_aggregation(
        input_ds, time_period="month", agg_method="mean"
    )
    yearly_std = geo_utils.temporal_aggregation(
        input_ds, time_period="year", agg_method="std"
    )
    assert daily_maxs["u100"].values[0][0][0] == pytest.approx(8.5950927734375)
    assert monthly_means["u100"].values[0][0][0] == pytest.approx(-0.19848433136940002)
    assert yearly_std["u100"].values[0][0][0] == pytest.approx(6.490322113037109)


def test_spatial_aggregation(input_ds):
    """
    Test various descriptive statistics methods with the spatial aggregation approach
    """
    mean_vals_all_pts = geo_utils.spatial_aggregation(input_ds, "mean")
    min_val_rep_pt = geo_utils.spatial_aggregation(input_ds, "min")
    assert float(mean_vals_all_pts["u100"].values[0]) == pytest.approx(1.5880329608917236)
    assert float(min_val_rep_pt["u100"].values[0]) == pytest.approx(-9.5386962890625)
