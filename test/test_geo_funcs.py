import pytest

import dclimate_zarr_client.geo_temporal_utils as geo_temporal_utils
from dclimate_zarr_client.dclimate_zarr_errors import NoDataFoundError


def test_get_single_point(input_ds):
    """
    Test the single point geographic extraction method
    """
    point = geo_temporal_utils.get_single_point(input_ds, 40, -120)
    assert point["u100"].values.shape == (168,)


def test_get_single_point_misaligned_error(input_ds):
    with pytest.raises(NoDataFoundError):
        geo_temporal_utils.get_single_point(input_ds, 40, -120.1, snap_to_grid=False)


def test_get_points_in_circle(input_ds):
    """
    Test the circular points geographic extraction method
    """
    ds = geo_temporal_utils.get_points_in_circle(input_ds, 40, -120, 50)
    assert float(ds.latitude.min()) == 39.75
    assert float(ds.latitude.max()) == 40.25
    assert float(ds.longitude.min()) == -120.5
    assert float(ds.longitude.max()) == -119.5


def test_get_point_for_small_polygon(input_ds, undersized_polygons_mask):
    """
    Test that providing get_points_in_polygons a polygon_mask smaller than any grid cell returns a single
    point dataset for the point closest to that polygon's centroid
    """
    ds = geo_temporal_utils.get_points_in_polygons(
        input_ds, polygons_mask=undersized_polygons_mask
    )
    assert ds["u100"].values[0] == pytest.approx(-2.3199463)


def test_rolling_aggregation(input_ds):
    """
    Test various descriptive statistics methods with the rolling aggregation approach
    """
    mean_vals = geo_temporal_utils.rolling_aggregation(input_ds, 5, "mean")
    max_vals = geo_temporal_utils.rolling_aggregation(input_ds, 5, "max")
    std_vals = geo_temporal_utils.rolling_aggregation(input_ds, 5, "std")
    assert mean_vals["u100"].values[0][0][0] == pytest.approx(7.912628173828125)
    assert max_vals["u100"].values[0][0][0] == pytest.approx(8.5950927734375)
    assert std_vals["u100"].values[0][0][0] == pytest.approx(0.5272848606109619)


def test_temporal_aggregation(input_ds):
    """
    Test various descriptive statistics methods with the temporal (resampling) aggregation approach
    """
    # [0][0][0] returns the first value for latitude 45.0, longitue -140.0
    daily_maxs = geo_temporal_utils.temporal_aggregation(
        input_ds, time_period="day", agg_method="max", time_unit=2
    )
    monthly_means = geo_temporal_utils.temporal_aggregation(
        input_ds, time_period="month", agg_method="mean"
    )
    yearly_std = geo_temporal_utils.temporal_aggregation(
        input_ds, time_period="year", agg_method="std"
    )
    assert daily_maxs["u100"].values[0][0][0] == pytest.approx(8.5950927734375)
    assert monthly_means["u100"].values[0][0][0] == pytest.approx(-0.19848469)
    assert yearly_std["u100"].values[0][0][0] == pytest.approx(6.490322113037109)


def test_spatial_aggregation(input_ds):
    """
    Test various descriptive statistics methods with the spatial aggregation approach
    """
    mean_vals_all_pts = geo_temporal_utils.spatial_aggregation(input_ds, "mean")
    min_val_rep_pt = geo_temporal_utils.spatial_aggregation(input_ds, "min")
    assert float(mean_vals_all_pts["u100"].values[0]) == pytest.approx(
        1.5880329608917236
    )
    assert float(min_val_rep_pt["u100"].values[0]) == pytest.approx(-9.5386962890625)
