import datetime
import pathlib

import numpy as np
import pytest
import xarray as xr
import zarr

import dclimate_zarr_client.client as client
from dclimate_zarr_client.dclimate_zarr_errors import (
    SelectionTooLargeError,
    ConflictingGeoRequestError,
    NoDataFoundError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
)


def patched_get_ipns_name_hash(ipns_key):
    """
    Patch ipns name hash retrieval function to return a prepared ipfs key referring to a testing dataset
    """
    return "bafyreiglm3xvfcwkjbdqlwg3mc6zgngxuyfj6tkgfb6qobtmlzobpp63sq"


def patched_get_dataset_by_ipns_hash(ipfs_hash, as_of):
    """
    Patch ipns dataset function to return a prepared dataset for testing
    """
    with zarr.ZipStore(
            pathlib.Path(__file__).parent / "etc" / "sample_zarrs" / f"{ipfs_hash}.zip",
            mode="r",
    ) as in_zarr:
        return xr.open_zarr(in_zarr).compute()


@pytest.fixture(scope="module", autouse=True)
def default_session_fixture(module_mocker):
    """
    Patch IPNS dataset retrieval functions in this test
    """
    module_mocker.patch(
        "dclimate_zarr_client.client.get_ipns_name_hash",
        patched_get_ipns_name_hash,
    )
    module_mocker.patch(
        "dclimate_zarr_client.client.get_dataset_by_ipns_hash",
        patched_get_dataset_by_ipns_hash,
    )


def test_geo_temporal_query(polygons_mask, points_mask):
    """
    Test the `geo_temporal_query` method's functionalities: geographic queries, aggregation methods, and export formats
    Geographic queries include point, rectangle, circle, and polygon queries
    Aggregation methods include spatial, temporal, and rolling temporal approaches for various mathematical operations
    Exports can be of numpy array (default) or NetCDF format
    """
    point = client.geo_temporal_query(
        dataset_name="era5_wind_100m_u-hourly",
        bucket_name="zarr-dev",
        point_kwargs={"lat": 39.75, "lon": -118.5},
        rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
        point_limit=None,
    )
    rectangle = client.geo_temporal_query(
        dataset_name="era5_wind_100m_u-hourly",
        bucket_name="zarr-dev",
        rectangle_kwargs={
            "min_lat": 39.75,
            "min_lon": -120.5,
            "max_lat": 40.25,
            "max_lon": -119.5,
        },
    )
    rectangle_nc = client.geo_temporal_query(
        dataset_name="era5_wind_100m_u-hourly",
        bucket_name="zarr-dev",
        rectangle_kwargs={
            "min_lat": 39.75,
            "min_lon": -120.5,
            "max_lat": 40.25,
            "max_lon": -119.5,
        },
        spatial_agg_kwargs={"agg_method": "max"},
        output_format="netcdf",
    )
    circle = client.geo_temporal_query(
        dataset_name="era5_wind_100m_u-hourly",
        bucket_name="zarr-dev",
        circle_kwargs={"center_lat": 40, "center_lon": -120, "radius": 150},
        spatial_agg_kwargs={"agg_method": "std"},
        temporal_agg_kwargs={"time_period": "day", "agg_method": "std", "time_unit": 1},
    )
    polygon = client.geo_temporal_query(
        dataset_name="era5_wind_100m_u-hourly",
        bucket_name="zarr-dev",
        polygon_kwargs={"polygons_mask": polygons_mask, "epsg_crs": "epsg:4326"},
        spatial_agg_kwargs={"agg_method": "mean"},
        rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
    )

    points_arr = client.geo_temporal_query(
        dataset_name="era5_wind_100m_u-hourly",
        bucket_name="zarr-dev",
        multiple_points_kwargs={"points_mask": points_mask, "epsg_crs": "epsg:4326"},
    )
    points_nc = client.geo_temporal_query(
        dataset_name="era5_wind_100m_u-hourly",
        bucket_name="zarr-dev",
        multiple_points_kwargs={"points_mask": points_mask, "epsg_crs": "epsg:4326"},
        output_format="netcdf",
    )

    points_nc = xr.open_dataset(points_nc)

    for i, (lat, lon) in enumerate(points_arr["points"]):
        nc_vals = (
            points_nc.where(
                (points_nc.latitude == lat) & (points_nc.longitude == lon), drop=True
            )
            .u100.values.flatten()
            .tolist()
        )
        assert nc_vals == points_arr["data"][i]

    assert point["data"][0] == pytest.approx(-2.013934326171875)
    assert rectangle["data"][0][0][0] == pytest.approx(-1.9547119140625)
    assert rectangle_nc[0] == 67
    assert circle["data"][0] == pytest.approx(0.44366344809532166)
    assert polygon["data"][0] == pytest.approx(-1.1927716255187988)


def test_geo_conflicts():
    """
    Test that `geo_temporal_query` fails as predicted when conflicting geography requests are specified.
    """
    with pytest.raises(ConflictingGeoRequestError) as multi_exc_info:
        client.geo_temporal_query(
            dataset_name="era5_wind_100m_u-hourly",
            bucket_name="zarr-dev",
            rectangle_kwargs={
                "min_lat": 39.75,
                "min_lon": -120.5,
                "max_lat": 40.25,
                "max_lon": -119.5,
            },
            circle_kwargs={"center_lat": 40, "center_lon": -120, "radius": 10},
        )
    with pytest.raises(ConflictingGeoRequestError) as single_exc_info:
        client.geo_temporal_query(
            dataset_name="era5_wind_100m_u-hourly",
            bucket_name="zarr-dev",
            point_kwargs={"lat": 39.75, "lon": -118.5},
            spatial_agg_kwargs={"agg_method": "std"},
        )
    assert multi_exc_info.match("User requested more than one type of geographic query")
    assert single_exc_info.match(
        "User requested spatial aggregation methods on a single point"
    )


def test_temp_agg_conflicts():
    """
    Test that `geo_temporal_query` fails as predicted when conflicting temporal aggregation approaches are specified.
    """
    with pytest.raises(ConflictingAggregationRequestError):
        client.geo_temporal_query(
            dataset_name="era5_wind_100m_u-hourly",
            bucket_name="zarr-dev",
            rectangle_kwargs={
                "min_lat": 39.75,
                "min_lon": -120.5,
                "max_lat": 40.25,
                "max_lon": -119.5,
            },
            temporal_agg_kwargs={
                "time_period": "day",
                "agg_method": "std",
                "time_unit": 1,
            },
            rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
        )


def test_invalid_export():
    """
    Test that `geo_temporal_query` fails as predicted when invalid export formats are specified.
    """
    with pytest.raises(InvalidExportFormatError):
        client.geo_temporal_query(
            dataset_name="era5_wind_100m_u-hourly",
            bucket_name="zarr-dev",
            point_kwargs={"lat": 39.75, "lon": -118.5},
            output_format="GRIB",
        )


def test_selection_size_conflicts(oversized_polygons_mask):
    """
    Test that `geo_temporal_query` fails as predicted when selections of inappropriate size are requested.
    """
    with pytest.raises(SelectionTooLargeError) as too_many_points_exc_info:
        client.geo_temporal_query(
            dataset_name="era5_wind_100m_u-hourly",
            bucket_name="zarr-dev",
            circle_kwargs={"center_lat": 40, "center_lon": -120, "radius": 5000},
            point_limit=100,
        )

    assert too_many_points_exc_info.match("data points is more than limit of 100")


def test_no_data_in_selection_error():
    with pytest.raises(NoDataFoundError):
        client.geo_temporal_query(
            dataset_name="era5_wind_100m_u-hourly",
            bucket_name="zarr-dev",
            time_range=[datetime.datetime(1900, 1, 1), datetime.datetime(1910, 1, 1)],
            point_kwargs={"lat": 39.75, "lon": -118.5},
        )


def test_multiple_points_not_on_grid(points_mask):
    with pytest.raises(NoDataFoundError):
        client.geo_temporal_query(
            dataset_name="era5_wind_100m_u-hourly",
            bucket_name="zarr-dev",
            multiple_points_kwargs={
                "points_mask": points_mask,
                "epsg_crs": "epsg:4326",
                "snap_to_grid": False,
            },
        )


class TestClient:
    class TestGeoTemporalQueryFunction:

        @pytest.fixture()
        def fake_dataset(self):
            time = xr.DataArray(np.arange(5), dims="time", coords={"time": np.arange(5)})
            lat = xr.DataArray(np.arange(10, 50, 10), dims="lat",
                               coords={"lat": np.arange(10, 50, 10)})
            lon = xr.DataArray(np.arange(100, 140, 10), dims="lon",
                               coords={"lon": np.arange(100, 140, 10)})
            data = xr.DataArray(np.random.randn(5, 4, 4), dims=("time", "lat", "lon"),
                                coords=(time, lat, lon))

            fake_dataset = xr.Dataset({"data_var": data})
            return fake_dataset

        def test__given_bucket_and_dataset_names__then__fetch_geo_temporal_query_from_S3(
                self,
                mocker,
                fake_dataset
        ):
            dataset_name = "copernicus_ocean_salinity_1p5_meters"
            bucket_name = "zarr-prod",
            forecast_hour = None
            get_dataset_from_s3_mock = mocker.patch(
                "dclimate_zarr_client.client.get_dataset_from_s3", return_value=fake_dataset)
            mocker.patch("dclimate_zarr_client.client._prepare_dict", return_value=fake_dataset)

            client.geo_temporal_query(
                dataset_name=dataset_name,
                bucket_name=bucket_name,
                source="s3",
            )

            get_dataset_from_s3_mock.assert_called_with(
                dataset_name,
                bucket_name,
                forecast_hour
            )
