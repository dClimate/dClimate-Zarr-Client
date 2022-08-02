import pytest
import dclimate_zarr_client.client as client
import xarray as xr
import pathlib
import zipfile

from dclimate_zarr_client.dclimate_zarr_errors import (
    SelectionTooLargeError,
    SelectionTooSmallError,
    ConflictingGeoRequestError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
)


def patched_get_ipns_name_hash(ipns_key):
    """
    Patch ipns name hash retrieval function to return a prepared ipfs key referring to a testing dataset
    """
    return "bafyreiglm3xvfcwkjbdqlwg3mc6zgngxuyfj6tkgfb6qobtmlzobpp63sq"

def patched_get_dataset_by_ipns_hash(ipfs_hash):
    """
    Patch ipns dataset function to return a prepared dataset for testing
    """ 
    with zipfile.Zipfile(pathlib.Path(__file__).parent / "etc" / "sample_zarrs" / f"{ipfs_hash}.zarr") as zipped_zarr:
        zarr = xr.open_zarr(zipped_zarr)
    return zarr


@pytest.fixture(scope="module", autouse=True)
def default_session_fixture(module_mocker):
    """
    Patch metadata and Zarr retrieval functions in this test
    """
    module_mocker.patch(
        "dclimate_zarr_client.ipfs_retrieval.get_ipns_name_hash",
        patched_get_ipns_name_hash,
    )
    module_mocker.patch(
        "dclimate_zarr_client.ipfs_retrieval.get_dataset_by_ipns_hash",
        patched_get_dataset_by_ipns_hash,
    )


def test_geo_temporal_query(polygons_mask):
    """
    Test the `geo_temporal_query` method's functionalities: geographic queries, aggregation methods, and export formats
    Geographic queries include point, rectangle, circle, and polygon queries
    Aggregation methods include spatial, temporal, and rolling temporal approaches for various mathematical operations
    Exports can be of numpy array (default) or NetCDF format
    """
    point = client.geo_temporal_query(
        ipns_key_str="era5_wind_100m_u-hourly",
        point_kwargs={"lat": 39.75, "lon": -118.5},
        rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
    )
    rectangle = client.geo_temporal_query(
        ipns_key_str="era5_wind_100m_u-hourly",
        rectangle_kwargs={
            "min_lat": 39.75,
            "min_lon": -120.5,
            "max_lat": 40.25,
            "max_lon": -119.5,
        },
        spatial_agg_kwargs={"agg_method": "max"},
    )
    rectangle_nc = client.geo_temporal_query(
        ipns_key_str="era5_wind_100m_u-hourly",
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
        ipns_key_str="era5_wind_100m_u-hourly",
        circle_kwargs={"center_lat": 40, "center_lon": -120, "radius": 150},
        spatial_agg_kwargs={"agg_method": "std"},
        temporal_agg_kwargs={"time_period": "day", "agg_method": "std", "time_unit": 1},
    )
    polygon = client.geo_temporal_query(
        ipns_key_str="era5_wind_100m_u-hourly",
        polygon_kwargs={"polygons_mask": polygons_mask, "epsg_crs": "epsg:4326"},
        spatial_agg_kwargs={"agg_method": "mean"},
        rolling_agg_kwargs={"window_size": 5, "agg_method": "mean"},
    )

    assert point[0] == -2.013934326171875
    assert rectangle[0] == -1.7886962890625
    assert rectangle_nc[0] == 67
    assert circle[0] == 0.44366344809532166
    assert polygon[0] == -1.1927716255187988


def test_geo_conflicts():
    """
    Test that `geo_temporal_query` fails as predicted when conflicting geography requests are specified.
    """
    with pytest.raises(ConflictingGeoRequestError) as multi_exc_info:
        client.geo_temporal_query(
            ipns_key_str="era5_wind_100m_u-hourly",
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
            ipns_key_str="era5_wind_100m_u-hourly",
            point_kwargs={"lat": 39.75, "lon": -118.5},
            spatial_agg_kwargs={"agg_method": "std"},
        )
    assert (
        multi_exc_info.match("User requested more than one type of geographic query")
        == True
    )
    assert (
        single_exc_info.match(
            "User requested spatial aggregation methods on a single point"
        )
        == True
    )


def test_temp_agg_conflicts():
    """
    Test that `geo_temporal_query` fails as predicted when conflicting temporal aggregation approaches are specified.
    """
    with pytest.raises(ConflictingAggregationRequestError) as multi_agg_exc_info:
        client.geo_temporal_query(
            ipns_key_str="era5_wind_100m_u-hourly",
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
    assert multi_agg_exc_info.type == ConflictingAggregationRequestError


def test_invalid_export():
    """
    Test that `geo_temporal_query` fails as predicted when invalid export formats are specified.
    """
    with pytest.raises(InvalidExportFormatError) as invalid_export_exc_info:
        client.geo_temporal_query(
            ipns_key_str="era5_wind_100m_u-hourly",
            point_kwargs={"lat": 39.75, "lon": -118.5},
            output_format="GRIB",
        )
    assert invalid_export_exc_info.type == InvalidExportFormatError


def test_selection_size_conflicts(oversized_polygons_mask):
    """
    Test that `geo_temporal_query` fails as predicted when selections of inappropriate size are requested.
    """
    with pytest.raises(SelectionTooLargeError) as too_large_exc_info:
        client.geo_temporal_query(
            ipns_key_str="era5_wind_100m_u-hourly",
            polygon_kwargs={
                "polygons_mask": oversized_polygons_mask,
                "epsg_crs": "epsg:4326",
            },
        )
    with pytest.raises(SelectionTooSmallError) as too_small_exc_info:
        client.geo_temporal_query(
            ipns_key_str="era5_wind_100m_u-hourly",
            circle_kwargs={"center_lat": 40, "center_lon": -120, "radius": 10},
            spatial_agg_kwargs={"agg_method": "std"},
        )
    assert too_large_exc_info.type == SelectionTooLargeError
    assert too_small_exc_info.type == SelectionTooSmallError
