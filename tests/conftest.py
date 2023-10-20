import pytest
import xarray as xr
import zarr
import geopandas as gpd
import pathlib

HERE = pathlib.Path(__file__).parent
ETC = HERE / "etc"


@pytest.fixture
def input_ds():
    with zarr.ZipStore(ETC / "retrieval_test.zip", mode="r") as in_zarr:
        return xr.open_zarr(in_zarr, chunks=None).compute()


@pytest.fixture
def forecast_ds():
    with zarr.ZipStore(ETC / "forecast_retrieval_test.zip", mode="r") as in_zarr:
        return xr.open_zarr(in_zarr, chunks=None).compute()


@pytest.fixture
def oversized_polygons_mask():
    shp = gpd.read_file(ETC / "northern_ca_counties.geojson")
    return shp.geometry.values


@pytest.fixture
def undersized_polygons_mask():
    shp = gpd.read_file(ETC / "central_ca_farm.geojson")
    return shp.geometry.values


@pytest.fixture
def polygons_mask():
    shp = gpd.read_file(ETC / "central_northern_ca_counties.geojson")
    return shp.geometry.values


@pytest.fixture
def points_mask():
    points = gpd.read_file(ETC / "northern_ca_points.geojson")
    return points.geometry.values
