import pytest
import xarray as xr
import zarr
import geopandas as gpd
import pathlib


@pytest.fixture
def input_ds():
    with zarr.ZipStore("etc/retrieval_test.zip", mode="r") as in_zarr:
        return xr.open_zarr(in_zarr).compute()


@pytest.fixture
def oversized_polygons_mask():
    shp = gpd.read_file(
        pathlib.Path(__file__).parent / "etc" / "northern_ca_counties.geojson"
    )
    return shp.geometry.values


@pytest.fixture
def polygons_mask():
    shp = gpd.read_file(
        pathlib.Path(__file__).parent / "etc" / "central_northern_ca_counties.geojson"
    )
    return shp.geometry.values