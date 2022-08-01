import pytest
import xarray as xr
import geopandas as gpd
import pathlib

@pytest.fixture 
def input_ds():
    return xr.open_zarr(pathlib.Path(__file__).parent / "etc" / "retrieval_test.zarr")

@pytest.fixture
def polygon_mask():
    shp = gpd.read_file(pathlib.Path(__file__).parent / "etc" / "northern_ca_counties.geojson")
    return shp.geometry.values
    
@pytest.fixture
def reduced_polygon_mask():
    shp = gpd.read_file(pathlib.Path(__file__).parent / "etc" / "central_northern_ca_counties.geojson")
    return shp.geometry.values
    