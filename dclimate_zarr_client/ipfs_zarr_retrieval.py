import typing

import numpy as np
import requests
import xarray as xr
from ipldstore import get_ipfs_mapper
from shapely.geometry.multipolygon import MultiPolygon

from .dclimate_zarr_errors import *
from .geo_utils import haversine

DEFAULT_HOST = "http://127.0.0.1:5001/api/v0"

# Users should not select more than this number of data points
POINT_LIMIT = 100 * 100 * 200_000


def get_dataset_by_ipns_hash(ipns_name_hash: str) -> xr.Dataset:
    r = requests.post(
        f"{DEFAULT_HOST}/name/resolve", params={"arg": ipns_name_hash}
    )
    r.raise_for_status()
    ipfs_hash = r.json()["Path"].split("/")[-1]
    return get_dataset_by_ipfs_hash(ipfs_hash)

    # TODO use "dag/get" to find the correct hash in the STAC metadata


def get_dataset_by_ipfs_hash(ipfs_hash: str) -> xr.Dataset:
    ipfs_mapper = get_ipfs_mapper()
    ipfs_mapper.set_root(ipfs_hash)
    return xr.open_zarr(ipfs_mapper)


def get_single_point(ds: xr.Dataset, latitude: float, longitude: float) -> np.ndarray:
    point_ds = ds.sel(latitude=latitude, longitude=longitude, method="nearest")
    var_name = list(point_ds.data_vars.keys())[0]
    return point_ds[var_name].values


def _check_dataset_size(ds: xr.Dataset):
    num_points = len(ds.latitude) * len(ds.longitude) * len(ds.time)
    if num_points > POINT_LIMIT:
        raise SelectionTooLargeError(
            f"Selection of ~ {num_points} data points is more than limit of {POINT_LIMIT}")


def get_points_in_circle(ds: xr.Dataset, center_lat: float, center_lon: float, radius: float) -> xr.Dataset:
    distances = haversine(center_lat, center_lon,
                          ds['latitude'], ds['longitude'])
    circle_ds = ds.where(distances < radius, drop=True)
    _check_dataset_size(circle_ds)
    return circle_ds


def get_points_in_rectangle(ds: xr.Dataset, min_lat: float, min_lon: float, max_lat: float, max_lon) -> xr.Dataset:
    rectangle_ds = ds.where(
        (min_lat < ds.latitude) &
        (ds.latitude < max_lat) &
        (min_lon < ds.longitude) &
        (ds.longitude < max_lon),
        drop=True
    )
    _check_dataset_size(rectangle_ds)
    return rectangle_ds


def get_points_in_polygons(ds: xr.Dataset, polygons_mask: typing.List[MultiPolygon], epsg_crs: int) -> xr.Dataset:
    ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    ds.rio.write_crs("epsg:4326", inplace=True)
    shaped_ds = ds.rio.clip(polygons_mask, epsg_crs, drop=True)
    _check_dataset_size(shaped_ds)
    return shaped_ds
