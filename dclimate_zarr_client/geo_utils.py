import datetime
import typing

import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon


def get_single_point(ds: xr.Dataset, latitude: float, longitude: float) -> np.ndarray:
    """Gets array corresponding to the full time series for a single point in a dataset

    Args:
        ds (xr.Dataset): dataset to subset
        latitude (float): latitude coordinate
        longitude (float): longitude coordinate

    Returns:
        np.ndarray: time series array
    """
    point_ds = ds.sel(latitude=latitude, longitude=longitude, method="nearest")
    var_name = list(point_ds.data_vars)[0]
    return point_ds[var_name].values


def _haversine(
    lat1: typing.Union[np.ndarray, float],
    lon1: typing.Union[np.ndarray, float],
    lat2: typing.Union[np.ndarray, float],
    lon2: typing.Union[np.ndarray, float],
) -> typing.Union[np.ndarray, float]:
    """Calculates arclength distance in km between coordinate pairs,
        assuming the earth is a perfect sphere

    Args:
        lat1 (typing.Union[np.ndarray, float]): latitude coordinate for first point
        lon1 (typing.Union[np.ndarray, float]): longitude coordinate for first point
        lat2 (typing.Union[np.ndarray, float]): latitude coordinate for second point
        lon2 (typing.Union[np.ndarray, float]): longitude coordinate for second point

    Returns:
        typing.Union[np.ndarray, float]: distance between coordinate pairs in km
    """
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # radius of earth in km
    r = 6371
    return c * r


def get_points_in_circle(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
    radius: float,
) -> xr.Dataset:
    """Subsets dataset to points within radius of given center coordinates

    Args:
        ds (xr.Dataset): dataset to subset
        center_lat (float): latitude coordinate of center
        center_lon (float): longitude coordinate of center
        radius (float): radius of circle in kilometers

    Returns:
        xr.Dataset: subsetted dataset
    """
    distances = _haversine(center_lat, center_lon, ds["latitude"], ds["longitude"])
    circle_ds = ds.where(distances < radius, drop=True)
    return circle_ds


def get_points_in_rectangle(
    ds: xr.Dataset,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> xr.Dataset:
    """Subsets dataset to points in rectangle

    Args:
        ds (xr.Dataset): dataset to subset
        min_lat (float): southern limit of rectangle
        min_lon (float): western limit of rectangle
        max_lat (float): northern limit of rectangle
        max_lon (float): eastern limit of rectangle

    Returns:
        xr.Dataset: subsetted dataset
    """
    rectangle_ds = ds.where(
        (ds.latitude > min_lat)
        & (ds.latitude < max_lat)
        & (ds.longitude > min_lon)
        & (ds.longitude < max_lon),
        drop=True,
    )
    return rectangle_ds


def get_points_in_polygons(
    ds: xr.Dataset,
    polygons_mask: typing.List[MultiPolygon],
    epsg_crs: int,
) -> xr.Dataset:
    """Subsets dataset to points within arbitrary shape. Requires rioxarray to be installed

    Args:
        ds (xr.Dataset): dataset to subset
        polygons_mask (typing.List[MultiPolygon]): list of polygons defining shape
        epsg_crs (int): epsg code for polygons_mask (see https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset)

    Returns:
        xr.Dataset: subsetted dataset
    """
    ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    ds.rio.write_crs("epsg:4326", inplace=True)
    shaped_ds = ds.rio.clip(polygons_mask, epsg_crs, drop=True)
    return shaped_ds


def get_data_in_time_range(
    ds: xr.Dataset, start_time: datetime.datetime, end_time: datetime.datetime
) -> xr.Dataset:
    """Subset dataset to be within a contiguous time range. Can be combined with spatial subsetters defined above

    Args:
        ds (xr.Dataset): dataset to subset
        start_time (datetime.datetime): beginning of time range
        end_time (datetime.datetime): end of time range

    Returns:
        xr.Dataset: subsetted dataset
    """
    return ds.sel(time=slice(str(start_time), str(end_time)))


"""
Spatial aggregations to a single point, as well as full queries
Temporal aggregations to daily, weekly, yearly, full data
xarray.rolling
option to send back netcdf
"""
