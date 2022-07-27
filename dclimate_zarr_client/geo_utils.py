import datetime
import typing

import numpy as np
import pandas as pd
import xarray as xr
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape

from dclimate_zarr_client.dclimate_zarr_errors import InvalidAggregationMethodError, InvalidTimePeriodError


def _check_input_parameters(time_period = None, agg_method = None):
    """Checks whether input parameters align with permitted time periods and aggregation methods
    
    Args:
        time_period (str, optional): a string specifying the time period to resample a dataset by
        agg_method (str, optional): a string specifying the aggregation method to use on a dataset

    Raises:
        InvalidTimePeriodError: Raised when the specified time period is not accepted
        InvalidAggregationMethodError: Raised when the specified aggregation method is not accepted
    """
    if time_period and time_period not in ["hour", "day", "week", "month", "quarter", "year", "all"]:
        raise InvalidTimePeriodError(
            f"Specified time period {time_period} not among permitted periods: 'hour', 'day', 'week', 'month', 'quarter', 'year', 'all'"
        )
    if agg_method and agg_method not in ["min", "max", "median", "mean", "std", "sum"]:
        raise InvalidAggregationMethodError(
            f"Specified method {agg_method} not among permitted methods: 'min', 'max', 'median', 'mean', 'std', 'sum'"
        )

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
    polygons_mask: pd.Series(shapely.geometry.Polygon),
    epsg_crs: int,
) -> xr.Dataset:
    """Subsets dataset to points within arbitrary shape. Requires rioxarray to be installed

    Args:
        ds (xr.Dataset): dataset to subset
        polygons_mask (pd.Series(Polygon)): list of polygons defining shape
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


def reduce_polygon_to_point(
    ds: xr.Dataset,
    polygon_mask: pd.Series(shapely.geometry.Polygon),
) -> xr.Dataset:
    """Subsets data to a representative point approximately at the center of an arbitrary shape.
        Note that this point will always be within the shape, even if the exact center is not.
        NOTE a more involved alternative would be to return the average for values in the entire polygon at this point

    Args:
        ds (xr.Dataset): dataset to subset
        polygons_mask (shapely.geometry.Polygon): polygon defining shape

    Returns:
        xr.Dataset: subsetted dataset
    """
    pt = MultiPolygon(polygon_mask).representative_point()
    ds = ds.sel(latitude=pt.y, longitude=pt.x, method="nearest")
    return ds


def rolling_aggregation(
    ds: xr.Dataset,
    window_size: int,
    agg_method: str,
) -> np.ndarray:
    """Creates a rolling aggregate of data values along a dataset's "time" dimension. 
        The size of the window and the aggregation method are specified by the user.
        Method must be one of "min", "max", "median", "mean", "std", or "sum".

    Args:
        ds (xr.Dataset): dataset to subset
        window_size (int): size of rolling window to apply
        method (str): method to aggregate by

    Returns:
        np.ndarray: time series array
    """
    _check_input_parameters(agg_method=agg_method)
    if agg_method not in ["min", "max", "median", "mean", "std", "sum"]:
        raise InvalidAggregationMethodError
    # Aggregate by the specified method over the specified rolling window length
    rolled = ds.rolling(time=window_size)
    aggregator = getattr(xr.core.rolling.DataArrayRolling, agg_method)
    rolled_agg = aggregator(rolled).dropna("time") # remove NAs at beginning/end of array where window size is not large enough to compute a value
    
    return rolled_agg


def temporal_aggregation(
    ds: xr.Dataset,
    time_period: str,
    agg_method: str,
    time_unit: int = 1,
) -> np.ndarray:
    """Subsets data to the average per specified time period.

    Args:
        ds (xr.Dataset): dataset to subset
        time_unit (int): number of time periods to aggregate by. Defaults to 1. Ignored if "all" time periods specified.
        time_period (str): time period to aggregate by, parsed into DateOffset objects as per https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        method (str): method to aggregate by

    Returns:
        np.ndarray: time series array
    """
    _check_input_parameters(time_period=time_period, agg_method=agg_method)
    period_strings = {"hour" : f"{time_unit}H", "day" : f"{time_unit}D", "week" : f"{time_unit}W", "month" : f"{time_unit}M", \
        "quarter" : f"{time_unit}Q", "year" : f"{time_unit}Y", "all" : f"{len(set(ds.time.dt.year.values))}Y"}
    # Resample by the specified time period and aggregate by the specified method
    resampled = ds.resample(time=period_strings[time_period])
    aggregator = getattr(xr.core.resample.DataArrayResample, agg_method)
    resampled_agg = aggregator(resampled)

    return resampled_agg
    
    

"""
Spatial aggregations to a single point, as well as full queries
Temporal aggregations to daily, weekly, yearly, full data
xarray.rolling
option to send back netcdf
"""
