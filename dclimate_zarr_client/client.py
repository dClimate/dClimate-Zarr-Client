"""
Functions that will map to endpoints in the flask app
"""

import datetime
import typing

import numpy as np
import xarray as xr

from .dclimate_zarr_errors import SelectionTooLargeError, InvalidAggregationMethodError, InvalidTimePeriodError
from .geo_utils import (
    get_data_in_time_range,
    # get_single_point,
    get_points_in_circle,
    # get_points_in_polygons,
    # get_points_in_rectangle,
)
from .ipfs_retrieval import get_dataset_by_ipns_hash

# Users should not select more than this number of data points
DEFAULT_POINT_LIMIT = 100 * 100 * 200_000



def _check_dataset_size(ds: xr.Dataset, point_limit: int = DEFAULT_POINT_LIMIT):
    """Checks how many data points are in a dataset

    Args:
        ds (xr.Dataset): dataset to check size of
        point_limit (int, optional): limit for dataset size. Defaults to DEFAULT_POINT_LIMIT.

    Raises:
        SelectionTooLargeError: Raised when dataset size limit is violated
    """
    num_points = len(ds.latitude) * len(ds.longitude) * len(ds.time)
    if num_points > point_limit:
        raise SelectionTooLargeError(
            f"Selection of ~ {num_points} data points is more than limit of {point_limit}"
        )

def circle_query(
    ipns_name_hash: str,
    center_lat: float,
    center_lon: float,
    radius: float,
    time_range: typing.Optional[typing.List[datetime.datetime]] = None,
    as_of: typing.Optional[datetime.datetime] = None,
    point_limit: int = DEFAULT_POINT_LIMIT,
    output_format: str = "array"
) -> typing.Union[np.ndarray, bytes]:
    """Queries all data in a circle for a dataset

    Args:
        ipns_name_hash (str): fixed ipns hash from which to get data
        center_lat (float): latitude coordinate of center
        center_lon (float): longitude coordinate of center
        radius (float): radius of circle in kilometers
        time_range (typing.Optional[typing.List[datetime.datetime]], optional):
            time range in which to subset data. Defaults to None.
        as_of (typing.Optional[datetime.datetime], optional):
            pull in most recent data created before this time. If None, just get most recent.
            Defaults to None.
        point_limit (int, optional): maximum number of data points user can fill. Defaults to DEFAULT_POINT_LIMIT.
        output_format (str, optional): Current supported formats are `array` and `netcdf`. Defaults to "array".

    Returns:
        typing.Union[np.ndarray, bytes]: Output data as array or bytes representing a netcdf file
    """

    ds = get_dataset_by_ipns_hash(ipns_name_hash, as_of=as_of)
    ds = get_points_in_circle(ds, center_lat, center_lon, radius)
    if time_range:
        ds = get_data_in_time_range(ds, *time_range)
    _check_dataset_size(ds, point_limit)
    var_name = list(ds.data_vars)[0]
    if output_format == "netcdf":
        return ds[var_name].to_netcdf()
    else:
        return ds[var_name].values
    # arr_of_vals = ds[var_name].values
