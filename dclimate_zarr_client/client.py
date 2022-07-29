"""
Functions that will map to endpoints in the flask app
"""

import datetime
import typing

import numpy as np
import xarray as xr

from .dclimate_zarr_errors import SelectionTooLargeError, ConflictingGeoRequestError, ConflictingAggregationRequestError
from .geo_utils import (
    get_data_in_time_range,
    # get_single_point,
    get_points_in_circle,
    get_points_in_polygons,
    get_points_in_rectangle,
    rolling_aggregation,
    spatial_aggregation,
    temporal_aggregation,
    # get_points_in_polygons,
    # get_points_in_rectangle,
)
from .ipfs_retrieval import get_dataset_by_ipns_hash, get_ipns_name_hash

# Users should not select more than this number of data points
DEFAULT_POINT_LIMIT = 100 * 100 * 200_000
DEFAULT_AREA_LIMIT = 100 # units are square decimal degrees



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

def _check_request_area(ds, area_limit: int = DEFAULT_AREA_LIMIT):
    """Checks the total area of the request

    Args:
        ds (xr.Dataset): dataset to check area of
        point_limit (int, optional): limit for dataset area. Defaults to DEFAULT_AREA_LIMIT.

    Raises:
        SelectionTooLargeError: Raised when dataset area limit is violated
    """
    request_area = len(ds.latitude) * len(ds.longitude) * ds.attrs["resolution"]
    if request_area > area_limit:
        raise SelectionTooLargeError(
            f"Selection of ~ {request_area} square degrees area is more than limit of {area_limit}"
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


def geo_temporal_query(
    ipns_name: str,
    circle_kwargs: dict = None,
    rectangle_kwargs: dict = None,
    polygon_kwargs: dict = None,
    spatial_agg_kwargs: dict = None,
    temporal_agg_kwargs: dict = None,
    rolling_agg_kwargs: dict = None,
    time_range: typing.Optional[typing.List[datetime.datetime]] = None,
    as_of: typing.Optional[datetime.datetime] = None,
    area_limit: int = DEFAULT_AREA_LIMIT,
    point_limit: int = DEFAULT_POINT_LIMIT,
    output_format: str = "array"
) -> typing.Union[np.ndarray, bytes]:
    """Queries all data in a circle for a dataset
        Only one of circle, rectangle, or polygon kwargs may be provided

    Args:
        ipns_name (str): name used to link dataset to an ipns_name hash
        circle_kwargs (dict): a dictionary of parameters relevant to a circular query
        rectangle_kwargs (dict): a dictionary of parameters relevant to a rectangular query
        polygon_kwargs (dict): a dictionary of parameters relevant to a polygonal query
        spatial_agg_kwargs: a dictionary of parameters relevant to a spatial aggregation operation
        temporal_agg_kwargs: a dictionary of parameters relevant to a temporal aggregation operation
        rolling_agg_kwargs: a dictionary of parameters relevant to a rolling aggregation operation
        time_range (typing.Optional[typing.List[datetime.datetime]], optional):
            time range in which to subset data. Defaults to None.
        as_of (typing.Optional[datetime.datetime], optional):
            pull in most recent data created before this time. If None, just get most recent.
            Defaults to None.
        area_limit (int, optional): maximum area in decimal degrees squared that a user may request. Defaults to DEFAULT_AREA_LIMIT.
        point_limit (int, optional): maximum number of data points user can fill. Defaults to DEFAULT_POINT_LIMIT.
        output_format (str, optional): Current supported formats are `array` and `netcdf`. Defaults to "array".

    Returns:
        typing.Union[np.ndarray, bytes]: Output data as array (default) or NetCDF
    """
    # Use the provided dataset string to find the dataset via IPNS
    ipns_name_hash = get_ipns_name_hash(ipns_name)
    ds = get_dataset_by_ipns_hash(ipns_name_hash, as_of=as_of)
    if len([kwarg_dict for kwarg_dict in [circle_kwargs, rectangle_kwargs, polygon_kwargs] if kwarg_dict is not None]) > 1:
        raise ConflictingGeoRequestError(
            f"User requested more than one type of geographic query, but only one can be submitted at a time"
        )
    # Filter data down temporally, then spatially, and check that the size of resulting dataset fits within the limit
    if time_range:
        ds = get_data_in_time_range(ds, *time_range)
    if circle_kwargs:
        ds = get_points_in_circle(ds, **circle_kwargs)
    elif rectangle_kwargs:
        ds = get_points_in_rectangle(ds, **rectangle_kwargs)
    else:
        ds = get_points_in_polygons(ds, **polygon_kwargs)
    # NOTE should get_single_point be the final fallback option?
    ds.attrs["resolution"] = 0.25 # TODO remove
    _check_request_area(ds, area_limit)
    # Aggregate data spatially, then temporally/on a rolling basis
    if spatial_agg_kwargs:
        ds = spatial_aggregation(ds, **spatial_agg_kwargs)
    if temporal_agg_kwargs and rolling_agg_kwargs:
        raise ConflictingAggregationRequestError(
            f"User requested both rolling and temporal aggregation, but these are mutually exclusive. Only one may be requested at a time."
        )
    elif temporal_agg_kwargs:
        ds = temporal_aggregation(ds, **temporal_agg_kwargs)
    elif rolling_agg_kwargs:
        ds = rolling_aggregation(ds, **rolling_agg_kwargs)
        import ipdb; ipdb.set_trace(context=4)
    _check_dataset_size(ds, point_limit)
    # Export
    var_name = list(ds.data_vars)[0]
    if output_format == "netcdf":
        return ds[var_name].to_netcdf()
    else:
        return ds[var_name].values


    # def function(circle_kwargs, rectangle_kwargs, polygon_kwargs) -- these "kwargs" are dicts from a POST request
    # Exactly 0 or 1 of these dicts
    # Don't unpack in the function definition
    # Where do turn the dataset name into an IPNS hash -- function in IPFS retrieval
    # We're going to use Requests, not ipfshttpclient, so we'll need to rewrite functions that use those things
    # r = requests.post(f"{DEFAULT_HOST}/dag/get", params={"arg": ipfs_hash})

    # Prepare a powerful "geo_temporal_query"
    # Take a name hash --> DS
    # Circle keyword args, rectangle kwargs, polygon kwargs --- center_lat/center_lon / bbox / multi-polygon and CRS -- REJECT MORE THAN ONE
    # Apply timerange, spatial agg, temporal aggregation

    # Masking is efficient, mask before _check_dataset_size
    # Prevent aggregation if too many points because aggregation is expensive
    # Actual output size is not problematic

    # Will need some error handling
    # E.g. can't do rolling with temporal
    # Input validation methods that live in client
    # Prevent user from requesting the entire dataset for a single time point b/c this will be very expensive with our chunking strategy
    # Build error handling for this
    # Two checks -- Spatial Check and Total Points Check

    # What is the return value?
    # Return NetCDF or dict with Array and metadata {Data : Metadata}.
    # Make a list of truly essential metadata -- units, etc.
