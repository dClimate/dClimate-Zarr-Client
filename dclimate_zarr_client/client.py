"""
Functions that will map to endpoints in the flask app
"""

import datetime
import typing

import numpy as np
import xarray as xr

from .dclimate_zarr_errors import (
    SelectionTooLargeError,
    SelectionTooSmallError,
    ConflictingGeoRequestError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
)
from .geo_utils import (
    get_data_in_time_range,
    get_single_point,
    get_points_in_circle,
    get_points_in_polygons,
    get_points_in_rectangle,
    rolling_aggregation,
    spatial_aggregation,
    temporal_aggregation,
)
from .ipfs_retrieval import get_dataset_by_ipns_hash, get_ipns_name_hash

# Users should not select more than this number of data points and coordinates
DEFAULT_POINT_LIMIT = 100 * 100 * 200_000
DEFAULT_AREA_LIMIT = (
    2500  # square coordinates, whatever their actual size in km or degrees
)


def _check_request_area(
    ds: xr.Dataset, area_limit: int = DEFAULT_AREA_LIMIT, spatial_agg_kwargs=None
):
    """Checks the total area of the request

    Args:
        ds (xr.Dataset): dataset to check area of
        point_limit (int, optional): limit for dataset area. Defaults to DEFAULT_AREA_LIMIT.

    Raises:
        SelectionTooLargeError: Raised when dataset area limit is violated
        SelectionTooSmallError: Raised when dataset is 1x1 and a spatial aggregation method is called
    """
    request_area = len(ds.latitude) * len(ds.longitude)
    if request_area > area_limit:
        raise SelectionTooLargeError(
            f"Selection of ~ {request_area} square coordinates is more than limit of {area_limit}"
        )
    elif request_area == 1 and spatial_agg_kwargs:
        raise SelectionTooSmallError(
            "Selection of 1 square degree is incompatible with spatial aggregation as it will return all 0s."
            " Consider re-submitting with a larger target area or radius."
        )


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


def _prepare_dict(ds: xr.Dataset):
    var_name = list(ds.data_vars)[0]
    vals = ds[var_name].values
    ret_dict = {}
    dimensions = []
    ret_dict["unit of measurement"] = ds.attrs["unit of measurement"]
    if "time" in ds:
        ret_dict["times"] = (
            np.datetime_as_string(ds.time.values.flatten(), unit="s").tolist(),
        )
        dimensions.append("time")
    if "longitude" in ds:
        ret_dict["longitudes"] = ds.longitude.values.flatten().tolist()
        dimensions.append("longitude")
    if "latitude" in ds:
        ret_dict["latitudes"] = ds.latitude.values.flatten().tolist()
        dimensions.append("latitude")
    ret_dict["data"] = np.where(~np.isfinite(vals), None, vals).tolist()
    ret_dict["dimensions_order"] = dimensions
    return ret_dict


def geo_temporal_query(
    ipns_key_str: str,
    point_kwargs: dict = None,
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
    output_format: str = "array",
) -> typing.Union[dict, bytes]:
    """Filter an XArray dataset by specified spatial and/or temporal bounds and aggregate \
        according to spatial and/or temporal logic, if desired.
        Before aggregating check that the filtered data fits within specified point and area maximums \
            to avoid computationally expensive retrieval and processing operations.
        When bounds or aggregation logic are not provided, pass the dataset along untouched.

        Return either a numpy array of data values or a NetCDF file.

        Only one of point, circle, rectangle, or polygon kwargs may be provided
        Only one of temporal or rolling aggregation kwargs may be provided, \
            although they can be chained with spatial aggregations if desired.

    Args:
        ipns_key_str (str): name used to link dataset to an ipns_name hash
        circle_kwargs (dict, optional): a dictionary of parameters relevant to a circular query
        rectangle_kwargs (dict, optional): a dictionary of parameters relevant to a rectangular query
        polygon_kwargs (dict, optional): a dictionary of parameters relevant to a polygonal query
        spatial_agg_kwargs (dict, optional): a dictionary of parameters relevant to a spatial aggregation operation
        temporal_agg_kwargs (dict, optional): a dictionary of parameters relevant to a temporal aggregation operation
        rolling_agg_kwargs (dict, optional): a dictionary of parameters relevant to a rolling aggregation operation
        time_range (typing.Optional[typing.List[datetime.datetime]], optional):
            time range in which to subset data.
            Defaults to None.
        as_of (typing.Optional[datetime.datetime], optional):
            pull in most recent data created before this time. If None, just get most recent.
            Defaults to None.
        area_limit (int, optional): maximum area in decimal degrees squared that a user may request.
            Defaults to DEFAULT_AREA_LIMIT.
        point_limit (int, optional): maximum number of data points user can fill.
            Defaults to DEFAULT_POINT_LIMIT.
        output_format (str, optional): Current supported formats are `array` and `netcdf`.
            Defaults to "array", which provides a numpy array of float32 values.

    Returns:
        typing.Union[np.ndarray, bytes]: Output data as array (default) or NetCDF
    """
    # Check for incompatible request parameters
    if (
        len(
            [
                kwarg_dict
                for kwarg_dict in [
                    circle_kwargs,
                    rectangle_kwargs,
                    polygon_kwargs,
                    point_kwargs,
                ]
                if kwarg_dict is not None
            ]
        )
        > 1
    ):
        raise ConflictingGeoRequestError(
            "User requested more than one type of geographic query, but only one can be submitted at a time"
        )
    if spatial_agg_kwargs and point_kwargs:
        raise ConflictingGeoRequestError(
            "User requested spatial aggregation methods on a single point, but these are mutually exclusive parameters. \
                Only one may be requested at a time."
        )
    if temporal_agg_kwargs and rolling_agg_kwargs:
        raise ConflictingAggregationRequestError(
            "User requested both rolling and temporal aggregation, but these are mutually exclusive operations. \
                Only one may be requested at a time."
        )
    if output_format not in ["array", "netcdf"]:
        raise InvalidExportFormatError(
            "User requested an invalid export format. Only 'array' or 'netcdf' permitted."
        )
    # Use the provided dataset string to find the dataset via IPNS\
    ipns_name_hash = get_ipns_name_hash(ipns_key_str)
    ds = get_dataset_by_ipns_hash(ipns_name_hash, as_of=as_of)
    # Filter data down temporally, then spatially, and check that the size of resulting dataset fits within the limit.
    # While a user can get the entire DS by providing no filters, \
    # this will almost certainly cause the size checks to fail
    if time_range:
        ds = get_data_in_time_range(ds, *time_range)
    if point_kwargs:
        ds = get_single_point(ds, **point_kwargs)
    elif circle_kwargs:
        ds = get_points_in_circle(ds, **circle_kwargs)
    elif rectangle_kwargs:
        ds = get_points_in_rectangle(ds, **rectangle_kwargs)
    elif polygon_kwargs:
        ds = get_points_in_polygons(ds, **polygon_kwargs)
    # Check that size of reduced data won't prove too expensive to request and process, according to specified limits
    try:
        _check_request_area(ds, area_limit, spatial_agg_kwargs)
        _check_dataset_size(ds, point_limit)
    except TypeError:  # TypeError indicates a single point DS, which is always of acceptable size
        pass
    # Perform all requested valid aggregations. First aggregate data spatially, then temporally or on a rolling basis.
    if spatial_agg_kwargs:
        ds = spatial_aggregation(ds, **spatial_agg_kwargs)
    if temporal_agg_kwargs:
        ds = temporal_aggregation(ds, **temporal_agg_kwargs)
    elif rolling_agg_kwargs:
        ds = rolling_aggregation(ds, **rolling_agg_kwargs)
    # Export
    if output_format == "netcdf":
        # remove nested attributes, which to_netcdf to bytes doesn't support
        for bad_key in ["bbox", "date range", "tags"]:
            if bad_key in ds.attrs:
                del ds.attrs[bad_key]
        return ds.to_netcdf()
    else:
        return _prepare_dict(ds)
