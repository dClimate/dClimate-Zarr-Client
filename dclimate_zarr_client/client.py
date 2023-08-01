"""
Functions that will map to endpoints in the flask app
"""

import datetime
import typing

import numpy as np
import xarray as xr

from xarray.core.variable import MissingDimensionsError
from .dclimate_zarr_errors import (
    ConflictingGeoRequestError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
    InvalidForecastRequestError
)
from .geo_temporal_utils import (
    check_dataset_size,
    check_has_data,
    get_forecast_dataset,
    get_data_in_time_range,
    get_single_point,
    get_points_in_circle,
    get_points_in_polygons,
    get_multiple_points,
    get_points_in_rectangle,
    rolling_aggregation,
    spatial_aggregation,
    temporal_aggregation,
    DEFAULT_POINT_LIMIT,
)
from .ipfs_retrieval import get_dataset_by_ipns_hash, get_ipns_name_hash
from .s3_retrieval import get_dataset_from_s3


def _prepare_dict(ds: xr.Dataset) -> dict:
    """Prepares dict containing metadata and values from dataset

    Args:
        ds (xr.Dataset): dataset to turn into dict

    Returns:
        dict: dict with metadata and data values included
    """
    var_name = list(ds.data_vars)[0]
    vals = ds[var_name].values
    ret_dict = {}
    dimensions = []
    ret_dict["unit of measurement"] = ds.attrs["unit of measurement"]
    if "time" in ds:
        ret_dict["times"] = (
            np.datetime_as_string(ds.time.values, unit="s").flatten().tolist()
        )
        dimensions.append("time")
    if "point" in ds.dims:
        ret_dict["points"] = list(zip(ds.latitude.values, ds.longitude.values))
        ret_dict["point_coords_order"] = ["latitude", "longitude"]
        dimensions.insert(0, "point")
        ret_dict["data"] = np.where(~np.isfinite(vals), None, vals).T.tolist()
    else:
        for dim in ds[var_name].dims:
            if dim != "time":
                ret_dict[f"{dim}s"] = ds[dim].values.flatten().tolist()
                dimensions.append(dim)
        ret_dict["data"] = np.where(~np.isfinite(vals), None, vals).tolist()
    ret_dict["dimensions_order"] = dimensions
    try:
        if ds.update_in_progress and not ds.update_is_append_only:
            ret_dict["update_date_range"] = ds.attrs["update_date_range"]
    except AttributeError:
        pass
    return ret_dict


def _prepare_netcdf_bytes(ds: xr.Dataset) -> bytes:
    """Drops nested attributes from zarr and sets 'updating date range'
    if the dataset is currently undergoing a historical update, then converts
    zarr to bytes representing netcdf

    Args:
        ds (xr.Dataset): dataset to turn into netcdf bytes

    Returns:
        bytes: netcdf representation of zarr as bytes
    """
    try:
        if ds.update_in_progress and not ds.update_is_append_only:
            update_date_range = ds.attrs["update_date_range"]
            ds.attrs[
                "updating date range"
            ] = f"{update_date_range[0]}-{update_date_range[1]}"
    except AttributeError:
        pass
    # remove nested and None attributes, which to_netcdf to bytes doesn't support
    for bad_key in [
        "bbox",
        "date range",
        "tags",
        "finalization date",
        "update_date_range",
    ]:
        if bad_key in ds.attrs:
            del ds.attrs[bad_key]
    return ds.to_netcdf()


def geo_temporal_query(
    dataset_name: str,
    source: str = "ipfs",
    bucket_name: str = None,
    forecast_reference_time: str = None,
    point_kwargs: dict = None,
    circle_kwargs: dict = None,
    rectangle_kwargs: dict = None,
    polygon_kwargs: dict = None,
    multiple_points_kwargs: dict = None,
    spatial_agg_kwargs: dict = None,
    temporal_agg_kwargs: dict = None,
    rolling_agg_kwargs: dict = None,
    time_range: typing.Optional[typing.List[datetime.datetime]] = None,
    as_of: typing.Optional[datetime.datetime] = None,
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
        dataset_name (str): name used to link dataset to an ipns_name hash
        bucket_name (str): S3 bucket name where the datasets are going to be fetched
        forecast_reference_time (str): Isoformatted string representing the desire date to return all available forecasts for
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
                    multiple_points_kwargs,
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
            "User requested spatial aggregation methods on a single point, \
            but these are mutually exclusive parameters. Only one may be requested at a time."
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
    # Set defaults to avoid Nones accidentally passed by users causing a TypeError
    if not point_limit:
        point_limit = DEFAULT_POINT_LIMIT
    # Use the provided dataset string to find the dataset via IPNS\
    if source == "ipfs":
        ipns_name_hash = get_ipns_name_hash(dataset_name)
        ds = get_dataset_by_ipns_hash(ipns_name_hash, as_of=as_of)
    elif source == "s3":
        ds = get_dataset_from_s3(dataset_name, bucket_name, forecast_reference_time=forecast_reference_time)
    else:
        raise ValueError("only possible sources are s3 and IPFS")
    # Filter data down temporally, then spatially, and check that the size of resulting dataset fits within the limit.
    # While a user can get the entire DS by providing no filters, \
    # this will almost certainly cause the size checks to fail
    if "forecast_reference_time" in ds and not forecast_reference_time:
        raise InvalidForecastRequestError("Forecast dataset requested without forecast reference time. \
                                   Provide a forecast reference time or request to a different dataset if you desire observations, not projections.")
    if forecast_reference_time:
        if "forecast_reference_time" in ds:
            ds = get_forecast_dataset(ds, forecast_reference_time)
        else:
            raise MissingDimensionsError(f"Forecasts are not available for the requested dataset {dataset_name}")
    if time_range:
        ds = get_data_in_time_range(ds, *time_range)
    if point_kwargs:
        ds = get_single_point(ds, **point_kwargs)
    elif circle_kwargs:
        ds = get_points_in_circle(ds, **circle_kwargs)
    elif rectangle_kwargs:
        ds = get_points_in_rectangle(ds, **rectangle_kwargs)
    elif polygon_kwargs:
        ds = get_points_in_polygons(ds, **polygon_kwargs, point_limit=point_limit)
    elif multiple_points_kwargs:
        ds = get_multiple_points(ds, **multiple_points_kwargs)
    # Check that size of reduced data won't prove too expensive to request and process, according to specified limits
    check_dataset_size(ds, point_limit)
    check_has_data(ds)
    if multiple_points_kwargs:
        # Aggregations pull whole dataset when ds is structured as multiple points. Forcing xarray to do subsetting
        # before aggregation drastically speeds up agg
        ds = ds.compute()
    # Perform all requested valid aggregations. First aggregate data spatially, then temporally or on a rolling basis.
    if spatial_agg_kwargs:
        ds = spatial_aggregation(ds, **spatial_agg_kwargs)
    if temporal_agg_kwargs:
        ds = temporal_aggregation(ds, **temporal_agg_kwargs)
    elif rolling_agg_kwargs:
        ds = rolling_aggregation(ds, **rolling_agg_kwargs)
    # Export
    if output_format == "netcdf":
        return _prepare_netcdf_bytes(ds)
    else:
        return _prepare_dict(ds)
