"""
Functions that will map to endpoints in the flask app
"""

import datetime
import typing

from xarray.core.variable import MissingDimensionsError
from .dclimate_zarr_errors import (
    ConflictingGeoRequestError,
    ConflictingAggregationRequestError,
    InvalidExportFormatError,
    InvalidForecastRequestError,
)
from .geotemporal_data import GeotemporalData, DEFAULT_POINT_LIMIT
from .ipfs_retrieval import get_dataset_by_ipns_hash, get_ipns_name_hash
from .s3_retrieval import get_dataset_from_s3


def load_ipns(
    dataset_name: str,
    as_of: typing.Optional[datetime.datetime] = None,
) -> GeotemporalData:
    """
    Load a Geotemporal dataset from IPLD.

    Parameters
    ----------

    dataset_name: str
        Name used to link dataset to an ipns_name hash
    as_of: datetime.datetime, optional
        Pull in most recent data created before this time. If ``None``, just get most
        recent. Defaults to ``None``.
    """
    ipns_name_hash = get_ipns_name_hash(dataset_name)
    ds = get_dataset_by_ipns_hash(ipns_name_hash, as_of=as_of)
    return GeotemporalData(ds)


def load_s3(
    dataset_name: str,
    bucket_name: str,
) -> GeotemporalData:
    """
    Load a Geotemporal dataset from an S3 bucket.

    Parameters
    ----------

    dataset_name: str
        The name of the dataset in the bucket.
    bucket_name: str
        S3 bucket name where the dataset is going to be fetched
    """
    ds = get_dataset_from_s3(dataset_name, bucket_name)
    return GeotemporalData(ds)


def geo_temporal_query(
    dataset_name: str,
    source: str = "ipfs",
    bucket_name: str = None,
    var_name: str = None,
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
    """Filter an XArray dataset

    Filter an XArray dataset by specified spatial and/or temporal bounds and aggregate
    according to spatial and/or temporal logic, if desired. Before aggregating check
    that the filtered data fits within specified point and area maximums to avoid
    computationally expensive retrieval and processing operations. When bounds or
    aggregation logic are not provided, pass the dataset along untouched.

    Return either a numpy array of data values or a NetCDF file.

    Only one of point, circle, rectangle, or polygon kwargs may be provided. Only one of
    temporal or rolling aggregation kwargs may be provided, although they can be chained
    with spatial aggregations if desired.

    Args:
        dataset_name (str): name used to link dataset to an ipns_name hash
        bucket_name (str): S3 bucket name where the datasets are going to be fetched
        forecast_reference_time (str): Isoformatted string representing the desire date
            to return all available forecasts for
        circle_kwargs (dict, optional): a dictionary of parameters relevant to a
            circular query
        rectangle_kwargs (dict, optional): a dictionary of parameters relevant to a
            rectangular query
        polygon_kwargs (dict, optional): a dictionary of parameters relevant to a
            polygonal query
        spatial_agg_kwargs (dict, optional): a dictionary of parameters relevant to a
            spatial aggregation operation
        temporal_agg_kwargs (dict, optional): a dictionary of parameters relevant to a
            temporal aggregation operation
        rolling_agg_kwargs (dict, optional): a dictionary of parameters relevant to a
            rolling aggregation operation
        time_range (typing.Optional[typing.List[datetime.datetime]], optional):
            time range in which to subset data.
            Defaults to None.
        as_of (typing.Optional[datetime.datetime], optional):
            pull in most recent data created before this time. If None, just get most
            recent. Defaults to None.
        area_limit (int, optional): maximum area in decimal degrees squared that a user
            may request. Defaults to DEFAULT_AREA_LIMIT.
        point_limit (int, optional): maximum number of data points user can fill.
            Defaults to DEFAULT_POINT_LIMIT.
        output_format (str, optional): Current supported formats are `array` and
            `netcdf`. Defaults to "array", which provides a numpy array of float32
            values.

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
            "User requested more than one type of geographic query, but only one can " "be submitted at a time"
        )
    if spatial_agg_kwargs and point_kwargs:
        raise ConflictingGeoRequestError(
            "User requested spatial aggregation methods on a single point, "
            "but these are mutually exclusive parameters. Only one may be requested at "
            "a time."
        )
    if temporal_agg_kwargs and rolling_agg_kwargs:
        raise ConflictingAggregationRequestError(
            "User requested both rolling and temporal aggregation, but these are "
            "mutually exclusive operations. Only one may be requested at a time."
        )
    if output_format not in ["array", "netcdf"]:
        raise InvalidExportFormatError(
            "User requested an invalid export format. Only 'array' or 'netcdf' " "permitted."
        )

    # Set defaults to avoid Nones accidentally passed by users causing a TypeError
    if not point_limit:
        point_limit = DEFAULT_POINT_LIMIT

    # Use the provided dataset string to find the dataset via IPNS
    if source == "ipfs":
        data = load_ipns(dataset_name, as_of=as_of)
    elif source == "s3":
        data = load_s3(dataset_name, bucket_name)
    else:
        raise ValueError("only possible sources are s3 and IPFS")

    # If specific variable is requested, use that
    if var_name is not None:
        data = data.use(var_name)

    # Filter data down temporally, then spatially, and check that the size of resulting
    # dataset fits within the limit. While a user can get the entire DS by providing no
    # filters, this will almost certainly cause the size checks to fail
    if "forecast_reference_time" in data.data and not forecast_reference_time:
        raise InvalidForecastRequestError(
            "Forecast dataset requested without forecast reference time. "
            "Provide a forecast reference time or request to a different dataset if "
            "you desire observations, not projections."
        )
    if forecast_reference_time:
        if "forecast_reference_time" in data.data:
            data = data.forecast(forecast_reference_time)
        else:
            raise MissingDimensionsError(f"Forecasts are not available for the requested dataset {dataset_name}")

    data = data.query(
        forecast_reference_time=forecast_reference_time,
        point_kwargs=point_kwargs,
        circle_kwargs=circle_kwargs,
        rectangle_kwargs=rectangle_kwargs,
        polygon_kwargs=polygon_kwargs,
        multiple_points_kwargs=multiple_points_kwargs,
        spatial_agg_kwargs=spatial_agg_kwargs,
        temporal_agg_kwargs=temporal_agg_kwargs,
        rolling_agg_kwargs=rolling_agg_kwargs,
        time_range=time_range,
        point_limit=point_limit,
    )

    # Export
    if output_format == "netcdf":
        return data.to_netcdf()
    else:
        return data.as_dict()
