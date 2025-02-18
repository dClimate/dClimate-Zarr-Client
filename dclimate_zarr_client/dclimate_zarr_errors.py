class ZarrClientError(Exception):
    """Parent class for library-level Exceptions"""


class SelectionTooLargeError(ZarrClientError):
    """Raised when user selects too many data points"""


class ConflictingGeoRequestError(ZarrClientError):
    """Raised when user requests more than one type of geographic query"""


class ConflictingAggregationRequestError(ZarrClientError):
    """Raised when user requests more than one type of geographic query"""


class NoMetadataFoundError(ZarrClientError):
    """Raised when user selects as_of before earliest existing metadata"""


class NoDataFoundError(ZarrClientError):
    """Raised when user's selection is all NA"""


class DatasetNotFoundError(ZarrClientError):
    """Raised when dataset not available over IPNS"""


class InvalidForecastRequestError(ZarrClientError):
    """Raised when regular time series are requested from a forecast dataset"""


class InvalidAggregationMethodError(ZarrClientError):
    """Raised when user provides an aggregation method outside of [min, max, median,
    mean, std, sum]"""


class InvalidTimePeriodError(ZarrClientError):
    """Raised when user provides a time period outside of [hour, day, week, month,
    quarter, year]"""


class InvalidExportFormatError(ZarrClientError):
    """Raised when user specifies an export format other than [array, netcdf]"""


class BucketNotFoundError(ZarrClientError):
    """Raised when bucket does not exist in AWS S3"""


class PathNotFoundError(ZarrClientError):
    """Raised when bucket does not exist in AWS S3"""


class AmbiguousDataVariableError(ZarrClientError):
    """Raised when method that requires a specific data variable is called, the dataset
    has more than variable, and the dataset hasn't been specified by a call to
    :method:`dclimate_zarr_client.geotemporal_data.GeotemporalData.use`"""
