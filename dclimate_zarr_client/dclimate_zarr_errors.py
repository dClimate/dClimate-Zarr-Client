class ZarrClientError(Exception):
    """Parent class for library-level Exceptions"""

    pass


class SelectionTooLargeError(ZarrClientError):
    """Raised when user selects too many data points"""

    pass


class ConflictingGeoRequestError(ZarrClientError):
    """Raised when user requests more than one type of geographic query"""

    pass


class ConflictingAggregationRequestError(ZarrClientError):
    """Raised when user requests more than one type of geographic query"""

    pass


class NoMetadataFoundError(ZarrClientError):
    """Raised when user selects as_of before earliest existing metadata"""

    pass


class NoDataFoundError(ZarrClientError):
    """Raised when user's selection is all NA"""

    pass


class DatasetNotFoundError(ZarrClientError):
    """Raised when dataset not available over IPNS"""

    pass


class InvalidForecastRequestError(ZarrClientError):
    """Raised when regular time series are requested from a forecast dataset"""

    pass


class InvalidAggregationMethodError(ZarrClientError):
    """Raised when user provides an aggregation method outside of [min, max, median, mean, std, sum]"""

    pass


class InvalidTimePeriodError(ZarrClientError):
    """Raised when user provides a time period outside of [hour, day, week, month, quarter, year]"""

    pass


class InvalidExportFormatError(ZarrClientError):
    """Raised when user specifies an export format other than [array, netcdf]"""

    pass


class BucketNotFoundError(ZarrClientError):
    """Raised when bucket does not exist in AWS S3"""

    pass


class PathNotFoundError(ZarrClientError):
    """Raised when bucket does not exist in AWS S3"""

    pass
