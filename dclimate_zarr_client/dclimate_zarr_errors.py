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


class InvalidAggregationMethodError(ZarrClientError):
    """Raised when user provides an aggregation method outside of [min, max, median, mean, std, sum]"""

    pass

class InvalidTimePeriodError(ZarrClientError):
    """Raised when user provides a time period outside of [hour, day, week, month, quarter, year]"""

    pass

class InvalidSpatialUnitError(ZarrClientError):
    """Raised when user provides a spatial unit outside of [point, all]"""

    pass
