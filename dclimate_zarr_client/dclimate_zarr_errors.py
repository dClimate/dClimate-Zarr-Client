class ZarrClientError(Exception):
    """Parent class for library-level Exceptions"""

    pass


class SelectionTooLargeError(ZarrClientError):
    """Raised when user selects too many data points"""

    pass


class NoMetadataFoundError(ZarrClientError):
    """Raised when user selects as_of before earliest existing metadata"""

    pass
