class MissingDistributionsWarning(UserWarning):
    """A known income-paying fund has no distribution history in the data API."""


class ShortPeriodLengthError(Exception):
    """
    Raised when available period length is too short for the asset.
    """

    pass


class RollingWindowLengthBelowOneYearError(Exception):
    """
    Raised when rolling window size is below one year.
    """

    pass


class LongRollingWindowLengthError(Exception):
    """
    Raised when rolling window size is more than data history depth.
    """

    pass
