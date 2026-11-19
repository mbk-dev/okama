"""Exceptions raised while computing the multi-period Efficient Frontier."""

from okama.common.error import ShortPeriodLengthError


class InsufficientFrontierHistoryError(ShortPeriodLengthError):
    """Too few usable monthly returns shared by the frontier assets."""

    def __init__(
        self, required_months: int, common_months: int, common_first_month: str | None, common_last_month: str | None
    ) -> None:
        super().__init__(required_months, common_months, common_first_month, common_last_month)
        self.required_months = required_months
        self.common_months = common_months
        self.common_first_month = common_first_month
        self.common_last_month = common_last_month

    def __str__(self) -> str:
        message = (
            f"The efficient frontier needs {self.required_months} common monthly returns. Found {self.common_months}"
        )
        if self.common_first_month is not None and self.common_last_month is not None:
            message += f" ({self.common_first_month} to {self.common_last_month})"
        return f"{message}."


class FrontierSolverError(RuntimeError):
    """The numerical optimizer did not find a verified frontier solution."""
