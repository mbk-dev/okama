from __future__ import annotations

import math
from typing import Optional, Union

import pandas as pd

import okama.portfolios.core as core
from okama import settings
from okama.common import validators


class CashFlow:
    """
    Parent class for cash flow strategies.

    Parameters
    ----------
    parent : Portfolio
        Parent Portfolio instance.
    """

    NAME = "cash_flow"

    def __init__(
        self,
        parent: core.Portfolio,
        frequency: Optional[str] = "none",
        initial_investment: float = 1000.0,
        time_series_dic: dict = {},
        time_series_discounted_values: bool = False,
    ):
        """
        Initialize the CashFlow strategy.

        Parameters
        ----------
        parent : Portfolio
            The parent Portfolio instance.
        frequency : str, optional
            Frequency of cash flows. Default is "none".
        initial_investment : float, optional
            Initial investment amount. Default is 1000.0.
        time_series_dic : dict, optional
            Dictionary with dates and cash flow values. Default is empty dict.
        time_series_discounted_values : bool, optional
            If True, values in time_series_dic are considered as discounted (PV). Default is False.
        """
        self.parent = parent
        self._frequency = frequency
        self._initial_investment = initial_investment
        self._pandas_frequency = settings.frequency_mapping.get(self.frequency)
        self.time_series = pd.Series(dtype=float)
        self.time_series_dic = time_series_dic
        self.time_series_discounted_values = time_series_discounted_values

    @property
    def frequency(self) -> str:
        """
        The frequency of regular withdrawals or contributions in the strategy.

        Allowed values for frequency:

        - 'none' no frequency (default value)
        - 'year' annual cash flows
        - 'half-year' 6 months cash flows
        - 'quarter' 3 months cash flows
        - 'month' 1 month cash flows

        Returns
        -------
        str
            The frequency of withdrawals or contributions.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        if frequency in settings.frequency_mapping.keys():
            self._clear_cf_cache()
            self._frequency = frequency
            self._pandas_frequency = settings.frequency_mapping.get(self.frequency)
        else:
            raise ValueError(
                f"frequency must be in {settings.frequency_mapping.keys()}"
            )

    @property
    def periods_per_year(self) -> int:
        """
        Get the number of periods per year based on the frequency.

        Returns
        -------
        int
            Number of periods per year.
        """
        return settings.frequency_periods_per_year[self.frequency]

    @property
    def initial_investment(self) -> float:
        """
        Initial investment (FV) amount at the start of the calculation period.

        For historical backtesting, this is the investment at `first_date`.
        For Monte Carlo simulations, this is the investment at `last_date`.
        Initial investment must be positive.

        Returns
        -------
        float
            Portfolio initial investment.
        """
        return self._initial_investment

    @initial_investment.setter
    def initial_investment(self, initial_investment):
        if initial_investment is not None:
            validators.validate_real("initial_investment", initial_investment)
            if initial_investment <= 0:
                raise ValueError("Initial investment must be positive.")
        self._clear_cf_cache()
        self._initial_investment = initial_investment

    @property
    def time_series_dic(self) -> dict:
        """
        Cash flow time series in form of dictionary.

        Negative number corresponds to withdrawals, positive number corresponds to contributions.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(first_date="2015-01", last_date="2024-10")  # create Portfolio with default parameters
        >>> # create simple dictionary with cash flow amounts and dates
        >>> d = {"2018-02": 2_000, "2024-03": -4_000}
        >>> ts = ok.TimeSeriesStrategy(pf)  # create TimeSeresStrategy linked to the portfolio
        >>> ts.time_series_dic = d  # use the dictionary to set cash flow
        >>> ts.initial_investment = 1_000  # add initial investments size (optional)
        >>> # Assign the strategy to Portfolio
        >>> pf.dcf.cashflow_parameters = ts
        >>> # Plot wealth index with cash flow
        >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
        >>> plt.show()
        """
        # TODO: rename to extra_cashflow_ts
        return self._time_series_dic

    @time_series_dic.setter
    def time_series_dic(self, time_series_dic):
        self._clear_cf_cache()
        if isinstance(time_series_dic, dict):
            self._time_series_dic = time_series_dic
        else:
            raise TypeError("time_series_dic must be a dictionary.")
        self._make_series_from_dic()

    def _make_series_from_dic(self):
        """
        Create cash flow time series in form of Pandas.Series.
        """
        self.time_series = pd.Series(self._time_series_dic)
        self.time_series.index = pd.to_datetime(self.time_series.index).to_period("M")
        self.time_series.sort_index(inplace=True)
        self.time_series.name = "cashflow_ts"

    def _clear_cf_cache(self):
        if hasattr(self.parent, "dcf"):
            self.parent.dcf._monte_carlo_wealth_fv = pd.DataFrame(dtype=float)
            self.parent.dcf._wealth_index_fv = pd.DataFrame(dtype=float)
            self.parent.dcf._cash_flow_fv = pd.DataFrame(dtype=float)
            self.parent.dcf._monte_carlo_cash_flow_fv = pd.DataFrame(dtype=float)


class IndexationStrategy(CashFlow):
    """
    Cash flow strategy with regular indexed withdrawals or contributions.

    Parameters
    ----------
    parent : Portfolio
        Parent Portfolio instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(first_date="2015-01", last_date="2024-10")  # create Portfolio with default parameters
    >>> # Set the cash flow strategy
    >>> ind = ok.IndexationStrategy(pf) # create IndexationStrategy linked to the portfolio
    >>> ind.initial_investment = 10_000  # add initial investments size
    >>> ind.frequency = "year"  # set cash flow frequency
    >>> ind.amount = -1_500  # set withdrawal size
    >>> ind.indexation = "inflation"
    >>> # Assign the strategy to Portfolio
    >>> pf.dcf.cashflow_parameters = ind
    >>> # Plot wealth index with cash flow
    >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
    >>> plt.show()
    """

    NAME = "fixed_amount"

    def __init__(
        self,
        parent: core.Portfolio,
        frequency: Optional[str] = "none",
        initial_investment: float = 1000.0,
        time_series_dic: dict = {},
        time_series_discounted_values: bool = False,
        amount: float = 0,
        indexation: Optional[Union[str, float]] = None,
    ):
        """
        Initialize the IndexationStrategy.

        Parameters
        ----------
        parent : Portfolio
            The parent Portfolio instance.
        frequency : str, optional
            Frequency of cash flows. Default is "none".
        initial_investment : float, optional
            Initial investment amount. Default is 1000.0.
        time_series_dic : dict, optional
            Dictionary with dates and cash flow values. Default is empty dict.
        time_series_discounted_values : bool, optional
            If True, values in time_series_dic are considered as discounted (PV). Default is False.
        amount : float, optional
            Amount of regular cash flow. Negative for withdrawals, positive for contributions. Default is 0.
            The frequency of withdrawals or contributions is determined by the `frequency` parameter.
        indexation : str or float, optional
            Indexation rate (CAGR). Can be 'inflation' (if portfolio has inflation data) or a float value. Default is None.
        """
        super().__init__(
            parent,
            frequency=frequency,
            initial_investment=initial_investment,
            time_series_dic=time_series_dic,
            time_series_discounted_values=time_series_discounted_values,
        )
        self.portfolio = self.parent
        self._amount = amount
        self.indexation = indexation

    def __repr__(self):
        dic = {
            "Strategy name": self.NAME,
            "Portfolio symbol": self.parent.symbol,
            "Cash flow initial investment": self.initial_investment,
            "Cash flow frequency": self.frequency,
            "Cash flow strategy": self.NAME,
            "Cash flow amount": self.amount,
            "Cash flow indexation": self.indexation,
        }
        return repr(pd.Series(dic))

    @property
    def amount(self):
        """
        Portfolio regular withdrawal or contribution size.
        Negative value corresponds to withdrawals. Positive value corresponds to contributions.
        Cash flow value is indexed each period by 'indexation'.

        The frequency of withdrawals or contributions is determined by the `frequency` parameter.

        Returns
        -------
        float
            Portfolio regular cash flow size.
        """
        return self._amount

    @amount.setter
    def amount(self, amount):
        self._clear_cf_cache()
        validators.validate_real("amount", amount)
        if amount < 0 and abs(amount) > self.initial_investment:
            raise ValueError("It's not possible to withdraw more than the initial investment.")
        self._amount = amount

    @property
    def indexation(self) -> float:
        """
        Portfolio cash flow indexation rate.

        Returns
        -------
        float
            Cash flow indexation rate.
        """
        return self._indexation

    @indexation.setter
    def indexation(self, indexation: Optional[float]):
        if indexation in [None, "inflation"] and hasattr(self.portfolio, "inflation"):
            self._indexation = self.portfolio.get_cagr().loc[self.portfolio.inflation]
        elif indexation == "inflation" and not hasattr(self.portfolio, "inflation"):
            raise ValueError(
                "There is no information about historical inflation. Set inflation=True to calculate."
            )
        elif indexation is None and not hasattr(self.portfolio, "inflation"):
            self._indexation = settings.DEFAULT_DISCOUNT_RATE
        else:
            validators.validate_real("indexation", indexation)
            self._clear_cf_cache()
            self._indexation = indexation


class PercentageStrategy(CashFlow):
    """
    Cash flow strategy with regular fixed percentage withdrawals or contributions.

    Parameters
    ----------
    parent : Portfolio
        The parent Portfolio instance.
    frequency : str, optional
        Frequency of cash flows. Default is "none".
    initial_investment : float, optional
        Initial investment amount. Default is 1000.0.
    time_series_dic : dict, optional
        Dictionary with dates and cash flow values. Default is empty dict.
    time_series_discounted_values : bool, optional
        If True, values in time_series_dic are considered as discounted (PV). Default is False.
    percentage : float, optional
        Percentage of portfolio balance to be withdrawn or contributed. Negative for withdrawals. Default is 0.0.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(first_date="2015-01", last_date="2024-10")  # create Portfolio with default parameters
    >>> pc = ok.PercentageStrategy(pf)  # create PercentageStrategy linked to the portfolio
    >>> pc.initial_investment = 10_000  # add initial investments size
    >>> pc.frequency = "year"  # set cash flow frequency
    >>> pc.percentage = -0.12  # set withdrawal percentage
    >>> # Assign the strategy to Portfolio
    >>> pf.dcf.cashflow_parameters = pc
    >>> # Plot wealth index with cash flow
    >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
    >>> plt.show()
    """

    NAME = "fixed_percentage"

    def __init__(
        self,
        parent: core.Portfolio,
        frequency: Optional[str] = "none",
        initial_investment: float = 1000.0,
        time_series_dic: dict = {},
        time_series_discounted_values: bool = False,
        percentage: float = 0.0,
    ):
        super().__init__(
            parent,
            frequency=frequency,
            initial_investment=initial_investment,
            time_series_dic=time_series_dic,
            time_series_discounted_values=time_series_discounted_values,
        )
        self.portfolio = self.parent
        self._percentage = percentage

    def __repr__(self):
        dic = {
            "Strategy name": self.NAME,
            "Portfolio symbol": self.parent.symbol,
            "Cash flow initial investment": self.initial_investment,
            "Cash flow frequency": self.frequency,
            "Cash flow strategy": self.NAME,
            "Cash flow percentage": self.percentage,
        }
        return repr(pd.Series(dic))

    @property
    def percentage(self) -> float:
        """
        The percentage of withdrawals or contributions.

        The size of withdrawals or contribution is defined as a percentage of portfolio balance per year.

        Returns
        -------
        float
            The percentage of withdrawals or contributions.
        """
        return self._percentage

    @percentage.setter
    def percentage(self, percentage):
        self._clear_cf_cache()
        validators.validate_real("percentage", percentage)
        if percentage < -1:
            raise ValueError(
                "Withdrawal Percentage must less or equal to the Initial investment (100%)."
            )
        self._percentage = percentage


class TimeSeriesStrategy(CashFlow):
    """
    Cash flow strategy with user-defined withdrawals and contributions.

    Withdrawals, contributions, as well as their dates, are defined in the dictionary.

    Parameters
    ----------
    parent : Portfolio
        The parent Portfolio instance.
    initial_investment : float, optional
        Initial investment amount. Default is 0.
    time_series_dic : dict, optional
        Dictionary with dates and cash flow values. Default is empty dict.
    time_series_discounted_values : bool, optional
        If True, values in time_series_dic are considered as discounted (PV). Default is False.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(first_date="2015-01", last_date="2024-10")  # create Portfolio with default parameters
    >>> # create simple dictionary with cash flow amounts and dates
    >>> d = {"2018-02": 2_000, "2024-03": -4_000}
    >>> ts = ok.TimeSeriesStrategy(pf)  # create TimeSeresStrategy linked to the portfolio
    >>> ts.time_series_dic = d  # use the dictionary to set cash flow
    >>> ts.initial_investment = 1_000  # add initial investments size (optional)
    >>> # Assign the strategy to Portfolio
    >>> pf.dcf.cashflow_parameters = ts
    >>> # Plot wealth index with cash flow
    >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
    >>> plt.show()
    """

    NAME = "time_series"

    def __init__(
        self,
        parent: core.Portfolio,
        initial_investment: float = 0,
        time_series_dic: dict = {},
        time_series_discounted_values: bool = False,
    ):
        super().__init__(
            parent,
            frequency="none",
            initial_investment=initial_investment,
            time_series_dic=time_series_dic,
            time_series_discounted_values=time_series_discounted_values,
        )
        self.portfolio = self.parent

    def __repr__(self):
        dic = {
            "Strategy name": self.NAME,
            "Portfolio symbol": self.portfolio.symbol,
            "Cash flow initial investment": self.initial_investment,
            "Cash flow strategy": self.NAME,
        }
        return repr(pd.Series(dic))


class VanguardDynamicSpending(PercentageStrategy):
    """
    Vanguard Dynamic Spending strategy.

    The withdrawal amount is calculated as a percentage of the portfolio balance.

    The `floor_ceiling` parameter limits the withdrawals relative to the
    previous year's cash flow. 

    The absolute withdrawal amount can be optionally limited with
    `min_max_annual_withdrawal`. If `adjust_min_max=True`, these bounds are indexed
    using `indexation`.

    Parameters
    ----------
    parent : Portfolio
        The parent Portfolio instance.
    initial_investment : float, optional
        Initial investment amount. Default is 1000.0.
    time_series_dic : dict, optional
        Dictionary with dates and cash flow values. Default is empty dict.
    time_series_discounted_values : bool, optional
        If True, values in time_series_dic are considered as discounted (PV). Default is False.
    percentage : float, optional
        Percentage of portfolio balance to be withdrawn. Negative value. Default is 0.0.
    min_max_annual_withdrawals : tuple[float, float], optional
        Optional absolute min/max annual withdrawal amounts (positive values). Default is None.
    adjust_min_max : bool, optional
        If True, min/max bounds are indexed using `indexation`. Default is True.
    floor_ceiling : tuple[float, float], optional
        Year-to-year withdrawal change limits relative to the previous year's withdrawal.
        Example: (-0.025, 0.05) means the next withdrawal cannot be more than 2.5% lower
        or 5% higher than the previous year's withdrawal. Default is None.
    adjust_floor_ceiling : bool, optional
        If True, the previous year's withdrawal amount is indexed before applying
        floor/ceiling limits. Default is False.
    indexation : str or float, optional
        Indexation rate. Default is None.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(
    ...     ["SPY.US", "BND.US"],
    ...     weights=[.6, .4],
    ...     first_date="2010-01",
    ...     last_date="2024-10",
    ...     ccy="USD",
    ...     inflation=True
    ... )
    >>> vds = ok.VanguardDynamicSpending(
    ...     parent=pf,
    ...     initial_investment=1_000_000,
    ...     percentage=-0.08,
    ...     floor_ceiling=(-.025, .05),
    ...     min_max_annual_withdrawals=(40_000, 100_000),
    ...     adjust_min_max=True,
    ...     indexation="inflation",
    ... )
    >>> pf.dcf.cashflow_parameters = vds
    >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
    >>> plt.show()
    """

    NAME = "VDS"

    def __init__(
        self,
        parent: core.Portfolio,
        initial_investment: float = 1000.0,
        time_series_dic: dict = {},
        time_series_discounted_values: bool = False,
        percentage: float = 0.0,
        min_max_annual_withdrawals: Optional[tuple[float, float]] = None,
        adjust_min_max: bool = True,
        floor_ceiling: Optional[tuple[float, float]] = None,
        adjust_floor_ceiling: bool = False,
        indexation: Optional[Union[str, float]] = None,
    ):
        """
        Initialize the VanguardDynamicSpending strategy.


        """
        super().__init__(
            parent=parent,
            frequency="year",
            initial_investment=initial_investment,
            time_series_dic=time_series_dic,
            time_series_discounted_values=time_series_discounted_values,
            percentage=percentage,
        )
        self.portfolio = self.parent
        self._min_max_annual_withdrawals = min_max_annual_withdrawals
        self._adjust_min_max = adjust_min_max
        self._floor_ceiling = floor_ceiling
        self.adjust_floor_ceiling = adjust_floor_ceiling
        self.indexation = indexation

    def __repr__(self):
        dic = {
            "Strategy name": self.NAME,
            "Portfolio symbol": self.parent.symbol,
            "Cash flow initial investment": self.initial_investment,
            "Cash flow frequency": self.frequency,
            "Cash flow strategy": self.NAME,
            "Cash flow percentage": self.percentage,  # negative
            "Minimum annual withdrawal": self.min_max_annual_withdrawals[0]
            if self.min_max_annual_withdrawals is not None
            else None,  # positive
            "Maximum annual withdrawal": self.min_max_annual_withdrawals[1]
            if self.min_max_annual_withdrawals is not None
            else None,  # positive
            "Max and Min withdrawals are indexed": str(self.adjust_min_max),
            "Floor": self.floor_ceiling[0]
            if self.floor_ceiling is not None
            else None,  # negative
            "Ceiling": self.floor_ceiling[1]
            if self.floor_ceiling is not None
            else None,  # positive
            "Floor and Ceiling are indexed": str(self.adjust_floor_ceiling),
            "Indexation": self.indexation,
        }
        return repr(pd.Series(dic))

    @property
    def percentage(self) -> float:
        """
        The percentage of withdrawals (no contributions are allowed in VDS strategy). The value must be negative.

        The size of withdrawals is defined as a percentage of portfolio balance per year.

        Returns
        -------
        float
            The percentage of withdrawals or contributions.
        """
        return self._percentage

    @percentage.setter
    def percentage(self, percentage):
        self._clear_cf_cache()
        validators.validate_real("percentage", percentage)
        if percentage < -1:
            raise ValueError(
                "Withdrawal Percentage must less or equal to the Initial investment (100%)."
            )
        if percentage > 0:
            raise ValueError(
                "Only withdrawals are allowed in VDS strategy. Percentage must be negative or zero."
            )
        self._percentage = percentage

    @property
    def frequency(self):
        """
        Frequency of cash flows. Always 'year' for VDS.
        """
        return "year"

    @frequency.setter
    def frequency(self, value):
        if value != "year":
            raise AttributeError("In VDS the 'frequency' can only be equal to a year.")
        else:
            CashFlow.frequency.fset(self, "year")

    @property
    def min_max_annual_withdrawals(self):
        """
        Minimum and maximum annual withdrawal limits.
        """
        return self._min_max_annual_withdrawals

    @min_max_annual_withdrawals.setter
    def min_max_annual_withdrawals(self, value: Optional[tuple[float, float]]):
        if not isinstance(value, tuple):
            raise TypeError(
                "min_max_annual_withdrawals must be a tuple (float, float)."
            )
        min_w = value[0]
        max_w = value[1]
        validators.validate_real("minimum annual withdrawal", min_w)
        validators.validate_real("maximum annual withdrawal", max_w)
        if min_w < 0:
            raise ValueError("Min withdrawal cannot be negative.")
        if max_w < 0:
            raise ValueError("Max withdrawal cannot be negative.")
        if min_w > max_w:
            raise ValueError(
                "Minimum withdrawal cannot be greater than maximum withdrawal."
            )
        self._clear_cf_cache()
        self._min_max_annual_withdrawals = value

    @property
    def adjust_min_max(self):
        """
        Whether to adjust min and max limits by indexation.
        """
        return self._adjust_min_max

    @adjust_min_max.setter
    def adjust_min_max(self, value):
        if not isinstance(value, bool):
            raise TypeError("adjust_min_max must be a True or False.")
        self._clear_cf_cache()
        self._adjust_min_max = value

    @property
    def floor_ceiling(self):
        """
        Floor and ceiling percentages relative to the previous year's withdrawal.
        """
        return self._floor_ceiling

    @floor_ceiling.setter
    def floor_ceiling(self, value: Optional[tuple[float, float]]):
        if not isinstance(value, tuple):
            raise TypeError("floor_ceiling must be a tuple (float, float).")
        floor = value[0]
        ceiling = value[1]
        validators.validate_real("floor", floor)
        validators.validate_real("ceiling", ceiling)
        if floor >= 0:
            raise ValueError("Floor must be negative.")
        if ceiling <= 0:
            raise ValueError("Ceiling must be positive.")
        self._clear_cf_cache()
        self._floor_ceiling = value

    @property
    def adjust_floor_ceiling(self):
        """
        Whether to adjust floor and ceiling by indexation.
        """
        return self._adjust_floor_ceiling

    @adjust_floor_ceiling.setter
    def adjust_floor_ceiling(self, value):
        if not isinstance(value, bool):
            raise TypeError("adjust_min_max must be a True or False.")
        self._clear_cf_cache()
        self._adjust_floor_ceiling = value

    @property
    def indexation(self) -> float:
        """
        Indexation rate for Minimum/Maximum annual withdrawal and Floor/Ceiling.

        Returns
        -------
        float
            Indexation rate.
        """
        return self._indexation

    @indexation.setter
    def indexation(self, indexation: Optional[float]):
        if indexation in [None, "inflation"] and hasattr(self.portfolio, "inflation"):
            self._indexation = self.portfolio.get_cagr().loc[self.portfolio.inflation]
        elif indexation == "inflation" and not hasattr(self.portfolio, "inflation"):
            raise ValueError(
                "There is no information about historical inflation. Set inflation=True to calculate."
            )
        elif indexation is None and not hasattr(self.portfolio, "inflation"):
            self._indexation = settings.DEFAULT_DISCOUNT_RATE
        else:
            validators.validate_real("indexation", indexation)
            self._clear_cf_cache()
            self._indexation = indexation

    def _calculate_withdrawal_size(
        self, last_withdrawal: float, balance: float, number_of_periods: int
    ) -> float:
        """
        Calculate regular withdrawal size (Extra Withdrawals are not taken into account). Used in helpers.

        Parameters
        ----------
        last_withdrawal : float
            Withdrawal amount from the previous period.
        balance : float
            Current portfolio balance.
        number_of_periods : int
            Number of periods elapsed.

        Returns
        -------
        float
            Calculated withdrawal size.
        """
        # All values are positive
        withdrawal_size_by_percentage = balance * abs(self.percentage)
        if self.floor_ceiling is not None:
            floor, ceiling = self.floor_ceiling
            floor_indexed = (
                abs(last_withdrawal) * (1 + self.indexation) * (1 + floor)
                if self.adjust_floor_ceiling
                else abs(last_withdrawal) * (1 + floor)
            )
            ceiling_indexed = (
                abs(last_withdrawal) * (1 + self.indexation) * (1 + ceiling)
                if self.adjust_floor_ceiling
                else abs(last_withdrawal) * (1 + ceiling)
            )
        if self.min_max_annual_withdrawals is not None:
            min_withdrawal, max_withdrawal = self.min_max_annual_withdrawals
            min_indexed = (
                abs(min_withdrawal) * (1 + self.indexation) ** number_of_periods
                if self.adjust_min_max
                else abs(min_withdrawal)
            )
            max_indexed = (
                abs(max_withdrawal) * (1 + self.indexation) ** number_of_periods
                if self.adjust_min_max
                else abs(max_withdrawal)
            )
        # Chek what limitation is actual
        if (
            self.floor_ceiling is not None
            and self.min_max_annual_withdrawals is not None
        ):
            # Upper limit
            if ceiling_indexed > max_indexed:
                max_final = max_indexed
            elif min_indexed < ceiling_indexed <= max_indexed:
                max_final = ceiling_indexed
            else:
                # floor_indexed & ceiling_indexed = 0 for the first withdrawal (last_withdrawal = 0)
                max_final = max_indexed
            # Lower limit
            if floor_indexed > min_indexed:
                min_final = floor_indexed
            elif 0 < floor_indexed <= min_indexed:
                min_final = min_indexed
            else:
                # floor_indexed & ceiling_indexed = 0 for the first withdrawal (last_withdrawal = 0)
                min_final = min_indexed
        elif self.floor_ceiling is None and self.min_max_annual_withdrawals is not None:
            # print("ceiling is None")
            min_final = min_indexed
            max_final = max_indexed
        elif self.floor_ceiling is not None and self.min_max_annual_withdrawals is None:
            # print("min_max_annual_withdrawals is None")
            min_final = floor_indexed
            max_final = (
                ceiling_indexed
                if ceiling_indexed != 0
                else withdrawal_size_by_percentage
            )
        else:
            # no limits
            min_final = -math.inf
            max_final = math.inf
        # Apply the limitation to the withdrawal
        if min_final <= withdrawal_size_by_percentage <= max_final:
            withdrawal = -withdrawal_size_by_percentage
            # print(f"withdrawal by percentage. Max: {max_final: .0f}, Min: {min_final: .0f}")
        elif withdrawal_size_by_percentage > max_final:
            withdrawal = -max_final
            # print(f"withdrawal by max_final. By percentage was {withdrawal_size_by_percentage: .0f}")
        elif withdrawal_size_by_percentage < min_final:
            withdrawal = -min_final
            # print(f"withdrawal by min_final. By percentage was {withdrawal_size_by_percentage: .0f}")
        else:
            raise ValueError("Wrong withdrawal size. Check the calculation.")
        return withdrawal


class CutWithdrawalsIfDrawdown(IndexationStrategy):
    """
    Withdrawal strategy that reduces the withdrawal amount if the portfolio drawdown exceeds a certain threshold.

    The reduction coefficients are defined in `crash_threshold_reduction` list of tuples.

    Parameters
    ----------
    parent : Portfolio
        The parent Portfolio instance.
    frequency : str, optional
        Frequency of cash flows. Default is "year".
    initial_investment : float, optional
        Initial investment amount. Default is 1000.0.
    time_series_dic : dict, optional
        Dictionary with dates and cash flow values. Default is empty dict.
    time_series_discounted_values : bool, optional
        If True, values in time_series_dic are considered as discounted (PV). Default is False.
    amount : float, optional
        Regular withdrawal amount (negative value) before reduction. Default is 0.0.
        The frequency of withdrawals is determined by the `frequency` parameter.
    indexation : str or float, optional
        Indexation rate for the withdrawal amount. Default is None.
    crash_threshold_reduction : list[tuple[float, float]], optional
        List of tuples (threshold, reduction_coefficient).
        Example: [(0.20, 0.40)] means if drawdown > 20%, reduce withdrawal by 40%.
        Default is [(.20, .40), (.50, 1)].

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(
    ...     ["SPY.US", "BND.US"],
    ...     weights=[.6, .4],
    ...     first_date="2010-01",
    ...     last_date="2024-10",
    ...     ccy="USD",
    ...     inflation=True,
    ... )
    >>> cwid = ok.CutWithdrawalsIfDrawdown(
    ...     parent=pf,
    ...     initial_investment=1_000_000,
    ...     frequency="year",
    ...     amount=-60_000,
    ...     indexation="inflation",
    ...     crash_threshold_reduction=[(.10, .25), (.20, .50), (.35, 1)],
    ... )
    >>> pf.dcf.cashflow_parameters = cwid
    >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
    >>> plt.show()
    """

    NAME = "CWD"

    def __init__(
        self,
        parent: core.Portfolio,
        frequency: Optional[str] = "year",
        initial_investment: float = 1000.0,
        time_series_dic: dict = {},
        time_series_discounted_values: bool = False,
        amount: float = 0.0,
        indexation: Optional[Union[str, float]] = None,
        crash_threshold_reduction: list[tuple[float, float]] = [
            (0.20, 0.40),
            (0.50, 1),
        ],
    ):
        super().__init__(
            parent=parent,
            frequency=frequency,
            initial_investment=initial_investment,
            time_series_dic=time_series_dic,
            time_series_discounted_values=time_series_discounted_values,
            amount=amount,
            indexation=indexation,
        )
        self._crash_threshold_reduction_series = None
        self.portfolio = self.parent
        self._crash_threshold_reduction = crash_threshold_reduction
        self._crash_threshold_reduction_series = self.make_series_from_list(
            self.crash_threshold_reduction
        )

    def __repr__(self):
        dic = {
            "Strategy name": self.NAME,
            "Portfolio symbol": self.parent.symbol,
            "Cash flow initial investment": self.initial_investment,
            "Cash flow frequency": self.frequency,
            "Cash flow strategy": self.NAME,
            "Cash flow amount": self.amount,
            "Cash flow indexation": self.indexation,
            "Crash threshold reduction": self.crash_threshold_reduction,
        }
        return repr(pd.Series(dic))

    @property
    def amount(self):
        """
        Portfolio regular withdrawals size. Must be negative.
        Cash flow value is indexed each period by 'indexation'.

        The frequency of withdrawals or contributions is determined by the `frequency` parameter.

        Returns
        -------
        float
            Portfolio regular cash flow size.
        """
        return self._amount

    @amount.setter
    def amount(self, amount):
        self._clear_cf_cache()
        validators.validate_real("amount", amount)
        if abs(amount) > self.initial_investment:
            raise ValueError("It's not possible to withdraw more than the initial investment.")
        if amount > 0:
            raise ValueError("Amount must be negative or zero for withdrawals in CWD strategy.")
        self._amount = amount

    @property
    def crash_threshold_reduction(self):
        """
        List of tuples (threshold, reduction_coefficient) that define how much withdrawals are reduced
        based on the portfolio drawdown depth.

        Example
        -------
        >>> pf = ok.Portfolio(["SPY.US", "BND.US"], weights=[0.6, 0.4], ccy="USD")
        >>> strategy = ok.CutWithdrawalsIfDrawdown(parent=pf)
        >>> strategy.crash_threshold_reduction = [(0.05, 0.20), (0.10, 0.40), (0.20, 0.50), (0.30, 1.0)]
        >>> strategy.crash_threshold_reduction
        [(0.05, 0.2), (0.1, 0.4), (0.2, 0.5), (0.3, 1.0)]

        If drawdown is 5%, reduce withdrawal by 20%; if 10%, reduce by 40%; if 20%, reduce by 50%.
        If drawdown is 30% or more, withdrawals stop entirely (100% reduction).
        """
        return self._crash_threshold_reduction

    @crash_threshold_reduction.setter
    def crash_threshold_reduction(self, value):
        self._clear_cf_cache()
        self._crash_threshold_reduction_series = self.make_series_from_list(value)
        for threshold, reduction in self._crash_threshold_reduction_series.items():
            validators.validate_real("threshold", threshold)
            validators.validate_real("reduction", reduction)
            if abs(threshold) >= 1 or threshold == 0:
                raise ValueError(
                    "crash_threshold_reduction first values (threshold) must be in the interval (0, 1)."
                )
            if abs(reduction) > 1:
                raise ValueError(
                    "crash_threshold_reduction second values (reductiuon) must be in the interval [0, 1]."
                )
        self._crash_threshold_reduction = value

    def _calculate_withdrawal_size(
        self, drawdown: float, withdrawal_without_drawdowns: float
    ) -> float:
        """
        Calculate regular withdrawal size (Extra Withdrawals are not taken into account). Used in helpers.

        Parameters
        ----------
        drawdown : float
            Current portfolio drawdown (negative value).
        withdrawal_without_drawdowns : float
            Withdrawal amount calculated without considering drawdowns.

        Returns
        -------
        float
            Calculated withdrawal size.
        """
        withdrawal = abs(withdrawal_without_drawdowns)
        for threshold, reduction in self._crash_threshold_reduction_series.items():
            if abs(drawdown) >= threshold:
                withdrawal *= 1 - reduction
                break
        return -withdrawal

    def make_series_from_list(self, l: list[tuple[float, float]]) -> pd.Series:
        """
        Convert list of tuples to pandas Series.

        Parameters
        ----------
        l : list[tuple[float, float]]
            List of tuples.

        Returns
        -------
        pd.Series
            Series with threshold as index and reduction as value.
        """
        indices = [abs(index) for index, _ in l]
        values = [abs(value) for _, value in l]
        crash_series = pd.Series(values, index=indices)
        return crash_series.sort_index(ascending=False)

