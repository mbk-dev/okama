from __future__ import annotations

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

    def __init__(self, parent: core.Portfolio):
        self.parent = parent
        self.frequency: Optional[str] = "none"
        self.initial_investment: float = 1000.0
        self._pandas_frequency = settings.frequency_mapping.get(self.frequency)
        self.time_series_dic = {}
        self.time_series = pd.Series(dtype=float)
        self.time_series_discounted_values = False

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
            raise ValueError(f"frequency must be in {settings.frequency_mapping.keys()}")

    @property
    def periods_per_year(self) -> int:
        """
        Show the number of periods per year. Period is defined by the frequency.
        """
        return settings.frequency_periods_per_year[self.frequency]

    @property
    def initial_investment(self) -> float:
        """
        Portfolio initial investment FV size (at last_date).

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
        self.parent.dcf._monte_carlo_wealth_fv = pd.DataFrame(dtype=float)
        self.parent.dcf._wealth_index_fv = pd.DataFrame(dtype=float)
        self.parent.dcf._cash_flow_fv = pd.DataFrame(dtype=float)
        self.parent.dcf._monte_carlo_cash_flow_fv = pd.DataFrame(dtype=float)


class IndexationStrategy(CashFlow):
    """
    Cash flow strategy with regualr indexed withdrawals or contributions.

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
    >>> pf.dcf.use_discounted_values = False  # do not discount initial investment value
    >>> # Plot wealth index with cash flow
    >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
    >>> plt.show()
    """

    NAME = "fixed_amount"

    def __init__(
        self,
        parent: core.Portfolio,
    ):
        super().__init__(parent)
        self.portfolio = self.parent
        self.amount: float = 0
        self.indexation: Optional[Union[str, float]] = None

    def __repr__(self):
        dic = {
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
        Portfolio regular contributions or withdrawals size. Negative value corresponds to withdrawals.
        Positive value corresponds to contributions. Cash flow value is indexed each period by 'indexation'.

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
        if amount > self.initial_investment:
            raise ValueError("Amount must be less or equal to the initial investment.")
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
            raise ValueError("There is no information about historical inflation. Set inflation=True to calculate.")
        elif indexation is None and not hasattr(self.portfolio, "inflation"):
            self._indexation = settings.DEFAULT_DISCOUNT_RATE
        else:
            validators.validate_real("indexation", indexation)
            self._indexation = indexation


class PercentageStrategy(CashFlow):
    """
    Cash flow strategy with regular fixed percentage withdrawals or contributions.

    Parameters
    ----------
    parent : Portfolio
        Parent Portfolio instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(first_date="2015-01", last_date="2024-10")  # create Portfolio with default parameters
    >>> pc = ok.PercentageStrategy(portf)  # create PercentageStrategy linked to the portfolio
    >>> pc.initial_investment = 10_000  # add initial investments size
    >>> pc.frequency = "year"  # set cash flow frequency
    >>> pc.percentage = -0.12  # set withdrawal percentage
    >>> # Assign the strategy to Portfolio
    >>> pf.dcf.cashflow_parameters = pc
    >>> pf.dcf.use_discounted_values = False  # do not discount initial investment value
    >>> # Plot wealth index with cash flow
    >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
    >>> plt.show()
    """

    NAME = "fixed_percentage"

    def __init__(
        self,
        parent: core.Portfolio,
    ):
        super().__init__(parent)
        self.portfolio = self.parent
        self.percentage = 0

    def __repr__(self):
        dic = {
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
            raise ValueError("Withdrawal Percentage must less or equal to the Initial investment (100%).")
        self._percentage = percentage


class TimeSeriesStrategy(CashFlow):
    """
    Cash flow strategy with user-defined withdrawals and contributions.

    Withdrawals, contributions, as well as their dates, are defined in the dictionary.

    Parameters
    ----------
    parent : Portfolio
        Parent Portfolio instance.

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
    ):
        super().__init__(parent)
        self.portfolio = self.parent

    def __repr__(self):
        dic = {
            "Portfolio symbol": self.parent.symbol,
            "Cash flow initial investment": self.initial_investment,
            "Cash flow strategy": self.NAME,
        }
        return repr(pd.Series(dic))
