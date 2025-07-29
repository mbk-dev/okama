from __future__ import annotations

import pandas as pd

from okama.common import validators
from okama.portfolios import dcf as dcf


class MonteCarlo:
    """
    Monte Carlo simulation parameters for investment portfolio.

    Parameters
    ----------
    parent : PortfolioDCF
        Parent PortfolioDCF instance.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(first_date='2015-01', last_date='2024-10')  # create Portfolio with default parameters
    >>> # Set Monte Carlo parameters
    >>> pf.dcf.set_mc_parameters(
    ... distribution='t',
    ... period=10,
    ... number=100
    ... )
    >>> # Set the cash flow strategy. It's required to generate random wealth indexes.
    >>> ind = ok.IndexationStrategy(pf) # create IndexationStrategy linked to the portfolio
    >>> ind.initial_investment = 10_000  # add initial investments size
    >>> ind.frequency = 'year'  # set cash flow frequency
    >>> ind.amount = -1_500  # set withdrawal size
    >>> ind.indexation = 'inflation'
    >>> # Assign the strategy to Portfolio
    >>> pf.dcf.cashflow_parameters = ind
    >>> pf.dcf.use_discounted_values = False  # do not discount initial investment value
    >>> # Plot wealth index with cash flow
    >>> pf.dcf.wealth_index_fv.plot()
    >>> plt.show()
    """

    def __init__(self, parent: dcf.PortfolioDCF):
        self.parent = parent
        self._distribution: str = "norm"
        self._period: int = 25
        self._mc_number: int = 100

    def __repr__(self):
        dic = {
            "Portfolio symbol": self.parent.parent.symbol,
            "Monte Carlo distribution": self.distribution,
            "Monte Carlo period": self.period,
            "Monte Carlo number": self.number,
        }
        return repr(pd.Series(dic))

    # TODO: add distribution parameters (return, risk etc)

    @property
    def distribution(self) -> str:
        """
        The type of a distribution to generate random rate of return.

        Allowed values for distribution:
        -'norm' for normal distribution
        -'lognorm' for lognormal distribution
        -'t' for Student's (t-distribution)

        Returns
        -------
        str
        """
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        validators.validate_distribution(distribution)
        self._clear_cf_cache()
        self._distribution = distribution

    @property
    def period(self) -> int:
        """
        Forecast period in years for portfolio wealth index time series.

        Returns
        -------
        int
        """
        return self._period

    @period.setter
    def period(self, period):
        validators.validate_integer("period", period)
        self._clear_cf_cache()
        self._period = period

    @property
    def number(self) -> int:
        """
        Number of random wealth indexes to generate with Monte Carlo simulation.

        Returns
        -------
        int
        """
        return self._mc_number

    @number.setter
    def number(self, mc_number):
        validators.validate_integer("mc_number", mc_number)
        self._clear_cf_cache()
        self._mc_number = mc_number

    def _clear_cf_cache(self):
        self.parent._monte_carlo_wealth_fv = pd.DataFrame()
