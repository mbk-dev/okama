from __future__ import annotations  # noqa: I001

import pandas as pd

from okama import settings
from okama.common import validators
from okama.portfolios import cashflow_strategies as cf
from okama.portfolios import core

ALLOWED_DISTRIBUTIONS = ("norm", "lognorm", "t")


class FinPlanStage:
    """One stage of a financial plan: a portfolio held for a number of years.

    A stage owns everything that can differ from one leg of a plan to another —
    the portfolio, how long it is held, the cash flow strategy in force, and the
    distribution used to simulate its returns. Everything shared by the whole
    horizon (the initial investment, the discount rate, the number of scenarios
    and the seed) belongs to `FinPlan` instead.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio held during the stage.
    period : int
        Stage length in whole years.
    cashflow_parameters : CashFlow or None, default None
        Cash flow strategy in force during the stage. Its `parent` must be
        `portfolio`. When None an `IndexationStrategy` on `portfolio` is created,
        matching what `PortfolioDCF` does.
    name : str or None, default None
        Label used in `FinPlan.__repr__`, `FinPlan.balance_percentiles` and on
        plots.
    distribution : {'norm', 'lognorm', 't'}, default 'norm'
        Distribution used to draw the stage's monthly returns.
    distribution_parameters : tuple or None, default None
        Parameters for the distribution; any element left as None is estimated
        from the portfolio's own history. Same semantics as `MonteCarlo`.

    Examples
    --------
    >>> pf = ok.Portfolio(["SPY.US", "AGG.US"], weights=[0.8, 0.2], ccy="USD")
    >>> stage = ok.FinPlanStage(pf, period=20, name="accumulation")
    >>> stage.period_months
    240
    """

    def __init__(
        self,
        portfolio: core.Portfolio,
        period: int,
        cashflow_parameters: cf.CashFlow | None = None,
        name: str | None = None,
        distribution: str = "norm",
        distribution_parameters: tuple | None = None,
    ):
        if not isinstance(portfolio, core.Portfolio):
            raise TypeError("portfolio must be a Portfolio instance.")
        validators.validate_integer("period", period)
        if period < 1:
            raise ValueError("period must be at least 1 year.")
        if cashflow_parameters is None:
            cashflow_parameters = cf.IndexationStrategy(parent=portfolio)
        elif not isinstance(cashflow_parameters, cf.CashFlow):
            raise TypeError("cashflow_parameters must be a CashFlow instance or None.")
        elif cashflow_parameters.parent is not portfolio:
            raise ValueError(
                "cashflow_parameters must be built on the stage portfolio: its 'parent' is "
                f"'{cashflow_parameters.parent.symbol}' but the stage holds '{portfolio.symbol}'. "
                "Otherwise indexation='inflation' would resolve against another portfolio."
            )
        if distribution not in ALLOWED_DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {ALLOWED_DISTRIBUTIONS}, got '{distribution}'.")
        self._portfolio = portfolio
        self._period = period
        self._cashflow_parameters = cashflow_parameters
        self._name = name
        self._distribution = distribution
        self._distribution_parameters = distribution_parameters

    def __repr__(self) -> str:
        dic = {
            "Stage name": self.name,
            "Portfolio symbol": self.portfolio.symbol,
            "Period (years)": self.period,
            "Cash flow strategy": getattr(self.cashflow_parameters, "NAME", None),
            "Distribution": self.distribution,
        }
        return repr(pd.Series(dic))

    @property
    def portfolio(self) -> core.Portfolio:
        """Portfolio held during the stage."""
        return self._portfolio

    @property
    def period(self) -> int:
        """Stage length in whole years."""
        return self._period

    @property
    def period_months(self) -> int:
        """Stage length in months."""
        return self._period * settings._MONTHS_PER_YEAR

    @property
    def cashflow_parameters(self) -> cf.CashFlow:
        """Cash flow strategy in force during the stage."""
        return self._cashflow_parameters

    @property
    def name(self) -> str | None:
        """Stage label used in reports and on plots."""
        return self._name

    @property
    def distribution(self) -> str:
        """Distribution used to draw the stage's monthly returns."""
        return self._distribution

    @property
    def distribution_parameters(self) -> tuple | None:
        """Distribution parameters; None elements are estimated from history."""
        return self._distribution_parameters
