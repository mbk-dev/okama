from __future__ import annotations  # noqa: I001

from collections.abc import Sequence

import pandas as pd

from okama import settings
from okama.common import validators
from okama.common.helpers import helpers
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


class FinPlan:
    """A financial plan: an ordered sequence of portfolio stages.

    Each stage holds its own portfolio under its own cash flow strategy for a
    fixed number of years. The terminal balance of one stage opens the next, per
    Monte Carlo scenario, which is what makes the plan more than a list of
    independent forecasts: a retirement is funded by whatever the accumulation
    produced, scenario by scenario.

    Parameters
    ----------
    stages : sequence of FinPlanStage
        Stages in chronological order. At least one is required and all stage
        portfolios must share a currency.
    initial_investment : float, default 1000.0
        Balance at the start of the plan. This is the only source of the opening
        balance: the `initial_investment` of the stages' cash flow strategies is
        ignored, because `CashFlow` always assigns a default and "unset" cannot
        be told apart from "deliberately set".
    discount_rate : float or None, default None
        Annual effective discount rate for the whole horizon. If None and the
        first stage's portfolio has inflation data, its inflation CAGR is used;
        otherwise `settings.DEFAULT_DISCOUNT_RATE`.
    mc_number : int, default 100
        Number of Monte Carlo scenarios, shared by every stage.
    seed : int or None, default None
        Plan-level seed. Each stage draws from its own stream spawned from it,
        so the whole plan is reproducible. Note that a one-stage plan is
        therefore *not* bit-identical with `Portfolio.dcf.monte_carlo_wealth()`
        at the same seed, only statistically equivalent.
    name : str, default 'plan'
        Label for the wealth column, `__repr__` and plots.

    Notes
    -----
    `indexation="inflation"` on a stage's cash flow strategy resolves to the
    inflation CAGR of *that stage's own portfolio history*, not of the plan.
    Stages built from assets with different history windows therefore index at
    different rates. To index the whole plan at one rate, pass an explicit float
    to every stage strategy.

    Examples
    --------
    >>> acc = ok.Portfolio(["SPY.US", "AGG.US"], weights=[0.8, 0.2], ccy="USD")
    >>> ret = ok.Portfolio(["AGG.US", "SPY.US"], weights=[0.7, 0.3], ccy="USD")
    >>> pension = ok.IndexationStrategy(ret)
    >>> pension.frequency = "month"
    >>> pension.amount = -3_000
    >>> pension.indexation = "inflation"
    >>> plan = ok.FinPlan(
    ...     stages=[
    ...         ok.FinPlanStage(acc, period=20, name="accumulation"),
    ...         ok.FinPlanStage(ret, period=30, cashflow_parameters=pension, name="retirement"),
    ...     ],
    ...     initial_investment=100_000,
    ...     mc_number=1_000,
    ...     seed=0,
    ... )
    >>> plan.probability_of_success()
    """

    def __init__(
        self,
        stages: Sequence[FinPlanStage],
        initial_investment: float = settings.DEFAULT_INITIAL_INVESTMENT,
        discount_rate: float | None = None,
        mc_number: int = 100,
        seed: int | None = None,
        name: str = "plan",
    ):
        stages = tuple(stages)
        if not stages:
            raise ValueError("A financial plan needs at least one stage.")
        for stage in stages:
            if not isinstance(stage, FinPlanStage):
                raise TypeError("Every element of 'stages' must be a FinPlanStage instance.")
        currencies = {stage.portfolio.currency for stage in stages}
        if len(currencies) > 1:
            raise ValueError(
                f"All stage portfolios must share one currency, got {sorted(currencies)}. "
                "Balances of different currencies cannot be chained."
            )
        self._stages = stages
        self.name = name
        self._mc_wealth_fv: pd.DataFrame | None = None
        self._mc_cash_flow_fv: pd.DataFrame | None = None
        self.initial_investment = initial_investment
        self.discount_rate = discount_rate
        self.mc_number = mc_number
        self.seed = seed

    def __repr__(self) -> str:
        stages = ", ".join(f"{s.name or s.portfolio.symbol}:{s.period}y" for s in self.stages)
        dic = {
            "Plan name": self.name,
            "Stages": stages,
            "Horizon (years)": self.period,
            "Initial investment": self.initial_investment,
            "Monte Carlo number": self.mc_number,
            "Discount rate": self.discount_rate,
        }
        return repr(pd.Series(dic))

    @property
    def stages(self) -> tuple[FinPlanStage, ...]:
        """Stages of the plan in chronological order."""
        return self._stages

    @property
    def base_portfolio(self) -> core.Portfolio:
        """Portfolio of the first stage; it anchors the plan clock and the discount rate."""
        return self._stages[0].portfolio

    @property
    def t0(self) -> pd.Timestamp:
        """Start of the forecast: the last date of the first stage's portfolio."""
        return self.base_portfolio.last_date

    @property
    def period(self) -> int:
        """Plan horizon in years."""
        return sum(stage.period for stage in self._stages)

    @property
    def period_months(self) -> int:
        """Plan horizon in months."""
        return sum(stage.period_months for stage in self._stages)

    @property
    def initial_investment(self) -> float:
        """Balance at the start of the plan."""
        return self._initial_investment

    @initial_investment.setter
    def initial_investment(self, initial_investment: float):
        validators.validate_real("initial_investment", initial_investment)
        if initial_investment <= 0:
            raise ValueError("initial_investment must be positive.")
        self.clear_cache()
        self._initial_investment = float(initial_investment)

    @property
    def discount_rate(self) -> float:
        """Annual effective discount rate used for the whole horizon."""
        return self._discount_rate

    @discount_rate.setter
    def discount_rate(self, discount_rate: float | None):
        self.clear_cache()
        if discount_rate is None and hasattr(self.base_portfolio, "inflation"):
            rate = helpers.Frame.get_cagr(self.base_portfolio.inflation_ts)
            self._discount_rate = settings.DEFAULT_DISCOUNT_RATE if rate is None else rate
        elif discount_rate is None:
            self._discount_rate = settings.DEFAULT_DISCOUNT_RATE
        else:
            validators.validate_real("discount rate", discount_rate)
            self._discount_rate = discount_rate

    @property
    def mc_number(self) -> int:
        """Number of Monte Carlo scenarios."""
        return self._mc_number

    @mc_number.setter
    def mc_number(self, mc_number: int):
        validators.validate_integer("mc_number", mc_number)
        if mc_number < 1:
            raise ValueError("mc_number must be at least 1.")
        self.clear_cache()
        self._mc_number = mc_number

    @property
    def seed(self) -> int | None:
        """Plan-level seed; per-stage streams are spawned from it."""
        return self._seed

    @seed.setter
    def seed(self, seed: int | None):
        if seed is not None:
            validators.validate_integer("seed", seed)
        self.clear_cache()
        self._seed = seed

    def clear_cache(self) -> None:
        """Discard cached Monte Carlo results.

        Plan-level setters call this on their own. Call it by hand after editing
        a stage's cash flow strategy in place (`pension.amount = -4000`), which
        the plan cannot intercept.
        """
        self._mc_wealth_fv = None
        self._mc_cash_flow_fv = None
