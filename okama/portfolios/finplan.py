from __future__ import annotations  # noqa: I001

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from okama import settings
from okama.common import validators
from okama.common.helpers import helpers
from okama.portfolios import cashflow_strategies as cf
from okama.portfolios import core
from okama.portfolios import dcf_calculations
from okama.portfolios import mc as mc_module

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
        if distribution_parameters is not None:
            validators.validate_distribution_parameters(distribution, distribution_parameters)
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

    def _run_monte_carlo(self) -> None:
        """Simulate every stage in order, handing the terminal balance forward.

        One `_simulate_paths_mc` pass per stage yields both the wealth index and
        the cash flow, so the engine runs once rather than twice.
        """
        seeds = np.random.SeedSequence(self.seed).spawn(len(self.stages))
        t0_period = self.t0.to_period("M")
        balance = np.full(self.mc_number, float(self.initial_investment))
        month_offset = 0
        wealth_parts: list[pd.DataFrame] = []
        cash_flow_parts: list[pd.DataFrame] = []
        for stage, seed_sequence in zip(self.stages, seeds, strict=True):
            index = pd.period_range(t0_period + month_offset, periods=stage.period_months, freq="M")
            ror = mc_module.generate_returns_ts(
                ror=stage.portfolio.ror,
                distribution=stage.distribution,
                distribution_parameters=stage.distribution_parameters,
                n_paths=self.mc_number,
                index=index,
                rng=np.random.default_rng(seed_sequence),
            )
            wealth, cash_flow = dcf_calculations._simulate_paths_mc(
                ror,
                stage.cashflow_parameters,
                self.discount_rate,
                initial_balance=balance,
                month_offset=month_offset,
            )
            wealth_parts.append(pd.DataFrame(wealth, index=index))
            cash_flow_parts.append(pd.DataFrame(cash_flow, index=index))
            # A depleted scenario opens the next stage at zero rather than at a
            # negative balance, which has no financial meaning.
            balance = np.maximum(wealth[-1], 0.0)
            month_offset += stage.period_months
        wealth_df = pd.concat(wealth_parts)
        opening = pd.DataFrame(
            np.full((1, self.mc_number), float(self.initial_investment)),
            index=wealth_df.index[:1] - 1,
        )
        self._mc_wealth_fv = pd.concat([opening, wealth_df])
        self._mc_cash_flow_fv = pd.concat(cash_flow_parts)

    def monte_carlo_wealth(
        self, discounting: Literal["fv", "pv"] = "fv", include_negative_values: bool = True
    ) -> pd.DataFrame:
        """Wealth index of the whole plan for every Monte Carlo scenario.

        Stages are simulated in order and concatenated, so the balance a
        scenario reaches at the end of one stage is the balance it starts the
        next one with (floored at zero).

        Parameters
        ----------
        discounting : {'fv', 'pv'}, default 'fv'
            'fv' returns nominal values; 'pv' discounts them to the plan start
            with `discount_rate`. The discounting is applied once over the whole
            horizon, so present values of different stages are comparable.
        include_negative_values : bool, default True
            If False, the first non-positive value of a scenario and everything
            after it become 0.

        Returns
        -------
        DataFrame
            `(period_months + 1, mc_number)`. The first row is the plan's
            opening balance, dated one month before `t0`.
        """
        if self._mc_wealth_fv is None:
            self._run_monte_carlo()
        wealth = (
            self._mc_wealth_fv.copy()
            if include_negative_values
            else dcf_calculations.zero_wealth_after_first_void(self._mc_wealth_fv)
        )
        return self._discount(wealth, discounting)

    def monte_carlo_cash_flow(
        self, discounting: Literal["fv", "pv"] = "fv", remove_if_wealth_index_negative: bool = True
    ) -> pd.DataFrame:
        """Cash flow of the whole plan for every Monte Carlo scenario.

        Parameters
        ----------
        discounting : {'fv', 'pv'}, default 'fv'
            As in `monte_carlo_wealth`.
        remove_if_wealth_index_negative : bool, default True
            If True, cash flow is zeroed for months in which the (floored)
            wealth index is zero.

        Returns
        -------
        DataFrame
            `(period_months, mc_number)`, starting at `t0`.
        """
        if self._mc_cash_flow_fv is None:
            self._run_monte_carlo()
        cash_flow = self._mc_cash_flow_fv.copy()
        if remove_if_wealth_index_negative:
            wealth = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
            cash_flow[wealth.reindex(cash_flow.index) == 0] = 0
        return self._discount(cash_flow, discounting)

    def monte_carlo_survival_period(self, threshold: float = 0) -> pd.Series:
        """How long each scenario keeps a positive balance, in years from `t0`.

        Parameters
        ----------
        threshold : float, default 0
            Share of the opening balance below which the plan counts as voided.
            Useful with `PercentageStrategy`, whose balance approaches zero
            asymptotically.

        Returns
        -------
        Series
            One survival period per Monte Carlo scenario.
        """
        wealth = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        dates = helpers.Frame.get_survival_date(wealth, self.discount_rate, threshold)
        return dates.apply(helpers.Date.get_period_length, args=(self.t0,))

    def monte_carlo_irr(self) -> pd.Series:
        """Money-weighted IRR of the whole plan for every scenario.

        The investor's cash flow runs from `t0` (minus the plan's initial
        investment) to the end of the horizon, where the terminal balance is
        added back.

        Returns
        -------
        Series
            One annualized effective IRR per scenario; NaN where the cash flow
            has no sign change.
        """
        wealth = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        cash_flow = self.monte_carlo_cash_flow(discounting="fv", remove_if_wealth_index_negative=False)
        # Zero a scenario's cash flow once its (floored) wealth is depleted.
        cash_flow = cash_flow.where(wealth.reindex(cash_flow.index) != 0, 0.0)
        terminal = wealth.iloc[-1]
        n_months, n_paths = cash_flow.shape
        flows = np.empty((n_months + 1, n_paths), dtype=float)
        flows[0, :] = -self.initial_investment
        flows[1:, :] = -cash_flow.to_numpy()
        flows[-1, :] += terminal.reindex(cash_flow.columns).to_numpy()
        irr = dcf_calculations.irr_of_cashflow_matrix(flows, periods_per_year=settings._MONTHS_PER_YEAR)
        return pd.Series(irr, index=cash_flow.columns, name="monte_carlo_irr")

    def probability_of_success(self, threshold: float = 0) -> float:
        """Share of scenarios that reach the end of the plan above `threshold`.

        For a retirement plan this is the headline number: the chance the money
        outlives the horizon.

        Returns
        -------
        float
            A value between 0 and 1.
        """
        wealth = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        return float((wealth.iloc[-1] > threshold).mean())

    def balance_percentiles(
        self,
        percentiles: tuple[int, ...] = (10, 50, 90),
        discounting: Literal["fv", "pv"] = "fv",
    ) -> pd.DataFrame:
        """Distribution of the balance at every stage boundary and at the end.

        The row for the accumulation stage answers the question a plan is built
        around: how much is there on the day the next stage has to live off it.

        Parameters
        ----------
        percentiles : tuple of int, default (10, 50, 90)
            Percentiles to report.
        discounting : {'fv', 'pv'}, default 'fv'
            As in `monte_carlo_wealth`.

        Returns
        -------
        DataFrame
            One row per stage, indexed by stage name; a `date` column followed by
            one column per percentile.
        """
        wealth = self.monte_carlo_wealth(discounting=discounting, include_negative_values=False)
        rows = []
        names = []
        month_offset = 0
        for number, stage in enumerate(self.stages, start=1):
            month_offset += stage.period_months
            # Row 0 holds the opening balance, so the stage's last month sits at
            # the cumulative month offset.
            balances = wealth.iloc[month_offset]
            row = {"date": wealth.index[month_offset]}
            row.update({f"{p}%": float(np.percentile(balances.to_numpy(), p)) for p in percentiles})
            rows.append(row)
            names.append(stage.name or f"stage {number}")
        return pd.DataFrame(rows, index=pd.Index(names, name="stage"))

    def _discount(self, values: pd.Series | pd.DataFrame, discounting: Literal["fv", "pv"]) -> pd.Series | pd.DataFrame:
        if discounting.lower() == "fv":
            return values
        if discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(values, self.discount_rate)
        raise ValueError("'discounting' must be either 'fv' or 'pv'")

    @property
    def history_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Earliest and latest dates covered by every stage portfolio at once."""
        first = max(stage.portfolio.first_date for stage in self.stages)
        last = min(stage.portfolio.last_date for stage in self.stages)
        return first, last

    def _run_backtest(self, first_date: str | pd.Timestamp | None) -> tuple[pd.Series, pd.Series]:
        """Chain the stages over actual history, returning (wealth, cash flow).

        The stages consume the common history window one after another: the
        first stage runs on its portfolio's real returns for its first
        `period_months` months, the second continues from the balance the first
        reached, and so on.
        """
        available_first, available_last = self.history_window
        start = available_first if first_date is None else pd.to_datetime(first_date)
        if start < available_first:
            raise ValueError(
                f"first_date {start:%Y-%m} precedes the common history of the stage portfolios, "
                f"which starts at {available_first:%Y-%m}."
            )
        required = self.period_months
        available = helpers.Date.get_difference_in_months(available_last, start).n + 1
        if available < required:
            raise ValueError(
                f"The plan needs {required} months of history, but only {available} months are "
                f"available for all stage portfolios from {start:%Y-%m} to {available_last:%Y-%m}."
            )
        start_period = start.to_period("M")
        balance = np.array([float(self.initial_investment)])
        month_offset = 0
        wealth_parts: list[pd.Series] = []
        cash_flow_parts: list[pd.Series] = []
        for stage in self.stages:
            index = pd.period_range(start_period + month_offset, periods=stage.period_months, freq="M")
            ror = stage.portfolio.ror.reindex(index).to_frame()
            if ror.isna().to_numpy().any():
                raise ValueError(
                    f"Stage '{stage.name or stage.portfolio.symbol}' has no returns for part of "
                    f"{index[0]}:{index[-1]}. Its portfolio history is "
                    f"{stage.portfolio.first_date:%Y-%m}:{stage.portfolio.last_date:%Y-%m}."
                )
            wealth, cash_flow = dcf_calculations._simulate_paths_mc(
                ror,
                stage.cashflow_parameters,
                self.discount_rate,
                initial_balance=balance,
                month_offset=month_offset,
                task="backtest",
            )
            wealth_parts.append(pd.Series(wealth[:, 0], index=index))
            cash_flow_parts.append(pd.Series(cash_flow[:, 0], index=index))
            balance = np.maximum(wealth[-1], 0.0)
            month_offset += stage.period_months
        wealth_s = pd.concat(wealth_parts)
        opening = pd.Series([float(self.initial_investment)], index=wealth_s.index[:1] - 1)
        return pd.concat([opening, wealth_s]), pd.concat(cash_flow_parts)

    def wealth_index(
        self,
        discounting: Literal["fv", "pv"] = "fv",
        include_negative_values: bool = False,
        first_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Backtest the plan over actual history.

        The stages divide the common history window sequentially: stage one runs
        on its portfolio's real returns for its own length, stage two continues
        from the balance stage one reached, and so on. This is a glide-path
        backtest, not a forecast — for the forecast use `monte_carlo_wealth`.

        Parameters
        ----------
        discounting : {'fv', 'pv'}, default 'fv'
            'fv' returns nominal values; 'pv' discounts to the window start.
        include_negative_values : bool, default False
            If False, the balance is zeroed from the first non-positive value on.
        first_date : str or Timestamp or None, default None
            Start of the window. By default the plan starts at the earliest date
            covered by every stage portfolio, which uses the longest available
            run.

        Returns
        -------
        DataFrame
            One column named after the plan, plus accumulated inflation when the
            first stage's portfolio carries inflation data.

        Raises
        ------
        ValueError
            If the common history is shorter than the plan's horizon.
        """
        wealth, _ = self._run_backtest(first_date)
        if not include_negative_values:
            wealth = dcf_calculations.remove_negative_values(wealth).fillna(0)
        frame = wealth.to_frame(name=self.name)
        base = self.base_portfolio
        if hasattr(base, "inflation"):
            inflation_ts = base.inflation_ts.reindex(wealth.index[1:])
            cumulative = helpers.Frame.get_wealth_indexes(
                ror=inflation_ts, initial_amount=float(self.initial_investment)
            )
            frame = pd.concat([frame, cumulative.rename(base.inflation)], axis="columns")
        return self._discount(frame, discounting)

    def cash_flow_ts(
        self,
        discounting: Literal["fv", "pv"] = "fv",
        remove_if_wealth_index_negative: bool = True,
        first_date: str | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Cash flow of the plan over actual history.

        Parameters
        ----------
        discounting : {'fv', 'pv'}, default 'fv'
            As in `wealth_index`.
        remove_if_wealth_index_negative : bool, default True
            If True, cash flow is zeroed for months in which the (floored)
            wealth index is zero. This matches `monte_carlo_cash_flow` behavior.
        first_date : str or Timestamp or None, default None
            As in `wealth_index`.

        Returns
        -------
        Series
            Monthly cash flow over the plan's historical window.
        """
        wealth, cash_flow = self._run_backtest(first_date)
        if remove_if_wealth_index_negative:
            # Floor at zero (matching wealth_index default)
            wealth_floored = dcf_calculations.remove_negative_values(wealth).fillna(0)
            cash_flow = cash_flow.where(wealth_floored.reindex(cash_flow.index) != 0, 0.0)
        cash_flow.name = "cash_flow"
        return self._discount(cash_flow, discounting)

    def plot_forecast_monte_carlo(self, figsize: tuple[float, float] | None = None) -> Axes:
        """Plot the plan's scenario cloud with the stage boundaries marked.

        Without the boundary markers a multi-stage chart gives no hint of where
        the portfolio and the cash flow regime change, which is the whole point
        of a plan.

        Parameters
        ----------
        figsize : tuple of (float, float) or None, default None
            Figure size in inches; matplotlib defaults are used when None.

        Returns
        -------
        Axes
            Matplotlib axes object.

        Notes
        -----
        Unlike `PortfolioDCF.plot_forecast_monte_carlo` this method has no
        `backtest` flag. A plan's history is a chained backtest of its own, with
        its own window; plot it with `plan.wealth_index().plot()`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> plan.plot_forecast_monte_carlo()
        >>> plt.show()
        """
        wealth = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        ax = wealth.plot(legend=None, figsize=figsize)
        month_offset = 0
        for number, stage in enumerate(self.stages, start=1):
            stage_start = month_offset
            month_offset += stage.period_months
            if number < len(self.stages):
                ax.axvline(wealth.index[month_offset + 1].ordinal, color="grey", linestyle="--", linewidth=1)
            middle = wealth.index[(stage_start + month_offset) // 2].ordinal
            ax.text(
                middle,
                ax.get_ylim()[1],
                stage.name or f"stage {number}",
                ha="center",
                va="top",
            )
        return ax
