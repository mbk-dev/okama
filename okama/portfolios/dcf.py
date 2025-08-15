from __future__ import annotations

from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd

import okama.portfolios.core as core
import okama.portfolios.mc as mc
import okama.portfolios.cashflow_strategies as cf
import okama.portfolios.dcf_calculations as dcf_calculations
from okama import settings
from okama.common import validators
from okama.common.helpers import helpers
from okama.common.solver import Result


class PortfolioDCF:
    """
    Class to access discounted cash flow (DCF) methods of Portfolio.
    All methods can be used in Portfolio instances trough construction:
    ```
    pf = Portfolio()
    pf.dcf.weatlh_index
    pf.dсf.cashflow_pv
    ```

    Parameters
    ----------
    discount_rate: float or None, default None
        Cash flow discount rate required to calculate Present value (PV) or Future (FV) of cashflow.
        If not provided geometric mean of inflation is taken.
        For portfolios without inflation the default value from settings is used.
    """

    def __init__(
        self,
        parent: core.Portfolio,
        discount_rate: Optional[float] = None,
    ):
        self.parent = parent
        self.discount_rate = discount_rate
        self._wealth_index_fv = pd.DataFrame(dtype=float)
        self._monte_carlo_wealth_fv = pd.DataFrame(dtype=float)
        self._monte_carlo_cash_flow_fv = pd.DataFrame(dtype=float)
        self._cash_flow_fv = pd.Series(dtype=float, name="cash_flow_fv")
        self.mc = mc.MonteCarlo(self)
        self._cashflow_parameters: Optional[cf.CashFlow] = None

    def __repr__(self):
        dic = {
            "Portfolio symbol": self.parent.symbol,
            "Monte Carlo distribution": self.mc.distribution,
            "Monte Carlo period": self.mc.period,
            "Cash flow strategy": self.cashflow_parameters.NAME if hasattr(self.cashflow_parameters, "NAME") else None,
            "discount_rate": self.discount_rate,
        }
        return repr(pd.Series(dic))

    @property
    def discount_rate(self) -> float:
        """
        Annual effective discount rate for portfolio cash flow.

        Returns
        -------
        float
            Cash flow discount rate.
        """
        return float(self._discount_rate)

    @discount_rate.setter
    def discount_rate(self, discount_rate: Optional[float]):
        self._wealth_index_fv = pd.DataFrame()
        self._monte_carlo_wealth_fv = pd.DataFrame()
        if discount_rate is None and hasattr(self.parent, "inflation"):
            self._discount_rate = helpers.Frame.get_cagr(self.parent.inflation_ts)
        elif discount_rate is None and not hasattr(self.parent, "inflation"):
            self._discount_rate = settings.DEFAULT_DISCOUNT_RATE
        else:
            validators.validate_real("discount rate", discount_rate)
            self._discount_rate = discount_rate

    @property
    def cashflow_parameters(self) -> Optional[cf.CashFlow]:
        return self._cashflow_parameters

    @cashflow_parameters.setter
    def cashflow_parameters(self, cashflow_parameters):
        self.cashflow_parameters._clear_cf_cache()
        self._cashflow_parameters = cashflow_parameters

    def set_mc_parameters(self, distribution: str, period: int, number: int):
        """
        Add Monte Carlo simulation parameters to PortfolioDCF.

        Parameters
        ----------
        distribution: str
            The type of a distribution to generate random rate of return.
            Allowed values for distribution:
            -'norm' for normal distribution
            -'lognorm' for lognormal distribution
            -'t' for Student's (t-distribution)

        period: int
            Forecast period for portfolio wealth index time series (in years).

        number: int
            Number of random wealth indexes to generate with Monte Carlo simulation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(first_date="2015-01", last_date="2024-10")  # create Portfolio with default parameters
        >>> # Set Monte Carlo parameters
        >>> pf.dcf.set_mc_parameters(distribution="lognorm", period=10, number=100)
        >>> # Set the cash flow strategy. It's required to generate random wealth indexes.
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
        self.mc.distribution = distribution
        self.mc.period = period
        self.mc.number = number

    def wealth_index(self, discounting: Literal["fv", "pv"], include_negative_values: bool = False) -> pd.DataFrame:
        """
        Wealth index Future Values (FV) time series for the portfolio with cash flow (contributions and
        withdrawals).

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period considering cash flows.

        Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        If there is no cash flow, Wealth index is obtained from the accumulated return multiplicated
        by the initial investments. That is: initial_amount_pv * (Acc_Return + 1)

        Returns
        -------
            Time series of wealth index values for portfolio and accumulated inflation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['VOO.US', 'GLD.US'], weights=[0.8, 0.2])
        >>> ind = ok.IndexationStrategy(pf)  # Set Cash Flow Strategy parameters
        >>> ind.initial_investment = 100  # initial investments value
        >>> ind.frequency = "year"  # withdrawals frequency
        >>> ind.amount = -0.5 * 12  # initial withdrawals amount
        >>> ind.indexation = "inflation"  # the indexation is equal to inflation
        >>> pf.dcf.cashflow_parameters = ind  # assign the strategy to Portfolio
        >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._wealth_index_fv.empty:
            df = self.parent._add_inflation()
            infl_symbol = self.parent.inflation if hasattr(self.parent, "inflation") else None
            df = dcf_calculations.get_wealth_indexes_fv_with_cashflow(
                ror=df,
                portfolio_symbol=self.parent.symbol,
                inflation_symbol=infl_symbol,
                cashflow_parameters=self.cashflow_parameters,
                task="backtest",
            )
            self._wealth_index_fv = self.parent._make_df_if_series(df)
        if not include_negative_values:
            wealth_index_fv = self._wealth_index_fv.copy()
            wealth_index_fv_s = dcf_calculations.remove_negative_values(self._wealth_index_fv[self.parent.name])
            wealth_index_fv[self.parent.name] = wealth_index_fv_s.fillna(0)
        else:
            wealth_index_fv = self._wealth_index_fv.copy()
        if discounting.lower() == "fv":
            return wealth_index_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(wealth_index_fv, self.discount_rate)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")

    def cash_flow_ts(self, discounting: Literal["fv", "pv"], remove_if_wealth_index_negative: bool = True) -> pd.Series:
        """
        Wealth index Future Values (FV) time series for the portfolio with cash flow (contributions and
        withdrawals).

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period considering cash flows.

        Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        If there is no cash flow, Wealth index is obtained from the accumulated return multiplicated
        by the initial investments. That is: initial_amount_pv * (Acc_Return + 1)

        Returns
        -------
            Time series of wealth index values for portfolio and accumulated inflation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['VOO.US', 'GLD.US'], weights=[0.8, 0.2])
        >>> ind = ok.IndexationStrategy(pf)  # Set Cash Flow Strategy parameters
        >>> ind.initial_investment = 100  # initial investments value
        >>> ind.frequency = "year"  # withdrawals frequency
        >>> ind.amount = -0.5 * 12  # initial withdrawals amount
        >>> ind.indexation = "inflation"  # the indexation is equal to inflation
        >>> pf.dcf.cashflow_parameters = ind  # assign the strategy to Portfolio
        >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._cash_flow_fv.empty:
            df = self.parent.ror
            self._cash_flow_fv = dcf_calculations.get_cash_flow_fv(
                ror=df,
                portfolio_symbol=self.parent.symbol,
                cashflow_parameters=self.cashflow_parameters,
                task="backtest",
            )
        if remove_if_wealth_index_negative:
            cash_flow_fv = self._cash_flow_fv.copy()
            wealth_index = self.wealth_index(discounting="fv", include_negative_values=False)
            condition = wealth_index[self.parent.name] == 0
            cash_flow_fv[condition] = 0
        else:
            cash_flow_fv = self._cash_flow_fv.copy()
        if discounting.lower() == "fv":
            return cash_flow_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(cash_flow_fv, self.discount_rate)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")

    @property
    def wealth_index_fv_with_assets(self) -> pd.DataFrame:
        """
        Wealth index Future Values (FV) time series for the portfolio and all assets considering cash flow (contributions and
        withdrawals).

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period. Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        Wealth index is obtained from the accumulated return multiplicated by the initial investments.
        initial_amount_pv * (Acc_Return + 1)

        If there is no cash flow, Wealth index is obtained from the accumulated return multiplicated
        by the initial investments. That is: initial_amount_pv * (Acc_Return + 1)

        Returns
        -------
        DataFrame
            Time series of wealth index values for portfolio, each asset and accumulated inflation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['VOO.US', 'GLD.US'], weights=[0.8, 0.2])
        >>> ind = ok.IndexationStrategy(pf)  # Set Cash Flow Strategy parameters
        >>> ind.initial_investment = 100  # initial investments value
        >>> ind.frequency = "year"  # withdrawals frequency
        >>> ind.amount = -0.5 * 12  # initial withdrawals amount
        >>> ind.indexation = "inflation"  # the indexation is equal to inflation
        >>> pf.dcf.cashflow_parameters = ind  # assign the strategy to Portfolio
        >>> pf.dcf.wealth_index_fv_with_assets.plot()
        >>> plt.show()
        """
        ls = [self.parent.ror, self.parent.assets_ror]
        ror_df = pd.concat(ls, axis=1, join="inner", copy="false")
        wealth_df = ror_df.apply(
            dcf_calculations.get_wealth_indexes_fv_with_cashflow,
            axis=0,
            args=(None, None, self.cashflow_parameters, "backtest"),  # symbol  # inflation_symbol
        )
        if hasattr(self.parent, "inflation"):
            inflation_wi = helpers.Frame.get_wealth_indexes(
                ror = self.parent.inflation_ts,
                initial_amount=self.cashflow_parameters.initial_investment
            )
            wealth_df = pd.concat([wealth_df, inflation_wi], axis=1, join="inner", copy="false")
        return wealth_df

    def survival_period_hist(self, threshold: float = 0) -> float:
        """
        Calculate the period when the portfolio has positive balance considering withdrawals on the historical data.

        The portfolio survival period (longevity period) depends on the investment strategy: asset allocation,
        rebalancing, withdrawals rate etc.

        Parameters
        ----------
        threshold : float, default 0
            The percentage of the initial investments when the portfolio balance considered voided.
            This parameter is important to use in cash flow strategies with a fixed
            whtdrawal percentage (PercentageStrategy).

        Returns
        -------
        float
            The portfolio survival period (longevity period) in years.

        Examples
        --------
        >>> pf = ok.Portfolio(
                ['SPY.US', 'AGG.US'],
                ccy='USD',
                first_date='2010-01',
                last_date='2024-10'
            )
        >>> # set cash flow strategy
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -2_500  # set annual withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency to year
        >>> pf.dcf.cashflow_parameters = ind
        >>> # Calculate the historical survival period for the cash flow strategy.
        >>> # The balance is considered voided when it's equal to 0 (threshold=0)
        >>> pf.dcf.survival_period_hist(threshold=0)
        5.1
        """
        return helpers.Date.get_period_length(
            last_date=self.survival_date_hist(threshold=threshold), first_date=self.parent.first_date
        )

    def survival_date_hist(self, threshold: float = 0) -> pd.Timestamp:
        """
        Get the date when the portfolio balance become negative considering withdrawals on the historical data.

        The portfolio survival date (longevity date) depends on the investment strategy: asset allocation,
        rebalancing, withdrawals rate etc.

        Parameters
        ----------
        threshold : float, default 0
            The percentage of the initial investments when the portfolio balance considered voided.
            This parameter is important to use in cash flow strategies with a fixed
            whtdrawal percentage (PercentageStrategy).

        Returns
        -------
        pd.Timestamp
            The portfolio survival date (longevity period) in years.

        Examples
        --------
        >>> pf = ok.Portfolio(
                ['SPY.US', 'AGG.US'],
                ccy='USD',
                first_date='2010-01',
                last_date='2024-10'
            )
        >>> # set cash flow strategy
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -2_500  # set annual withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency to year
        >>> pf.dcf.cashflow_parameters = ind
        >>> # Calculate the historical survival period for the cash flow strategy
        >>> pf.dcf.survival_date_hist(threshold=0)
        Timestamp('2015-01-31 00:00:00')
        """
        ws = self.wealth_index(discounting="fv", include_negative_values=False).loc[:, self.parent.symbol]
        # TODO: change threshold to nominal value (idea)
        return helpers.Frame.get_survival_date(ws, self.discount_rate, threshold)

    @property
    def initial_investment_pv(self) -> Optional[float]:
        """
        The discounted value (PV) of the initial investments at the historical first date.

        The future value (FV) is defined by `initial_amount` parameter.

        Returns
        -------
        float, None
            The discounted value (PV) of the initial investments at the historical first date.

        Examples
        --------
        >>> # Get discounted PV value of `initial_investment` for a portfolio with 4 years of history (at 2020-04).
        >>> pf = ok.Portfolio(['EQMX.MOEX', 'SBGB.MOEX'], ccy='RUB', last_date="2024-10")
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.discount_rate = 0.10  # define discount rate as 10%
        >>> pf.dcf.initial_investment_pv
        6574.643143611553
        """
        if hasattr(self.cashflow_parameters, "initial_investment"):
            return self.cashflow_parameters.initial_investment / (1.0 + self.discount_rate) ** self.parent.period_length
        else:
            return None

    @property
    def initial_investment_fv(self) -> Optional[float]:
        """
        The future value (FV) of the initial investments at the end of forecast period.

        The forecast period is defined in Monte Carlo parameters ('period').

        FV is defined by the discount rate and the initial investments:
        initial_investment_fv = initial_investment * (1 + discount_rate) ** period

        When 'initial_investment' parameter is not defined, `initial_investment_fv` set to None.

        Returns
        -------
        float, None
            The future value (FV) of the initial investments.

        Examples
        --------
        >>> # Get discounted FV of initial_investment value for a period of 10 years.
        >>> pf = ok.Portfolio(['EQMX.MOEX', 'SBGB.MOEX'], ccy='RUB')
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.mc.period = 10  # define forecast period
        >>> pf.dcf.discount_rate = 0.10  # define discount rate as 10%
        >>> pf.dcf.initial_investment_fv
        25937.424601000024
        """
        if hasattr(self.cashflow_parameters, "initial_investment"):
            return float(self.cashflow_parameters.initial_investment * (1.0 + self.discount_rate) ** self.mc.period)
        else:
            return None


    def monte_carlo_wealth(
            self,
            discounting: Literal["fv", "pv"],
            include_negative_values: bool = True
    ) -> pd.DataFrame:
        """
        Portfolio not discounted random wealth indexes with cash flows (withdrawals/contributions) by Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series (not discounted).
        Each wealth index is calculated with rate of return time series of a given distribution.

        First date of forecasted returns is portfolio last_date.
        First value for the forecasted wealth indexes is the last historical portfolio index value. It is useful
        for a chart with historical wealth index and forecasted values.

        Returns
        -------
        DataFrame
            Table with n random wealth indexes monthly time series.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_strategy='month')
        >>> pf.dcf.set_mc_parameters(distribution="t", period=10, number=100)  # Set Monte Carlo parameters
        >>> # set cash flow parameters
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -500  # set withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.monte_carlo_wealth_fv.plot()
        >>> plt.legend("")  # don't show legend for each line
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._monte_carlo_wealth_fv.empty:
            return_ts = self.parent.monte_carlo_returns_ts(
                distr=self.mc.distribution, years=self.mc.period, n=self.mc.number
            )
            self._monte_carlo_wealth_fv = return_ts.apply(
                dcf_calculations.get_wealth_indexes_fv_with_cashflow,
                axis=0,
                args=(
                    None,  # portfolio_symbol
                    None,  # inflation_symbol
                    self.cashflow_parameters,
                    "monte_carlo",  # calculate wealth index for Monte Carlo
                ),
            )
        if not include_negative_values:
            wealth_index_fv = self._monte_carlo_wealth_fv.copy()
            wealth_index_fv = wealth_index_fv.apply(dcf_calculations.remove_negative_values, axis=0)
            # all_cells_are_nan = wealth_index_fv.isna().all(axis=1)
            # monte_carlo_wealth_fv = wealth_index_fv[~all_cells_are_nan]
            monte_carlo_wealth_fv = wealth_index_fv.fillna(0)
        else:
            monte_carlo_wealth_fv = self._monte_carlo_wealth_fv.copy()
        if discounting.lower() == "fv":
            return monte_carlo_wealth_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(monte_carlo_wealth_fv, self.discount_rate)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")


    def monte_carlo_cash_flow(
            self,
            discounting: Literal["fv", "pv"],
            remove_if_wealth_index_negative: bool = True
    ) -> pd.DataFrame:
        """
        Portfolio not discounted random wealth indexes with cash flows (withdrawals/contributions) by Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series (not discounted).
        Each wealth index is calculated with rate of return time series of a given distribution.

        First date of forecasted returns is portfolio last_date.
        First value for the forecasted wealth indexes is the last historical portfolio index value. It is useful
        for a chart with historical wealth index and forecasted values.

        Returns
        -------
        DataFrame
            Table with n random wealth indexes monthly time series.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_strategy='month')
        >>> pf.dcf.set_mc_parameters(distribution="t", period=10, number=100)  # Set Monte Carlo parameters
        >>> # set cash flow parameters
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -500  # set withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.monte_carlo_wealth_fv.plot()
        >>> plt.legend("")  # don't show legend for each line
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._monte_carlo_cash_flow_fv.empty:
            return_ts = self.parent.monte_carlo_returns_ts(
                distr=self.mc.distribution, years=self.mc.period, n=self.mc.number
            )
            self._monte_carlo_cash_flow_fv = return_ts.apply(
                dcf_calculations.get_cash_flow_fv,
                axis=0,
                args=(
                    self.parent.symbol,  # portfolio_symbol
                    self.cashflow_parameters,
                    "monte_carlo",  # task
                ),
            )
        if remove_if_wealth_index_negative:
            mc_cash_flow_fv = self._monte_carlo_cash_flow_fv.copy()
            mc_wealth_index = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
            condition = mc_wealth_index == 0
            mc_cash_flow_fv[condition] = 0
        else:
            mc_cash_flow_fv = self._monte_carlo_cash_flow_fv.copy()
        if discounting.lower() == "fv":
            return mc_cash_flow_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(mc_cash_flow_fv, self.discount_rate)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")

    def plot_forecast_monte_carlo(
        self,
        backtest: bool = True,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        Plot Monte Carlo simulation for portfolio future wealth indexes optionally together with historical wealth index.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        time period considering cash flows (portfolio withdrawals/contributions).

        Random wealth indexes are generated according to a given distribution.

        Parameters
        ----------
        backtest : bool, default 'True'
            Include historical wealth index if 'True'.

        figsize : (float, float), optional
            Width, height in inches.
            If None default matplotlib figsize value is used.

        Returns
        -------
        None

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(assets=['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_strategy='year')
        >>> # Set Monte Carlo parameters
        >>> pf.dcf.set_mc_parameters(distribution="norm", period=50, number=200)
        >>> # set cash flow parameters
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -500  # set withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.plot_forecast_monte_carlo(backtest=True)
        >>> plt.yscale("log")  # Y-axis has logarithmic scale
        >>> plt.show()
        """
        # TODO: return axe
        if backtest:
            if self.cashflow_parameters is None:
                raise AttributeError("'cashflow_parameters' is not defined.")
            backup_obj = self.cashflow_parameters
            s1 = self.wealth_index(discounting="fv", include_negative_values=False)[self.parent.symbol]
            s1.plot(legend=None, figsize=figsize)
            last_backtest_value = s1.iloc[-1]
            if last_backtest_value > 0:
                self.cashflow_parameters.initial_investment = last_backtest_value
                if self.cashflow_parameters.NAME == "fixed_amount":
                    months = helpers.Date.get_difference_in_months(self.parent.last_date, self.parent.first_date).n
                    years = months / settings._MONTHS_PER_YEAR
                    periods = years / settings.frequency_periods_per_year[self.cashflow_parameters.frequency]
                    self.cashflow_parameters.amount *= (1.0 + self.cashflow_parameters.indexation) ** periods
                s2 = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
                for s in s2:
                    s2[s].plot(legend=None)
            self.cashflow_parameters = backup_obj
        else:
            s2 = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
            s2.plot(legend=None)
        self.cashflow_parameters._clear_cf_cache()

    def monte_carlo_survival_period(self, threshold: float = 0) -> pd.Series:
        """
        Generate a survival period distribution for a portfolio with cash flows by Monte Carlo simulation.

        Analyzing the result, finding "min", "max" and percentiles it's possible to see for how long
        will last the investment strategy - possible longevity period.

        Parameters
        ----------
        threshold : float, default 0
            The percentage of the initial investments when the portfolio balance considered voided.
            This parameter is important to use in cash flow strategies with a fixed
            whtdrawal percentage (PercentageStrategy).

        Returns
        -------
        Series
            Survival period distribution for a portfolio with cash flows.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05])
        >>> # set Monte Carlos parameters
        >>> pf.dcf.set_mc_parameters(
        ...        distribution="t",  # use Student's distribution (t-distribution)
        ...        period=50,  # make forecast for 50 years
        ...        number=200  # create 200 randow wealth indexes
        ...    )
        >>> # Set Cash Flow parameters
        >>> pc = ok.PercentageStrategy(pf)  # create PercentageStrategy linked to the portfolio
        >>> pc.initial_investment = 10_000  # add initial investments size
        >>> pc.frequency = "year"  # set cash flow frequency
        >>> pc.percentage = -0.20  # set withdrawal percentage
        >>> # Assign the strategy to Portfolio
        >>> pf.dcf.cashflow_parameters = pc
        >>> s = pf.dcf.monte_carlo_survival_period(threshold=0.10)  # the balance is considered voided at 10%
        >>> s.min()
        np.float64(10.5)
        >>> s.max()
        np.float64(33.5)
        >>> s.mean()
        np.float64(17.9055)
        >>> s.quantile(50 / 100)
        np.float64(17.5)
        """
        s2 = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        dates: pd.Series = helpers.Frame.get_survival_date(s2, self.discount_rate, threshold)
        return dates.apply(helpers.Date.get_period_length, args=(self.parent.last_date,))

    def find_the_largest_withdrawals_size(
        self,
        goal: Literal['maintain_balance_pv', 'maintain_balance_fv', 'survival_period'],
        withdrawals_range: Tuple[float, float] = (0, 1),
        target_survival_period: int = 25,
        percentile: int = 20,
        threshold: float = 0,
        tolerance_rel: float = 0.10,
        iter_max: int = 20,
    ) -> Result:
        """
        Find the largest withdrawals size for Monte Carlo simulation according to Cashflow Strategy.

        It's possible to find the largest withdrawl with 3 kind of goals:

        — 'maintain_balance_pv' to keep the purchasing power of the invesments after inflation
            for the whole period defined in Monte Carlo parameteres.
        — 'maintain_balance_fv' to keep the nominal size of the invesments for the whole period
            defined in Monte Carlo parameteres.
        — 'survival_period' to keep positive balance for a period defined by 'target_survival_period'.

        The method works with IndexationStrategy and PercentageStrategy only.

        The withdrawal size defined in cash flow strategy must be negative.

        The result of finding a solution has the following parameters:
        - 'success' - whether the solution was found or not.
        - 'withdrawal_abs' - the absolute amount of withdrawal size (the best solution if found).
        - 'withdrawal_rel' - the relative amount of withdrawal size (the best solution if found).
        - 'error_rel' - characterizes how accurately the goal is fulfilled.
        - 'solutions' - the history of attempts to find solutions (withdrawal values and error level).

        The algorithm uses bisection method to find the largest withdrawals size.

        Returns
        -------
        Result
            The result of finding solution process.

        Parameters
        ----------
        goal : {'maintain_balance_fv', 'maintain_balance_pv', 'survival_period'}
            'maintain_balance_fv' - the goal is to maintain the balance not lower than the nominal amount of the initial investment after inflation
            for the whole period defined in Monte Carlo parameteres.
            'maintain_balance_pv' - the goal is to keep the purchasing power of the invesments after inflation
            for the whole period defined in Monte Carlo parameteres.
            'survival_period' - the goal is to keep positive balance
            for a period defined by 'target_survival_period'.

        withdrawals_range : tuple of (float, float), default (0, 1)
            The expected range of annualized withdrawals size measured as a percentage
            of the Initial Investment (CashFlow.initial_investment).
            0.01 stands for 1%. (0.02, 0.05) means that expexted withdrawal is in range from 2% to 5% of Initial Investment.
            The first value is expected minimum withdrawal. The second value is expected maximum withdrawal.
            The search for a solution occurs only within this range.

        percentile : int, default 20
            The percentile of Monte-Carlo simulation distribution where the goal is achieved.
            Percentile must be form 0 to 100.
            1th or 5th percentiles are the examples of "bad" scenarios. 50th is mediane.
            95th or 99th are optimiststic scenarios.

        threshold : float, default 0
            The percentage of initial investments when the portfolio balance is considered voided.
            Important for the "fixed_percentage" Cash flow strategy.

        target_survival_period: int, default 25
            The smallest acceptable survival period. It wokrs with the 'survival_period' goal only.

        iter_max : integer, default 20
            The maximum number of iterations to find the solution.

        tolerance_rel : float, default 0.10
            The allowed tolerance for the solution. The tolerance is the largest error for the achieved goal.

        Examples
        --------
        >>> pf = ok.Portfolio(
         ...       assets=["MCFTR.INDX", "RUCBTRNS.INDX"],
         ...       weights=[.3, .7],
         ...       inflation=True,
         ...       ccy="RUB",
         ...       rebalancing_strategy=ok.Rebalance(period="year"),
         ...   )
        >>> # Fixed Percentage strategy
        >>> pc = ok.PercentageStrategy(pf)
        >>> pc.initial_investment = 10_000
        >>> pc.frequency = "year"
        >>> # Assign a strategy
        >>> pf.dcf.cashflow_parameters = pc
        >>> # Set Monte Carlo parameters
        >>> pf.dcf.set_mc_parameters(
        ...    distribution="norm",
        ...    period=50,
        ...    number=200
        ...)
        >>> res = pf.dcf.find_the_largest_withdrawals_size(
        ...    percentile=50,
        ...    goal="survival_period",
        ...    threshold=0.05,
        ...    target_survival_period=25
        ...)
        >>> res
        success                True
        withdrawal_abs   -917.96875
        withdrawal_rel     0.091797
        error_rel           0.00442
        attempts                 10
        dtype: object

        in the result the 'withdrawal_abs' is the absolute value of the withdrawal (the first withdrawal value),
        and the 'withdrawal_rel' the relative withdrawal size (the first withdrawal value divided by the initial investment).

        If the solution was not found it's still possible to see the intermediate steps.

        >>> res.solutions
          withdrawal_abs withdrawal_rel error_rel error_rel_change
        0       -10000.0              1     0.968                0
        1        -5000.0            0.5     0.848            -0.12
        2        -2500.0           0.25    0.6082          -0.2398
        3        -1250.0          0.125   0.24816         -0.36004
        4         -625.0         0.0625   0.55576           0.3076
        5         -937.5        0.09375   0.00442         -0.55134
        """
        if withdrawals_range[0] > withdrawals_range[1]:
            raise ValueError("withdrawals_range[0] must be smaller than withdrawals_range[1]")
        if withdrawals_range[0] < 0 or withdrawals_range[1] > 1:
            raise ValueError("withdrawals_range[0] and withdrawals_range[1] must be in range form 0 to 1.")
        if target_survival_period > self.mc.period:
            raise ValueError(
                f"target_survival_period must be less or equal than Monte Carlo simulation period ({self.mc.period})."
            )
        if percentile > 100 or percentile < 0:
            raise ValueError("percentile must be between 0 and 100")
        if threshold > 1 or threshold < 0:
            raise ValueError("threshold must be between 0 and 1")
        backup_obj = self.cashflow_parameters
        start_investment = self.cashflow_parameters.initial_investment
        if self.cashflow_parameters.NAME == "fixed_amount":
            expected_max_withdrawal = (
                -withdrawals_range[1] * start_investment / self.cashflow_parameters.periods_per_year
            )
            expected_min_withdrawal = (
                -withdrawals_range[0] * start_investment / self.cashflow_parameters.periods_per_year
            )
            self.cashflow_parameters.amount = expected_max_withdrawal
        elif self.cashflow_parameters.NAME == "fixed_percentage":
            expected_max_withdrawal = withdrawals_range[1]
            expected_min_withdrawal = withdrawals_range[0]
            self.cashflow_parameters.percentage = -expected_max_withdrawal
        else:
            raise ValueError("This method works with IndexationStrategy or PercentageStrategy only.")
        iter = 0
        solutions = pd.DataFrame(columns=["withdrawal_abs", "withdrawal_rel", "error_rel", "error_rel_change"])
        while True:
            sp_at_quantile = self.monte_carlo_survival_period(threshold=threshold).quantile(percentile / 100)
            if self.cashflow_parameters.NAME == "fixed_amount":
                main_parameter = self.cashflow_parameters.amount
            elif self.cashflow_parameters.NAME == "fixed_percentage":
                main_parameter = self.cashflow_parameters.percentage
            if goal in ["maintain_balance_fv", "maintain_balance_pv"]:
                print(f"the goal is {goal}")
                s = self.monte_carlo_wealth_pv if goal == "maintain_balance_pv" else self.monte_carlo_wealth_fv
                wealth_at_quantile = s.iloc[-1, :].quantile(percentile / 100)
                condition = (wealth_at_quantile >= start_investment) and (sp_at_quantile == self.mc.period)
                print(f"{wealth_at_quantile=:.2f}, {main_parameter=:.3f}")
                error_rel = abs(wealth_at_quantile - start_investment) / start_investment
            elif goal == "survival_period":
                condition = sp_at_quantile >= target_survival_period
                print(f"{sp_at_quantile=:.2f}, {main_parameter=:.3f}")
                error_rel = abs(sp_at_quantile - target_survival_period) / target_survival_period
            else:
                raise ValueError("The goal can be: maintain_balance_fv, maintain_balance_pv or survival_period.")

            withdrawal_abs = (
                main_parameter
                if self.cashflow_parameters.NAME == "fixed_amount"
                else main_parameter * start_investment / self.cashflow_parameters.periods_per_year
            )
            solutions.at[iter, "withdrawal_abs"] = withdrawal_abs
            withdrawal_rel = (
                abs(main_parameter / start_investment * self.cashflow_parameters.periods_per_year)
                if self.cashflow_parameters.NAME == "fixed_amount"
                else abs(self.cashflow_parameters.percentage)
            )
            solutions.at[iter, "withdrawal_rel"] = withdrawal_rel
            solutions.at[iter, "error_rel"] = error_rel
            gradient = solutions.at[iter, "error_rel"] - solutions.at[iter - 1, "error_rel"] if iter != 0 else 0
            solutions.at[iter, "error_rel_change"] = gradient

            print(f"{error_rel=:.3f}, {gradient=:.3f}")

            if error_rel < tolerance_rel:
                print(f"solution found: {withdrawal_abs:.2f} or {withdrawal_rel * 100:.2f}% after {iter + 1} steps.")
                result = Result(
                    success=True,
                    withdrawal_abs=withdrawal_abs,
                    withdrawal_rel=withdrawal_rel,
                    error_rel=error_rel,
                    solutions=solutions,
                )
                break

            if condition:
                expected_min_withdrawal = main_parameter
                delta = abs(expected_max_withdrawal - main_parameter)
                if self.cashflow_parameters.NAME == "fixed_amount":
                    self.cashflow_parameters.amount -= delta / 2
                elif self.cashflow_parameters.NAME == "fixed_percentage":
                    self.cashflow_parameters.percentage -= delta / 2
                print("increasing withdrawal")
            else:
                expected_max_withdrawal = main_parameter
                delta = abs(main_parameter - expected_min_withdrawal)
                if self.cashflow_parameters.NAME == "fixed_amount":
                    self.cashflow_parameters.amount += delta / 2
                elif self.cashflow_parameters.NAME == "fixed_percentage":
                    self.cashflow_parameters.percentage += delta / 2
                print("decreasing withdrawal")
            iter += 1
            if iter > iter_max - 1:
                condition = solutions["error_rel"].idxmin()
                best_result_abs = solutions.loc[condition]["withdrawal_abs"]
                best_result_rel = solutions.loc[condition]["withdrawal_rel"]
                best_err_rel = solutions.loc[condition]["error_rel"]
                print(
                    f"Didn't found solution after {iter} steps. "
                    f"The closest withdrawal was {best_result_abs} or {best_result_rel * 100:.2f}% "
                    f"with an error: {best_err_rel * 100:.2f}%"
                )
                result = Result(
                    success=False,
                    withdrawal_abs=best_result_abs,
                    withdrawal_rel=best_result_rel,
                    error_rel=best_err_rel,
                    solutions=solutions,
                )
                break

        self.cashflow_parameters = backup_obj
        self.cashflow_parameters._clear_cf_cache()
        return result
