import time
import itertools
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from scipy.optimize import minimize

import okama.common.helpers.rebalancing
from okama import asset_list, settings
from okama.common.helpers import helpers
from okama.common.helpers.rebalancing import Rebalance

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EfficientFrontierReb(asset_list.AssetList):
    """
    Efficient Frontier with multi-period optimization.

    In multi-period optimization portfolios are rebalanced with a given frequency.

    Rebalancing is the process by which an investor restores their portfolio to its target allocation
    by selling and buying assets. After rebalancing all the assets have original weights.

    Parameters
    ----------
    assets : list, default None
        List of assets. Could include tickers or asset like objects (Asset, Portfolio).
        If None a single asset list with a default ticker is used.

    first_date : str, default None
        First date of monthly return time series.
        If None the first date is calculated automatically as the oldest available date for the listed assets.

    last_date : str, default None
        Last date of monthly return time series.
        If None the last date is calculated automatically as the newest available date for the listed assets.

    ccy : str, default 'USD'
        Base currency for the list of assets. All risk metrics and returns are adjusted to the base currency.

    bounds: tuple of ((float, float),...)
        Bounds for the assets weights. Each asset can have weights limitation from 0 to 1.0.
        If an asset has limitation for 10 to 20%, bounds are defined as (0.1, 0.2).
        bounds = ((0, .5), (0, 1)) shows that in Portfolio with two assets first one has weight limitations
        from 0 to 50%. The second asset has no limitations.

    inflation : bool, default True
        Defines whether to take inflation data into account in the calculations.
        Including inflation could limit available data (last_date, first_date)
        as the inflation data is usually published with a one-month delay.

    rebalancing_strategy : Rebalance, default Rebalance(period='year', abs_deviation=None, rel_deviation=None)
        Rebalancing strategy for an investment portfolio. The rebalancing strategy si defined by:
        -period (rebalancing frequency): predetermined time intervals when the investor rebalances the portfolio.
        If 'none' assets weights are not rebalanced.
        -abs_deviation: the absolute deviation allowed for the assets weights in the portfolio.
        -rel_deviation: the relative deviation allowed for the assets weights in the portfolio.

    n_points : int, default 20
        Number of points in the Efficient Frontier.

    full_frontier : bool, default True
        Defines whether to show the full Efficient Frontier or only its upper part.
        If 'False' Efficient Frontier has only the points with the return above Global Minimum Volatility (GMV) point.

    verbose : bool, default False
        If verbose=True calculates elapsed time for each point and the total elapsed time.

    ticker_names : bool, default True
        Defines whether to include full names of assets in the optimization report or only tickers.

    Notes
    -----
    For monthly rebalanced portfolios okama.EfficientFrontier class could be used.
    """

    _FTOL = (1e-06, 1e-05, 1e-3, 1e-02)  # tolerance sequence for the optimizer

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        *,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        ccy: str = "USD",
        bounds: Optional[Tuple[Tuple[float, ...], ...]] = None,
        inflation: bool = False,
        full_frontier: bool = True,
        rebalancing_strategy: Rebalance = Rebalance(period="year"),
        n_points: int = 20,
        verbose: bool = False,
        ticker_names: bool = True,
    ):
        if len(assets) < 2:
            raise ValueError("The number of symbols cannot be less than two")
        super().__init__(
            assets=assets,
            first_date=first_date,
            last_date=last_date,
            ccy=ccy,
            inflation=inflation,
        )

        self._bounds = None
        self.bounds = bounds
        self.rebalancing_strategy = rebalancing_strategy
        self.n_points = n_points
        self.ticker_names = ticker_names
        self.verbose = verbose
        self.full_frontier = full_frontier
        self._ef_points = pd.DataFrame(dtype=float)

    def __repr__(self):
        dic = {
            "symbols": self.symbols,
            "currency": self._currency.ticker,
            "first_date": self.first_date.strftime("%Y-%m"),
            "last_date": self.last_date.strftime("%Y-%m"),
            "period_length": self._pl_txt,
            "rebalancing_period": self.rebalancing_strategy.period,
            "rebalancing_abs_deviation": self.rebalancing_strategy.abs_deviation,
            "rebalancing_rel_deviation": self.rebalancing_strategy.rel_deviation,
            "bounds": self.bounds,
            "inflation": self.inflation if hasattr(self, "inflation") else "None",
        }
        return repr(pd.Series(dic))

    @property
    def bounds(self) -> Tuple[Tuple[float, ...], ...]:
        """
        Return bounds for the assets weights.

        Bounds are used in optimization. Each asset can have weights limitation from 0 to 1.0.

        If an asset has limitation for 10 to 20% bounds are defined as (0.1, 0.2).
        bounds = ((0, .5), (0, 1)) shows that in Portfolio with two assets first one has weight limitations
        from 0 to 50%. The second asset has no limitations.

        Returns
        -------
        tuple of ((float, float),...)
            Weights bounds used for portfolio optimization.

        Examples
        --------
        >>> two_assets = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
        >>> two_assets.bounds
        ((0.0, 1.0), (0.0, 1.0))

        By default there are no limitations for assets weights.
        Bounds can be set for a Efficient Frontier object.

        >>> two_assets.bounds = ((0.5, 0.9), (0, 1.0))

        Now the optimization is bounded (SPY has weights limits from 50 to 90%).
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):

        self._ef_points = pd.DataFrame(dtype=float)

        if bounds:
            if len(bounds) != len(self.symbols):
                raise ValueError(
                    f"The number of symbols ({len(self.symbols)}) "
                    f"and the length of bounds ({len(bounds)}) should be equal."
                )
            self._bounds = bounds
        else:
            self._bounds = ((0.0, 1.0),) * len(self.symbols)  # an N-tuple of 2-tuples

    @property
    def n_points(self) -> int:
        """
        Return or set number of points in the Efficient Frontier.

        Returns
        -------
        int
            Number of points in the Efficient Frontier.

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'BND.US'])
        >>> frontier.n_points  # default number of points
        20
        """
        return self._n_points

    @n_points.setter
    def n_points(self, n_points: int):
        if not isinstance(n_points, int):
            raise ValueError("n_points should be an integer")
        if n_points <= 0:
            raise ValueError("n_points should be greater than zero")
        self._clear_cache()
        self._n_points = n_points

    def _clear_cache(self):
        self._ef_points = pd.DataFrame(dtype=float)  # renew EF points DataFrame

    @property
    def rebalancing_strategy(self):
        """
        Return or set rebalancing period for multi-period Efficient Frontier.

        Rebalancing periods could be:
        'year' - one Year (default)
        'none' - not rebalanced portfolios

        Returns
        -------
        str
            Rebalancing period value.

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'BND.US'])
        >>> frontier.rebalancing_strategy  # default rebalancing period is one year
        'year'

        >>> frontier.rebalancing_strategy = 'none'  # change for not rebalanced portfolios
        """
        return self._rebalancing_strategy

    @rebalancing_strategy.setter
    def rebalancing_strategy(self, rebalancing_strategy: Rebalance):
        if isinstance(rebalancing_strategy, Rebalance):
            self._clear_cache()
            self._rebalancing_strategy = rebalancing_strategy
        else:
            raise ValueError("rebalancing_strategy must be of type Rebalance")

    @property
    def ticker_names(self):
        """
        Return or set whether to show tickers or full stock names in the reports.

        Returns
        -------
        bool
            True - for tickers.
            False - for full stock (index) names.
        """
        return self._tickers

    @ticker_names.setter
    def ticker_names(self, tickers: bool):
        if not isinstance(tickers, bool):
            raise ValueError("tickers should be True or False")
        self._tickers = tickers

    @property
    def verbose(self):
        """
        Return or set whether to show technical information during the optimization.

        Returns
        -------
        bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        if not isinstance(verbose, bool):
            raise ValueError("verbose should be True or False")
        self._verbose = verbose

    @property
    def gmv_monthly_weights(self) -> np.ndarray:
        """
        Calculate asset weights of the Global Minimum Volatility (GMV) portfolio. The objective function is
        monthly risk (standard deviation of return).

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        Along the Efficient Frontier, the left-most point is a portfolio with minimum risk when compared to
        all possible portfolios of risky assets.

        Returns
        -------
        numpy.ndarray
            GMV portfolio assets weights.

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'AGG.US'])
        >>> frontier.gmv_monthly_weights
        array([0.0578446, 0.9421554])
        """
        ror = self.assets_ror
        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        # Set the objective function
        def objective_function(w):
            risk = okama.common.helpers.rebalancing.Rebalance(**args).return_ror_ts(w, ror).std()
            return risk

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_1,),
            bounds=self.bounds,
        )
        return weights.x

    @property
    def gmv_annual_weights(self) -> np.ndarray:
        """
        Calculate asset weights of the Global Minimum Volatility (GMV) portfolio. The objective function is
        annualized risk (standard deviation of return).

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        Along the Efficient Frontier, the left-most point is a portfolio with minimum risk when compared to
        all possible portfolios of risky assets.

        Returns
        -------
        numpy.ndarray
            GMV portfolio assets weights.

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'AGG.US'])
        >>> frontier.gmv_monthly_weights
        array([0.05373824, 0.94626176])
        """
        ror = self.assets_ror
        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        # Set the objective function
        def objective_function(w):
            ts = Rebalance(**args).return_ror_ts(w, ror)
            mean_return = ts.mean()
            risk = ts.std()
            return helpers.Float.annualize_risk(risk=risk, mean_return=mean_return)

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_1,),
            bounds=self.bounds,
        )
        return weights.x

    def _get_gmv_monthly(self) -> Tuple[float, float]:
        """
        Calculate the risk and return (mean, monthly) of the Global Minimum Volatility portfolio.

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        """
        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        ts = Rebalance(**args).return_ror_ts(self.gmv_monthly_weights, self.assets_ror)
        return ts.std(), ts.mean()

    @property
    def gmv_annual_values(self) -> Tuple[float, float]:
        """
        Calculate the annualized risk (standard deviation) and CAGR of the Global Minimum Volatility portfolio.

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        Compound annual growth rate (CAGR) is the rate of return that would be required for an investment to grow from
        its initial to its final value, assuming all incomes were reinvested.

        Returns
        -------
        tuple
            Annualized value of risk (standard deviation),
            Compound annual growth rate (CAGR)
            for Global Minimum Volatility portfolio (GMV).

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'AGG.US'])
        >>> frontier.gmv_annual_values
        (0.03695845106087943, 0.04418318557516887)
        """
        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        returns = Rebalance(**args).return_ror_ts(self.gmv_annual_weights, self.assets_ror)
        return (
            helpers.Float.annualize_risk(returns.std(), returns.mean()),
            (returns + 1.0).prod() ** (settings._MONTHS_PER_YEAR / returns.shape[0]) - 1.0,
        )

    @property
    def global_max_return_portfolio(self) -> dict:
        """
        Find a portfolio with global max CAGR.

        Compound annual growth rate (CAGR) is the rate of return that would be required for an investment to grow from
        its initial to its final value, assuming all incomes were reinvested.

        The objective function is Accumulated return for rebalanced portfolio time series for the period
        from 'first_date' to 'last_date'.

        Returns
        -------
        dict
            Weights of assets, CAGR, annualized risk, monthly risk.

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'AGG.US'])
        >>> frontier.global_max_return_portfolio
        {'Weights': array([1., 0.]), 'CAGR': 0.10797159166196812, 'Risk': 0.1583011735798155, 'Risk_monthly': 0.0410282468594492}
        """
        ror = self.assets_ror
        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        n = self.assets_ror.shape[1]  # Number of assets
        init_guess = np.repeat(1 / n, n)

        # Set the objective function
        def objective_function(w):
            # Accumulated return for rebalanced portfolio time series
            objective_function.returns = Rebalance(**args).return_ror_ts(w, ror)
            accumulated_return = (objective_function.returns + 1.0).prod() - 1.0
            return -accumulated_return

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            options={
                "disp": False,
                "maxiter": 100,
                "ftol": self._FTOL[0],
            },
            constraints=(weights_sum_to_1,),
            bounds=self.bounds,
        )
        portfolio_ts = objective_function.returns
        mean_return = portfolio_ts.mean()
        portfolio_risk = portfolio_ts.std()
        point = {
            "Weights": weights.x,
            "CAGR": (1 - weights.fun) ** (settings._MONTHS_PER_YEAR / self.assets_ror.shape[0]) - 1,
            "Risk": helpers.Float.annualize_risk(portfolio_risk, mean_return),
            "Risk_monthly": portfolio_risk,
        }
        return point

    def _get_cagr(self, weights):
        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        ts = Rebalance(**args).return_ror_ts(weights, self.assets_ror)
        acc_return = (ts + 1.0).prod() - 1.0
        return (1.0 + acc_return) ** (settings._MONTHS_PER_YEAR / ts.shape[0]) - 1.0

    def minimize_risk(self, target_value: float) -> Dict[str, float]:
        """
        Calculate the portfolio properties to minimize annualized value of risk at the target CAGR.

        Compound annual growth rate (CAGR) is the rate of return that would be required for an investment to grow from
        its initial to its final value, assuming all incomes were reinvested.

        The objective function is Annualized risk (standard deviation) for rebalanced portfolio time series
        for the period from 'first_date' to 'last_date'.

        Returns
        -------
        dict
            Weights of assets, CAGR, annualized risk.

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'AGG.US'])
        >>> frontier.minimize_risk(0.107)
        {'SPY.US': 0.9810857623382343, 'AGG.US': 0.018914237661765643, 'CAGR': 0.107, 'Risk': 0.1549703673806012}
        """

        n = self.assets_ror.shape[1]  # number of assets
        init_guess = np.repeat(1 / n, n)  # initial weights

        max_ratio_data = self._max_ratio_asset_right_to_max_cagr

        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )

        if max_ratio_data is not None:
            # TODO: create other guesses for intermedeate points
            #  (remember the weights for solved points, GMV, global max)
            init_guess = np.repeat(0, n)  # clear weights
            init_guess[self._min_ratio_asset["list_position"]] = 1.0

        def objective_function(w):
            # annual risk
            ts = Rebalance(**args).return_ror_ts(w, self.assets_ror)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            return helpers.Float.annualize_risk(risk_monthly, mean_return)

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        cagr_is_target = {
            "type": "eq",
            "fun": lambda weights: target_value - self._get_cagr(weights),
        }
        for i in range(4):
            weights = minimize(
                objective_function,
                init_guess,
                method="SLSQP",
                options={
                    "disp": False,
                    "maxiter": 80,
                    "ftol": self._FTOL[i],
                },
                constraints=(weights_sum_to_1, cagr_is_target),
                bounds=self.bounds,
            )

            # Calculate points of EF given optimal weights
            if weights.success:
                asset_labels = self.symbols if self.ticker_names else list(self.names.values())
                point = dict(zip(asset_labels, weights.x))
                point["CAGR"] = target_value
                point["Risk"] = weights.fun
                point["FTOL"] = self._FTOL[i]
                point["iter"] = weights.nit
                break
        if not weights.success:
            raise RecursionError(f"No solution found for target CAGR value: {target_value}.")
        return point

    def _maximize_risk(self, target_return: float) -> Dict[str, float]:
        """
        Calculate the portfolio properties to maximize annualized value of risk at the target CAGR.

        The objective function is Annualized risk (standard deviation) for rebalanced portfolio time series
        for the period from 'first_date' to 'last_date'.

        Returns
        -------
        dict
            Weights of assets, CAGR, annualized risk.
        """
        n = self.assets_ror.shape[1]  # number of assets

        init_guess = np.repeat(0, n)

        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        if self._max_ratio_asset_right_to_max_cagr:
            init_guess[self._max_ratio_asset_right_to_max_cagr["list_position"]] = 1.0

        def objective_function(w):
            # annual risk
            ts = Rebalance(**args).return_ror_ts(w, self.assets_ror)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            result = -helpers.Float.annualize_risk(risk_monthly, mean_return)
            return result

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        cagr_is_target = {
            "type": "eq",
            "fun": lambda weights: target_return - self._get_cagr(weights),
        }
        for i in range(4):
            weights = minimize(
                objective_function,
                init_guess,
                method="SLSQP",
                options={
                    "disp": False,
                    "ftol": self._FTOL[i],
                    "maxiter": 80,
                },
                constraints=(weights_sum_to_1, cagr_is_target),
                bounds=self.bounds,
            )

            # Calculate points of EF given optimal weights
            if weights.success:
                asset_labels = self.symbols if self.ticker_names else list(self.names.values())
                point = dict(zip(asset_labels, weights.x))
                point["CAGR"] = target_return
                point["Risk"] = -weights.fun
                point["FTOL"] = self._FTOL[i]
                point["iter"] = weights.nit
                break
        if not weights.success:
            raise RecursionError(f"No solution found for target CAGR value: {target_return}.")
        return point

    @property
    def _max_cagr_asset(self) -> dict:
        """
        Find an asset with max CAGR.
        """
        max_asset_cagr = helpers.Frame.get_cagr(self.assets_ror).max()
        ticker_with_largest_cagr = helpers.Frame.get_cagr(self.assets_ror).nlargest(1, keep="first").index.values[0]
        return {
            "max_asset_cagr": max_asset_cagr,
            "ticker_with_largest_cagr": ticker_with_largest_cagr,
            "list_position": self.symbols.index(ticker_with_largest_cagr),
        }

    @property
    def _min_ratio_asset(self) -> dict:
        """
        The asset with the minimum ratio between the CAGR (Compound Annual Growth Rate)
        and the risk for assets that are "to the left"
        of the portfolio with the maximum CAGR on the efficiency frontier.
        """
        cagr = helpers.Frame.get_cagr(self.assets_ror)
        risk_monthly = self.assets_ror.std()
        mean_return = self.assets_ror.mean()
        risk = helpers.Float.annualize_risk(risk_monthly, mean_return)

        global_max_cagr = self.global_max_return_portfolio["CAGR"]
        global_max_risk = self.global_max_return_portfolio["Risk"]

        cagr_diff = global_max_cagr - cagr
        risk_diff = global_max_risk - risk

        if risk_diff is not None and (risk_diff == 0).any():
            risk_diff += 0.0001  # to avoid division by zero

        ratio = cagr_diff / risk_diff
        left_assets = risk_diff > 0

        if left_assets.any():
            valid_ratios = ratio[left_assets]
            min_ticker = valid_ratios.idxmin()
            return {
                "min_asset_cagr": cagr[min_ticker],
                "ticker_with_smallest_ratio": min_ticker,
                "list_position": self.assets_ror.columns.get_loc(min_ticker),
            }
        else:
            right_assets = risk_diff < 0
            valid_ratios = ratio[right_assets]
            min_ticker = valid_ratios.idxmin()
            return {
                "min_asset_cagr": cagr[min_ticker],
                "ticker_with_smallest_ratio": min_ticker,
                "list_position": self.assets_ror.columns.get_loc(min_ticker),
            }

    @property
    def _max_ratio_asset_right_to_max_cagr(self) -> Optional[dict]:
        """
        The asset with the maximum ratio between the CAGR
        (Compound Annual Growth Rate) and the risk for assets that are “to the right”
        of the portfolio with the maximum CAGR on the efficiency frontier.
        """
        cagr = helpers.Frame.get_cagr(self.assets_ror)
        risk_monthly = self.assets_ror.std()
        mean_return = self.assets_ror.mean()
        risk = helpers.Float.annualize_risk(risk_monthly, mean_return)
        tolerance = 0.01

        global_max_cagr = self.global_max_return_portfolio["CAGR"]
        global_max_risk = self.global_max_return_portfolio["Risk"]

        global_max_cagr_is_not_asset = (cagr < global_max_cagr * (1 - tolerance)).all()
        if global_max_cagr_is_not_asset:
            cagr_diff = cagr - global_max_cagr
            risk_diff = risk - global_max_risk

            if risk_diff is not None and (risk_diff == 0).any():
                risk_diff += 0.0001  # to avoid division by zero

            ratio = cagr_diff / risk_diff
            right_assets = risk_diff > 0

            if right_assets.any():
                valid_ratios = ratio[right_assets]
                max_ticker = valid_ratios.idxmax()
                return {
                    "max_asset_cagr": cagr[max_ticker],
                    "ticker_with_largest_cagr": max_ticker,
                    "list_position": self.assets_ror.columns.get_loc(max_ticker),
                }
        return None

    @property
    def _max_annual_risk_asset(self) -> dict:
        """
        Find an asset with max annual risk.
        """
        max_risk = self.risk_annual.iloc[-1, :].max()
        ticker_with_largest_risk = self.risk_annual.iloc[-1, :].nlargest(1, keep="first").index.values[0]
        return {
            "max_annual_risk": max_risk,
            "ticker_with_largest_risk": ticker_with_largest_risk,
            "list_position": self.symbols.index(ticker_with_largest_risk),
        }

    @property
    def _target_cagr_range_left(self) -> np.ndarray:
        """
        Full range of CAGR values (from min to max).
        """
        min_ratio_data = self._min_ratio_asset
        max_ratio_data = self._max_ratio_asset_right_to_max_cagr

        if min_ratio_data is not None and max_ratio_data is not None:
            min_cagr = min_ratio_data.get("min_asset_cagr")
            max_cagr = self.global_max_return_portfolio["CAGR"]
            return np.linspace(min_cagr, max_cagr, self.n_points)

        if self.full_frontier:
            min_cagr = helpers.Frame.get_cagr(self.assets_ror).min()
        else:
            min_cagr = self.gmv_annual_values[1]

        max_cagr = self.global_max_return_portfolio["CAGR"]
        return np.linspace(min_cagr, max_cagr, self.n_points)

    @property
    def _target_cagr_range_right(self) -> Optional[np.ndarray]:
        """
        Range of CAGR values from the Global CAGR max to the max asset cagr
        to the right of the max CAGR point (if exists).
        """
        if self._max_ratio_asset_right_to_max_cagr:
            ticker_cagr = self._max_ratio_asset_right_to_max_cagr["max_asset_cagr"]
            max_cagr = self.global_max_return_portfolio["CAGR"]
            if not np.isclose(max_cagr, ticker_cagr, rtol=1e-3, atol=1e-05):
                k = abs((self._target_cagr_range_left[0] - self._target_cagr_range_left[-1]) / (max_cagr - ticker_cagr))
                # we don't want too many points in the right range. Therefore if k < 1 n_points value is used
                number_of_points = round(self.n_points / k) + 1 if k > 1 else self.n_points
                target_range = np.linspace(max_cagr, ticker_cagr, number_of_points)
                return target_range[1:]  # skip the first point (max cagr) as it presents in the left part of the EF
        return None

    @property
    def target_risk_range(self) -> np.ndarray:
        """
        Calculate range of annualized risk values (from min risk to max risk).

        The number of values in the range is defined by 'n_points'.
        The risk is defined as standard deviation of monthly rate or returns time series.

        Returns
        -------
        numpy.ndarray
            Annualized risk values (from min risk to max risk)

        Examples
        --------
        >>> frontier = ok.EfficientFrontierReb(['SPY.US', 'AGG.US'])
        >>> frontier.target_risk_range
        array([0.03695845, 0.04334491, 0.04973137, 0.05611783, 0.06250429,
               0.06889075, 0.07527721, 0.08166367, 0.08805012, 0.09443658,
               0.10082304, 0.1072095 , 0.11359596, 0.11998242, 0.12636888,
               0.13275534, 0.1391418 , 0.14552826, 0.15191472, 0.15830117])
        """
        min_std = self.gmv_annual_values[0]
        ticker_with_largest_risk = self.assets_ror.std().nlargest(1, keep="first").index.values[0]
        max_std_monthly = self.assets_ror.std().max()
        mean_return = self.assets_ror.loc[:, ticker_with_largest_risk].mean()
        max_std = helpers.Float.annualize_risk(max_std_monthly, mean_return)
        return np.linspace(min_std, max_std, self.n_points)

    @property
    def ef_points(self):
        """
        Generate multi-period Efficient Frontier.

        Each point on the Efficient Frontier is a rebalanced portfolio with optimized annual risk for a given CAGR.
        In case of non-convexity along the risk axis, the second part of the chart is generated,
        where the maximum risk value is found for each point.

        Returns
        -------
        DataFrame
            Table of weights and risk/return values for the Efficient Frontier.
            The columns:

            - assets weights
            - CAGR
            - Risk (standard deviation)

            All the values are annualized.

        Examples
        --------
        >>> ls = ['SPY.US', 'GLD.US']
        >>> curr = 'USD'
        >>> y = ok.EfficientFrontierReb(assets=ls,
        ...                             first_date='2004-12',
        ...                             last_date='2020-10',
        ...                             ccy=curr,
        ...                             rebalancing_strategy=ok.Rebalnce(period='year'),
        ...                             ticker_names=True,  # use tickers in DataFrame column names (can be set to False to show full assets names instead tickers)
        ...                             n_points=20,  # number of points in the Efficient Frontier
        ...                             full_frontier=False,  # draw the frontier to the global CAGR max only
        ...                             verbose=False)  # verbose mode is False to skip the progress while the EF points are calcualted
        >>> df_reb_year = y.ef_points
        >>> df_reb_year.head(5)
               Risk      CAGR    GLD.US    SPY.US
        0  0.159400  0.087763  0.000000  1.000000
        1  0.157205  0.088171  0.014261  0.985739
        2  0.155007  0.088580  0.028941  0.971059
        3  0.152810  0.088988  0.044079  0.955921
        4  0.150615  0.089397  0.059713  0.940287

        To compare the Efficient Frontiers of annually rebalanced portfolios with not rebalanced portfolios it's possible to draw 2 charts:
        rebalancing_strategy=ok.Rebalance(period='year') and period='none'.

        >>> import matplotlib.pyplot as plt
        >>> y.rebalancing_strategy = ok.Rebalance(period='none')
        >>> df_not_reb = y.ef_points
        >>> fig = plt.figure()
        >>> # Plot the assets points
        >>> y.plot_assets(kind='cagr')
        >>> ax = plt.gca()
        >>> # Plot the Efficient Frontier for annually rebalanced portfolios
        >>> ax.plot(df_reb_year.Risk, df_reb_year.CAGR, label='Annually rebalanced')
        >>> # Plot the Efficient Frontier for not rebalanced portfolios
        >>> ax.plot(df_not_reb.Risk, df_not_reb.CAGR, label='Not rebalanced')
        >>> # Set axis labels and the title
        >>> ax.set_title('Multi-period Efficient Frontier: 2 assets')
        >>> ax.set_xlabel('Risk (Standard Deviation)')
        >>> ax.set_ylabel('Return (CAGR)')
        >>> ax.legend()
        >>> plt.show()
        """
        if self._ef_points.empty:
            self._get_ef_points()
        return self._ef_points

    def _get_ef_points(self):
        """
        Get all the points for the Efficient Frontier running optimizer.

        If verbose=True calculates elapsed time for each point and the total elapsed time.
        """
        main_start_time = time.time()

        # left part of the EF
        def compute_left_part_of_ef(i, target_cagr):
            start_time = time.time()
            row = self.minimize_risk(target_cagr)
            end_time = time.time()
            if self.verbose:
                logger.info(f"left EF point #{i + 1}/{self.n_points} is done in {end_time - start_time:.2f} sec.")
            return row

        ef_points_records = Parallel(n_jobs=-1)(
            delayed(compute_left_part_of_ef)(i, target_cagr)
            for i, target_cagr in enumerate(self._target_cagr_range_left)
        )
        # right part of the EF
        range_right = self._target_cagr_range_right
        if range_right is not None:  # range_right can be a DataFrame. Must put an explicit "is not None"

            def compute_right_part_of_ef(i, target_cagr):
                start_time = time.time()
                row = self._maximize_risk(target_cagr)
                ef_points_records.append(row)
                end_time = time.time()
                if self.verbose:
                    logger.info(
                        f"right EF point #{i + 1}/{len(range_right)} is done in {end_time - start_time:.2f} sec."
                    )
                return row

            ef_points_records += Parallel(n_jobs=-1)(
                delayed(compute_right_part_of_ef)(i, target_cagr) for i, target_cagr in enumerate(range_right)
            )
        df = pd.DataFrame.from_records(ef_points_records)
        df = helpers.Frame.change_columns_order(df, ["Risk", "CAGR"])
        main_end_time = time.time()
        if self.verbose:
            logger.info(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
        self._ef_points = df

    def get_monte_carlo(self, n: int = 100) -> pd.DataFrame:
        """
        Generate N random rebalanced portfolios with Monte Carlo simulation.

        Risk (annualized standard deviation) and Return (CAGR) are calculated for a set of random weights.

        Returns
        -------
        DataFrame
            Table with Return (CAGR) and Risk values for random portfolios
            (portfolios with random asset weights).

        Parameters
        ----------
        n : int, default 100
            Number of random portfolios to generate with Monte Carlo simulation.

        Examples
        --------
        >>> ls_m = ['SPY.US', 'GLD.US', 'PGJ.US', 'RGBITR.INDX', 'MCFTR.INDX']
        >>> curr_rub = 'RUB'
        >>> x = ok.EfficientFrontierReb(assets=ls_m,
        ...                             first_date='2005-01',
        ...                             last_date='2020-11',
        ...                             ccy=curr_rub,
        ...                             rebalancing_strategy=ok.Rebalance(period='year'),  # set rebalancing period to one year
        ...                             n_points=20,
        ...                             verbose=False)
        >>> monte_carlo = x.get_monte_carlo(n=1000)  # it can take some time ...
        >>> monte_carlo.head(5)
               CAGR      Risk
        0  0.182937  0.178518
        1  0.184915  0.172965
        2  0.154892  0.141681
        3  0.185500  0.168739
        4  0.176748  0.192657

        Monte Carlo simulation results can be plotted togeather with the optimized portfolios on the Efficient Frontier.

        >>> import matplotlib.pyplot as plt
        >>> df_reb_year = x.ef_points  # optimize portfolios for EF. Calculations will take some time ...
        >>> fig = plt.figure()
        >>> # Plot the assets points (optional).
        >>> x.plot_assets(kind='cagr')
        >>> ax = plt.gca()
        >>> # Plot random portfolios (Monte Carlo simulation)
        >>> ax.scatter(monte_carlo.Risk, monte_carlo.CAGR)
        >>> # Plot the Efficient Frontier
        >>> ax.plot(df_reb_year.Risk, df_reb_year.CAGR, label='Annually rebalanced')
        >>> # Set the axis labels and Title
        >>> ax.set_title('Multi-period Efficient Frontier & Monte Carlo simulation')
        >>> ax.set_xlabel('Risk (Standard Deviation)')
        >>> ax.set_ylabel('CAGR')
        >>> ax.legend()
        >>> plt.show()
        """
        weights_df = helpers.Float.get_random_weights(n, self.assets_ror.shape[1], self.bounds)

        args = dict(
            period=self.rebalancing_strategy.period,
            abs_deviation=self.rebalancing_strategy.abs_deviation,
            rel_deviation=self.rebalancing_strategy.rel_deviation,
        )
        # Portfolio risk and cagr for each set of weights
        portfolios_ror = weights_df.aggregate(
            Rebalance(**args).return_ror_ts,
            ror=self.assets_ror,
        )
        random_portfolios = pd.DataFrame()
        for _, data in portfolios_ror.iterrows():
            risk_monthly = data.std()
            mean_return = data.mean()
            risk = helpers.Float.annualize_risk(risk_monthly, mean_return)
            cagr = helpers.Frame.get_cagr(data)
            row = {"Risk": risk, "CAGR": cagr}
            random_portfolios = pd.concat([random_portfolios, pd.DataFrame(row, index=[0])], ignore_index=True)
        return random_portfolios

    def plot_pair_ef(self, tickers="tickers", figsize: Optional[tuple] = None) -> plt.axes:
        """
        Plot Efficient Frontier of every pair of assets.

        Efficient Frontier is a set of portfolios which satisfy the condition that no other portfolio exists
        with a higher expected return but with the same risk (standard deviation of return).

        Arithmetic mean (expected return) is used for optimized portfolios.

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Parameters
        ----------
        tickers : {'tickers', 'names'} or list of str, default 'tickers'
            Annotation type for assets.
            'tickers' - assets symbols are shown in form of 'SPY.US'
            'names' - assets names are used like - 'SPDR S&P 500 ETF Trust'
            To show custom annotations for each asset pass the list of names.

        figsize: (float, float), optional
            Figure size: width, height in inches.
            If None default matplotlib size is taken: [6.4, 4.8]

        Notes
        -----
        It should be at least 3 assets.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> ls4 = ['SPY.US', 'BND.US', 'GLD.US', 'VNQ.US']
        >>> curr = 'USD'
        >>> last_date = '2021-07'
        >>> ef = ok.EfficientFrontierReb(ls4, ccy=curr, last_date=last_date)
        >>> ef.plot_pair_ef()
        >>> plt.show()

        It can be useful to plot the full Efficent Frontier (EF) with optimized 4 assets portfolios
        together with the EFs for each pair of assets.

        >>> ef4 = ok.EfficientFrontierReb(assets=ls4, ccy=curr, n_points=100)
        >>> df4 = ef4.ef_points
        >>> fig = plt.figure()
        >>> # Plot Efficient Frontier of every pair of assets. Optimized portfolios will have 2 assets.
        >>> ef4.plot_pair_ef()  # mean return is used for optimized portfolios.
        >>> ax = plt.gca()
        >>> # Plot the full Efficient Frontier for 4 asset portfolios.
        >>> ax.plot(df4['Risk'], df4['Mean return'], color = 'black', linestyle='--')
        >>> plt.show()
        """
        if len(self.symbols) < 3:
            raise ValueError("The number of symbols cannot be less than 3")
        # self._verify_axes()
        bool_inflation = hasattr(self, "inflation")
        fig, ax = plt.subplots(figsize=figsize)
        for i in itertools.combinations(self.asset_obj_dict.values(), 2):
            sym_pair = list(i)
            index0 = self.symbols.index(sym_pair[0].symbol)
            index1 = self.symbols.index(sym_pair[1].symbol)
            bounds_pair = (self.bounds[index0], self.bounds[index1])
            ef = EfficientFrontierReb(
                assets=sym_pair,
                ccy=self.currency,
                first_date=self.first_date,
                last_date=self.last_date,
                inflation=bool_inflation,
                full_frontier=True,
                n_points=self.n_points,
                bounds=bounds_pair,
            ).ef_points
            ax.plot(ef["Risk"], ef["CAGR"])
        self.plot_assets(kind="cagr", tickers=tickers)
        return ax
