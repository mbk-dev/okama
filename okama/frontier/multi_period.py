import time
import itertools
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.axes import Axes

from scipy.optimize import minimize

from okama import asset_list, settings
from okama.common.helpers import helpers
from okama.common.helpers.rebalancing import Rebalance

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EfficientFrontier(asset_list.AssetList):
    """
    Efficient Frontier with multi-period optimization.

    In multi-period optimization, portfolios are rebalanced according to a rebalancing strategy.

    Rebalancing is the process by which an investor restores a portfolio to its target allocation
    by selling and buying assets. After rebalancing, the portfolio assets have their target weights.

    Parameters
    ----------
    assets : list[str]
        List of assets. Could include tickers or asset-like objects (`Asset`, `Portfolio`).
        Must contain at least two assets.

    first_date : str, default None
        First date of monthly return time series.
        If None the first date is calculated automatically as the oldest available date for the listed assets.

    last_date : str, default None
        Last date of monthly return time series.
        If None the last date is calculated automatically as the newest available date for the listed assets.

    ccy : str, default 'USD'
        Base currency for the list of assets. All risk metrics and returns are adjusted to the base currency.

    bounds : tuple[tuple[float, float], ...] or None, default None
        Bounds for asset weights. Each asset weight is constrained within the corresponding (min, max) pair.
        For example, `((0.0, 0.5), (0.0, 1.0))` restricts the first asset to [0%, 50%] and leaves the second
        asset unconstrained.

    inflation : bool, default False
        Defines whether to take inflation data into account in the calculations.
        Including inflation could limit available data (`first_date`, `last_date`) as the inflation data is
        usually published with a one-month delay.

    rebalancing_strategy : Rebalance, default Rebalance(period='year')
        Rebalancing strategy used to generate portfolio return series during optimization.
        Only `rebalancing_strategy.period` is used; `abs_deviation` and `rel_deviation` are ignored.

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
    For classic single-period (monthly rebalanced, constant-weight) optimization, use `EfficientFrontierSingle`.
    """

    _FTOL = (1e-06, 1e-05, 1e-3, 1e-02)  # possible tolerance values for the optimizer

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
        self._mdp_points = pd.DataFrame(dtype=float)
        self._ror_cache: Dict[tuple, pd.Series] = {}  # Cache for portfolio return time series by weights

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
            "n_points": self.n_points,
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
        >>> two_assets = ok.EfficientFrontierSingle(['SPY.US', 'AGG.US'])
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
        self._mdp_points = pd.DataFrame(dtype=float)

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
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'BND.US'])
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
        self._mdp_points = pd.DataFrame(dtype=float)  # renew MDP points DataFrame
        self._ror_cache = {}  # clear portfolio return time series cache

    def _get_portfolio_ror_ts(self, weights: np.ndarray) -> pd.Series:
        """
        Get portfolio return time series with caching based on weights.

        This method caches the results of portfolio return calculations to avoid
        redundant computations during optimization when the same weights are evaluated
        multiple times (e.g., in objective function and constraints).

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights for each asset.

        Returns
        -------
        pd.Series
            Monthly rate of return time series for the rebalanced portfolio.
        """
        # Round weights to avoid floating point precision issues in cache keys
        weights_key = tuple(np.round(weights, decimals=10))

        if weights_key not in self._ror_cache:
            rebalance = Rebalance(period=self.rebalancing_strategy.period)
            self._ror_cache[weights_key] = rebalance.return_ror_ts_ef(weights, self.assets_ror)

        return self._ror_cache[weights_key]

    @property
    def rebalancing_strategy(self) -> Rebalance:
        """
        Rebalancing strategy used to compute portfolio return series during optimization.

        Only `rebalancing_strategy.period` is used in the optimization; `abs_deviation` and `rel_deviation`
        are ignored.

        Returns
        -------
        Rebalance
            Rebalancing strategy.

        Examples
        --------
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'BND.US'])
        >>> frontier.rebalancing_strategy.period
        'year'

        >>> frontier.rebalancing_strategy = ok.Rebalance(period='none')
        """
        return self._rebalancing_strategy

    @rebalancing_strategy.setter
    def rebalancing_strategy(self, rebalancing_strategy: Rebalance):
        if isinstance(rebalancing_strategy, Rebalance):
            if rebalancing_strategy.abs_deviation is not None or rebalancing_strategy.rel_deviation is not None:
                logger.warning(
                    "Absolute and relative constraints (abs_deviation, rel_deviation) are not considered "
                    "during portfolio optimization. Only the rebalancing period is used."
                )
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
        self._clear_cache()
        self._verbose = verbose

    def get_most_diversified_portfolio(
        self,
        target_return: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate assets weights and portfolio metrics for the most diversified portfolio within bounds.

        The most diversified portfolio is defined as the portfolio with the maximum Diversification Ratio.

        Parameters
        ----------
        target_return : float, default None
            Target Compound Annual Growth Rate (CAGR) for the portfolio. If provided, the optimizer searches for a
            portfolio with the target CAGR and the maximum Diversification Ratio. If None, the global most diversified
            portfolio is returned.

        Returns
        -------
        dict[str, float]
            Mapping with asset weights (keys are tickers or asset names depending on `ticker_names`) and portfolio
            metrics: 'CAGR', 'Risk', and 'Diversification ratio'.

        Examples
        --------
        >>> ls4 = ['SPY.US', 'AGG.US', 'VNQ.US', 'GLD.US']
        >>> x = ok.EfficientFrontier(assets=ls4, ccy='USD', last_date='2021-12')
        >>> x.get_most_diversified_portfolio()  # get a global most diversified portfolio
        {'SPY.US': 0.19612726258395477,
        'AGG.US': 0.649730553241489,
        'VNQ.US': 0.020096313783052246,
        'GLD.US': 0.13404587039150392,
        'CAGR': 0.062355715886719176,
        'Risk': 0.05510135025563423,
        'Diversification ratio': 1.5665720501693001}

        It is possible to get the most diversified portfolio for a given target CAGR.

        >>> x.get_most_diversified_portfolio(target_return=0.10)
        {'SPY.US': 0.3389762570274293,
        'AGG.US': 0.12915657041748244,
        'VNQ.US': 0.15083042115027034,
        'GLD.US': 0.3810367514048179,
        'CAGR': 0.09370688842211439,
        'Risk': 0.11725067815643951,
        'Diversification ratio': 1.4419864802150442}
        """
        ror = self.assets_ror
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        def objective_function(w):
            # Diversification Ratio
            assets_risk = ror.std()
            assets_mean_return = self.assets_ror.mean()
            assets_annualized_risk = helpers.Float.annualize_risk(assets_risk, assets_mean_return)
            weights = np.asarray(w)
            assets_sigma_weighted_sum = weights.T @ assets_annualized_risk

            portfolio_ror = self._get_portfolio_ror_ts(w)
            portfolio_mean_return_monthly = portfolio_ror.mean()
            portfolio_risk_monthly = portfolio_ror.std()

            objective_function.annual_risk = helpers.Float.annualize_risk(
                portfolio_risk_monthly, portfolio_mean_return_monthly
            )
            return -assets_sigma_weighted_sum / objective_function.annual_risk

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        cagr_is_target = {
            "type": "eq",
            "fun": lambda weights: target_return - self._get_cagr(weights),
        }
        constraints = (weights_sum_to_1,) if target_return is None else (weights_sum_to_1, cagr_is_target)

        # set optimizer
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            options={"disp": False},
            constraints=constraints,
            bounds=self.bounds,
        )
        if weights.success:
            # CAGR calculation
            cagr = self._get_cagr(weights.x)
            asset_labels = self.symbols if self.ticker_names else list(self.names.values())
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point["CAGR"] = cagr
            point["Risk"] = objective_function.annual_risk
            point["Diversification ratio"] = -weights.fun
            return point
        else:
            raise RecursionError("No solutions where found")

    def get_tangency_portfolio(self, rf_return: float = 0, rate_of_return: str = "cagr") -> dict:
        """
        Calculate asset weights, risk and return values for tangency portfolio within given bounds.

        Tangency portfolio or Maximum Sharpe Ratio (MSR) is the point on the Efficient Frontier where
        Sharpe Ratio reaches its maximum.

        The Sharpe ratio is the average annual return in excess of the risk-free rate
        per unit of risk (annualized standard deviation).

        Bounds are defined with 'bounds' property.

        Parameters
        ----------
        rf_return : float, default 0
            Risk-free rate of return.

        rate_of_return : {'cagr', 'mean_return'}, default 'cagr'
            Return definition used to calculate Sharpe ratio.

            - 'cagr': Compound Annual Growth Rate.
            - 'mean_return': Arithmetic mean return (annualized).

        Returns
        -------
        dict
             Weights of assets, risk and return of the tangency portfolio.

        Examples
        --------
        >>> three_assets = ['SPY.US', 'AGG.US', 'GLD.US']
        >>> ef = ok.EfficientFrontier(assets=three_assets, ccy='USD', last_date='2022-06')
        >>> msr = ef.get_tangency_portfolio(rf_return=0.03)  # risk free rate of return is 3%
        >>> sorted(msr)
        ['Rate_of_return', 'Risk', 'Weights']

        To calculate tangency portfolio parameters for arithmetic mean set rate_of_return='mean_return':

        >>> msr_mean = ef.get_tangency_portfolio(rate_of_return="mean_return", rf_return=0.03)
        >>> sorted(msr_mean)
        ['Rate_of_return', 'Risk', 'Weights']
        """
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        def of_arithmetic_mean(w):
            # Sharpe ratio with arithmetic mean
            portfolio_ror = self._get_portfolio_ror_ts(w)
            mean_return_monthly = portfolio_ror.mean()
            risk_monthly = portfolio_ror.std()
            of_arithmetic_mean.rate_of_return = helpers.Float.annualize_return(mean_return_monthly)
            of_arithmetic_mean.risk = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
            return -(of_arithmetic_mean.rate_of_return - rf_return) / of_arithmetic_mean.risk

        def of_geometric_mean(w):
            # Sharpe ratio with CAGR
            portfolio_ror = self._get_portfolio_ror_ts(w)
            mean_return_monthly = portfolio_ror.mean()
            of_geometric_mean.rate_of_return = helpers.Frame.get_cagr(portfolio_ror)
            # Risk
            risk_monthly = portfolio_ror.std()
            of_geometric_mean.risk = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
            return -(of_geometric_mean.rate_of_return - rf_return) / of_geometric_mean.risk

        if rate_of_return.lower() in {"cagr", "mean_return"}:
            rate_of_return = rate_of_return.lower()
        else:
            raise ValueError("rate_of_return must be 'cagr' or 'mean_return'")

        objective_function = of_geometric_mean if rate_of_return == "cagr" else of_arithmetic_mean
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
        if weights.success:
            return {
                "Weights": weights.x,
                "Rate_of_return": objective_function.rate_of_return,
                "Risk": objective_function.risk,
            }
        else:
            raise RecursionError("No solutions where found")

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
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
        >>> frontier.gmv_monthly_weights
        array([0.0578446, 0.9421554])
        """
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        # Set the objective function
        def objective_function(w):
            risk = self._get_portfolio_ror_ts(w).std()
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
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
        >>> frontier.gmv_monthly_weights
        array([0.05373824, 0.94626176])
        """
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        # Set the objective function
        def objective_function(w):
            ts = self._get_portfolio_ror_ts(w)
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
        Calculate the risk and geometric mean of the Global Minimum Volatility portfolio.

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        """
        ts = self._get_portfolio_ror_ts(self.gmv_monthly_weights)
        geometric_mean_return = (ts.add(1.0).prod()) ** (1 / ts.shape[0]) - 1.0
        return ts.std(), geometric_mean_return

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
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
        >>> frontier.gmv_annual_values
        (0.03695845106087943, 0.04418318557516887)
        """
        returns = self._get_portfolio_ror_ts(self.gmv_annual_weights)
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
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
        >>> frontier.global_max_return_portfolio
        {'Weights': array([1., 0.]), 'CAGR': 0.10797159166196812, 'Risk': 0.1583011735798155, 'Risk_monthly': 0.0410282468594492}
        """
        n = self.assets_ror.shape[1]  # Number of assets
        init_guess = np.repeat(1 / n, n)

        # Set the objective function
        def objective_function(w):
            # Accumulated return for rebalanced portfolio time series
            objective_function.returns = self._get_portfolio_ror_ts(w)
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

    def _get_cagr(self, weights: np.ndarray) -> float:
        ts = self._get_portfolio_ror_ts(weights)
        acc_return = (ts + 1.0).prod() - 1.0
        return (1.0 + acc_return) ** (settings._MONTHS_PER_YEAR / ts.shape[0]) - 1.0

    def minimize_risk(self, target_value: float) -> Dict[str, float]:
        """
        Calculate the portfolio properties to minimize annualized risk at the target CAGR.

        This method finds the portfolio weights that minimize the annualized risk (standard deviation)
        while achieving a specified target Compound Annual Growth Rate (CAGR).

        The optimization is performed for a rebalanced portfolio over the period from 'first_date' to 'last_date'.
        CAGR is the rate of return required for an investment to grow from its initial to its final value,
        assuming all incomes were reinvested.

        Parameters
        ----------
        target_value : float
            Target Compound Annual Growth Rate (CAGR) for the portfolio.
            Should be a decimal value (e.g., 0.107 for 10.7% annual return).

        Returns
        -------
        dict
            Dictionary containing:
            - Asset weights (one key per asset symbol or name)
            - 'CAGR': Target CAGR value
            - 'Risk': Minimized annualized risk (standard deviation)
            - 'Weights': Array of optimal weights
            - 'iterations': Number of optimization iterations performed

        Raises
        ------
        RecursionError
            If no solution is found for the given target CAGR value.

        Examples
        --------
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
        >>> point = frontier.minimize_risk(0.08)
        >>> round(point["CAGR"], 2)
        0.08
        """

        n = self.assets_ror.shape[1]  # number of assets
        init_guess = np.repeat(1 / n, n)  # initial weights

        max_ratio_data = self._max_ratio_asset_right_to_max_cagr

        if max_ratio_data is not None:
            init_guess = np.repeat(0, n)  # clear weights
            init_guess[self._min_ratio_asset["list_position"]] = 1.0

        def objective_function(w):
            # annual risk
            ts = self._get_portfolio_ror_ts(w)
            risk_monthly = ts.std()
            objective_function.mean_return = ts.mean()
            return helpers.Float.annualize_risk(risk_monthly, objective_function.mean_return)

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        cagr_is_target = {
            "type": "eq",
            "fun": lambda weights: target_value - self._get_cagr(weights),
        }
        # for i in range(4):
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            options={
                "disp": False,
                "maxiter": 80,
                "ftol": self._FTOL[0],
            },
            constraints=(weights_sum_to_1, cagr_is_target),
            bounds=self.bounds,
        )

        # Calculate points of EF given optimal weights
        if weights.success:
            asset_labels = self.symbols if self.ticker_names else list(self.names.values())
            point = dict(zip(asset_labels, weights.x))
            point["CAGR"] = target_value
            point["Mean return"] = objective_function.mean_return * settings._MONTHS_PER_YEAR
            point["Risk"] = weights.fun
            point["Weights"] = weights.x
            point["iterations"] = weights.nit
            # break
        if not weights.success:
            raise RecursionError(f"No solution found for target CAGR value: {target_value}.")
        return point

    def _maximize_risk(self, target_return: float) -> Dict[str, float]:
        """
        Calculate the portfolio properties to maximize annualized value of risk at the target CAGR.

        The objective function is Annualized risk (standard deviation) for rebalanced portfolio time series
        for the period from 'first_date' to 'last_date'.

        Parameters
        ----------
        target_return : float
            Target CAGR value

        Returns
        -------
        dict
            Weights of assets, CAGR, annualized risk and technical optimization parameters.
        """
        n = self.assets_ror.shape[1]  # number of assets

        def objective_function(w):
            # annual risk
            ts = self._get_portfolio_ror_ts(w)
            risk_monthly = ts.std()
            objective_function.mean_return = ts.mean()
            result = -helpers.Float.annualize_risk(risk_monthly, objective_function.mean_return)
            return result

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        cagr_is_target = {
            "type": "eq",
            "fun": lambda weights: target_return - self._get_cagr(weights),
        }

        constraints = [weights_sum_to_1, cagr_is_target]

        # Prepare initial guess hints
        init_guesses = dict()

        # Hint 1: max ratio asset
        if self._max_ratio_asset_right_to_max_cagr:
            init_guess_1 = np.repeat(0, n)
            init_guess_1[self._max_ratio_asset_right_to_max_cagr["list_position"]] = 1.0
            init_guesses["max_ratio_asset_right_to_max_cagr"] = init_guess_1

        # Hint 2: Global max return portfolio
        if hasattr(self, "_global_max_return_portfolio_weights"):
            init_guesses["global_max_return_portfolio"] = self._global_max_return_portfolio_weights.copy()
        else:
            global_max = self.global_max_return_portfolio
            init_guesses["global_max_return_portfolio"] = global_max["Weights"].copy()
            self._global_max_return_portfolio_weights = global_max["Weights"].copy()  # caching result

        solution = None
        for init_guess_key, init_guess_value in init_guesses.items():
            weights = minimize(
                objective_function,
                init_guess_value,
                method="SLSQP",
                options={
                    "disp": False,
                    "ftol": self._FTOL[0],
                    "maxiter": 80,
                },
                constraints=constraints,
                bounds=self.bounds,
            )

            # Calculate points of EF given optimal weights
            if weights.success:
                asset_labels = self.symbols if self.ticker_names else list(self.names.values())
                solution = dict(zip(asset_labels, weights.x))
                solution["CAGR"] = target_return
                solution["Mean return"] = objective_function.mean_return * settings._MONTHS_PER_YEAR
                solution["Risk"] = -weights.fun
                solution["Weights"] = weights.x
                solution["iterations"] = weights.nit
                solution["init_guess"] = init_guess_key
                break

        if solution is None:
            raise RecursionError(f"No solution found for target CAGR value: {target_return}.")
        return solution

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
        The asset with the maximum ratio between CAGR and risk among assets that are to the right
        of the portfolio with the maximum CAGR on the Efficient Frontier.
        """
        cagr = helpers.Frame.get_cagr(self.assets_ror)
        risk_monthly = self.assets_ror.std()
        mean_return = self.assets_ror.mean()
        risk = helpers.Float.annualize_risk(risk_monthly, mean_return)
        tolerance = 0.01

        global_max_cagr = self.global_max_return_portfolio["CAGR"]
        global_max_risk = self.global_max_return_portfolio["Risk"]
        # TODO: global_max_cagr_is_not_asset must be TRUE if CAGR difference is small but RISK difference is big
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
        if self.full_frontier:
            cagr_series = helpers.Frame.get_cagr(self.assets_ror)
            right_asset = self._max_ratio_asset_right_to_max_cagr
            if right_asset is not None:
                exclude_ticker = right_asset.get("ticker_with_largest_cagr")
                if exclude_ticker in cagr_series.index and len(cagr_series) > 1:
                    min_cagr = cagr_series.drop(labels=exclude_ticker).min()
                else:
                    min_cagr = cagr_series.min()
            else:
                min_cagr = cagr_series.min()
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
        >>> frontier = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
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
        >>> y = ok.EfficientFrontier(assets=ls,
        ...                             first_date='2004-12',
        ...                             last_date='2020-10',
        ...                             ccy=curr,
        ...                             rebalancing_strategy=ok.Rebalance(period='year'),
        ...                             ticker_names=True,  # use tickers in DataFrame column names (set to False to show full asset names instead of tickers)
        ...                             n_points=20,  # number of points in the Efficient Frontier
        ...                             full_frontier=False,  # draw the frontier to the global CAGR max only
        ...                             verbose=False)  # verbose mode is False to skip progress while EF points are calculated
        >>> df_reb_year = y.ef_points
        >>> df_reb_year.head(5)
               Risk      CAGR    GLD.US    SPY.US
        0  0.159400  0.087763  0.000000  1.000000
        1  0.157205  0.088171  0.014261  0.985739
        2  0.155007  0.088580  0.028941  0.971059
        3  0.152810  0.088988  0.044079  0.955921
        4  0.150615  0.089397  0.059713  0.940287

        To compare the Efficient Frontiers of annually rebalanced portfolios with not rebalanced portfolios it's possible to draw 2 charts:
        rebalancing_strategy=ok.Rebalance(period='year') and ok.Rebalance(period='none').

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
                        f"right EF point #{i + 1}/{len(range_right)} is done in {end_time - start_time:.2f} sec, init guess: {row['init_guess']}."
                    )
                return row

            ef_points_records += Parallel(n_jobs=-1)(
                delayed(compute_right_part_of_ef)(i, target_cagr) for i, target_cagr in enumerate(range_right)
            )
        df = pd.DataFrame.from_records(ef_points_records)
        df = helpers.Frame.change_columns_order(df, ["Risk", "Mean return", "CAGR"])
        main_end_time = time.time()
        if self.verbose:
            logger.info(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
        self._ef_points = df

    @property
    def mdp_points(self) -> pd.DataFrame:
        """
        Generate Most diversified portfolios frontier for rebalanced portfolios.

        Each point on the Most diversified portfolios frontier is a rebalanced portfolio with optimized
        Diversification ratio for a given CAGR.

        The points are obtained through the constrained optimization process (optimization with bounds).
        Bounds are defined with 'bounds' property.

        Returns
        -------
        DataFrame
            Table of weights and risk/return values for the Most Diversified Portfolios Frontier.
            The columns:

            - assets weights
            - CAGR (geometric mean)
            - Risk (standard deviation)
            - Diversification ratio

            All the values are annualized.

        Examples
        --------
        >>> ls4 = ['SP500TR.INDX', 'MCFTR.INDX', 'RGBITR.INDX', 'GC.COMM']
        >>> y = ok.EfficientFrontier(assets=ls4, ccy='RUB', last_date='2021-12', n_points=20)
        >>> y.mdp_points  # print mdp weights, risk, CAGR and Diversification ratio
                Risk      CAGR  Diversification ratio  ...    MCFTR.INDX   RGBITR.INDX  SP500TR.INDX
        0   0.066040  0.092220               1.234567  ...  2.081668e-16  1.000000e+00  0.000000e+00
        1   0.064299  0.093451               1.245678  ...  0.000000e+00  9.844942e-01  5.828671e-16
        ...

        To plot the Most diversification portfolios line use the DataFrame with the points data.
        Additionally 'Plot.plot_assets()' can be used to show the assets in the chart.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> # Plot the assets points
        >>> y.plot_assets(kind='cagr')  # kind should be set to "cagr" as we take "CAGR" column from the mdp_points.
        >>> ax = plt.gca()
        >>> # Plot the Most diversified portfolios line
        >>> df = y.mdp_points
        >>> ax.plot(df['Risk'], df['CAGR'])  # we chose to plot CAGR which is geometric mean of return series
        >>> # Set the axis labels and the title
        >>> ax.set_title('Most diversified portfolios line')
        >>> ax.set_xlabel('Risk (Standard Deviation)')
        >>> ax.set_ylabel('Return (CAGR)')
        >>> plt.show()
        """
        if self._mdp_points.empty:
            target_cagrs = self._target_cagr_range_left
            rows_list = []  # Collect all rows to concatenate once at the end
            for x in target_cagrs:
                row = self.get_most_diversified_portfolio(target_return=x)
                rows_list.append(row)
            df = pd.DataFrame.from_records(rows_list)
            df = helpers.Frame.change_columns_order(df, ["Risk", "CAGR"])
            self._mdp_points = df
        return self._mdp_points

    def get_monte_carlo(self, n: int = 100) -> pd.DataFrame:
        """
        Generate random rebalanced portfolios with Monte Carlo simulation.

        Risk (annualized standard deviation) and return (CAGR) are calculated for random weights within `bounds`.

        Parameters
        ----------
        n : int, default 100
            Number of random portfolios to generate with Monte Carlo simulation.

        Returns
        -------
        DataFrame
            Table with Return (CAGR) and Risk values for random portfolios (portfolios with random asset weights).

        Examples
        --------
        >>> ls_m = ['SPY.US', 'GLD.US', 'PGJ.US', 'RGBITR.INDX', 'MCFTR.INDX']
        >>> curr_rub = 'RUB'
        >>> x = ok.EfficientFrontier(assets=ls_m,
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

        Monte Carlo simulation results can be plotted together with the optimized portfolios on the Efficient Frontier.

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
        weights_series = helpers.Float.get_random_weights(n, self.assets_ror.shape[1], self.bounds)
        # Portfolio risk and cagr for each set of weights using cache
        rows_list = []  # Collect all rows to create DataFrame once at the end
        for weights in weights_series:
            # Use cached portfolio return time series calculation
            portfolio_ror = self._get_portfolio_ror_ts(weights)
            risk_monthly = portfolio_ror.std()
            mean_return = portfolio_ror.mean()
            risk = helpers.Float.annualize_risk(risk_monthly, mean_return)
            cagr = helpers.Frame.get_cagr(portfolio_ror)
            row = {"Risk": risk, "CAGR": cagr}
            rows_list.append(row)
        return pd.DataFrame.from_records(rows_list)

    def plot_pair_ef(self, tickers="tickers", figsize: Optional[tuple] = None) -> Axes:
        """
        Plot Efficient Frontier for every pair of assets.

        Efficient Frontier is a set of portfolios which satisfy the condition that no other portfolio exists
        with a higher expected return but with the same risk (standard deviation of return).

        Parameters
        ----------
        tickers : {'tickers', 'names'} or list[str], default 'tickers'
            Annotation type for assets.
            'tickers' - assets symbols are shown in form of 'SPY.US'
            'names' - assets names are used like - 'SPDR S&P 500 ETF Trust'
            To show custom annotations for each asset pass the list of names.

        figsize : tuple[float, float], default None
            Figure size (width, height) in inches. If `None`, matplotlib default is used.

        Returns
        -------
        Axes
            Matplotlib axes with the plot.

        Notes
        -----
        At least 3 assets are required.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> ls4 = ['SPY.US', 'BND.US', 'GLD.US', 'VNQ.US']
        >>> curr = 'USD'
        >>> last_date = '2021-07'
        >>> ef = ok.EfficientFrontier(ls4, ccy=curr, last_date=last_date)
        >>> ef.plot_pair_ef()
        >>> plt.show()

        It can be useful to plot the full Efficient Frontier (EF) with optimized 4 asset portfolios
        together with the EFs for each pair of assets.

        >>> ef4 = ok.EfficientFrontier(assets=ls4, ccy=curr, n_points=100)
        >>> df4 = ef4.ef_points
        >>> fig = plt.figure()
        >>> # Plot Efficient Frontier for every pair of assets. Optimized portfolios will have 2 assets.
        >>> ef4.plot_pair_ef()  # CAGR is used for optimized portfolios.
        >>> ax = plt.gca()
        >>> # Plot the full Efficient Frontier for 4 asset portfolios.
        >>> ax.plot(df4['Risk'], df4['CAGR'], color = 'black', linestyle='--')
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
            ef = EfficientFrontier(
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

    def plot_cml(self, rf_return: float = 0, figsize: Optional[tuple] = None):
        """
        Plot Capital Market Line (CML).

        The Capital Market Line (CML) is the tangent line drawn from the point of the risk-free asset (volatility is
        zero) to the point of tangency portfolio or Maximum Sharpe Ratio (MSR) point.

        The slope of the CML is the Sharpe ratio of the tangency portfolio.

        Parameters
        ----------
        rf_return : float, default 0
            Risk-free rate of return.

        figsize : tuple[float, float], default None
            Figure size (width, height) in inches. If `None`, matplotlib default is used.

        Returns
        -------
        Axes
            Matplotlib axes with the plot.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> three_assets = ['SPY.US', 'AGG.US', 'GLD.US']
        >>> ef = ok.EfficientFrontier(assets=three_assets, ccy='USD', full_frontier=True)
        >>> ef.plot_cml(rf_return=0.05)  # Risk-Free return is 5%
        >>> plt.show()
        """
        ef = self.ef_points
        tg = self.get_tangency_portfolio(rf_return=rf_return, rate_of_return="cagr")
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(ef.Risk, ef["CAGR"], color="black")
        ax.scatter(tg["Risk"], tg["Rate_of_return"], linewidth=0, color="green", zorder=10)
        ax.annotate(
            "MSR",
            (tg["Risk"], tg["Rate_of_return"]),
            textcoords="offset points",  # how to position the text
            xytext=(-10, 10),  # distance from text to points (x,y)
            ha="center",  # horizontal alignment can be left, right or center
        )
        # plot the line
        x, y = [0, tg["Risk"]], [rf_return, tg["Rate_of_return"]]
        ax.plot(x, y, linewidth=1)
        # set the axis size
        risk_monthly = self.assets_ror.std()
        mean_return_monthly = self.assets_ror.mean()
        risks = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
        max_return = self.global_max_return_portfolio["CAGR"]
        min_cagr = self.get_cagr().min()
        y_bottom = min(min_cagr, rf_return)
        plot_margin = 0.10
        ax.set_ylim(y_bottom * (1 - plot_margin), max_return * (1 + plot_margin))
        ax.set_xlim(0, max(risks) * (1 + plot_margin))
        # plot the assets
        self.plot_assets(kind="cagr")
        return ax

    def plot_transition_map(self, x_axe: str = "risk", figsize: Optional[tuple] = None) -> Axes:
        """
        Plot Transition Map for optimized portfolios on the Efficient Frontier.

        Transition Map shows the relation between asset weights and optimized portfolio properties:

        - CAGR (Compound annual growth rate)
        - Risk (annualized standard deviation of return)

        Parameters
        ----------
        x_axe : {'risk', 'cagr'}, default 'risk'
            Show the relation between weights and CAGR (if 'cagr') or between weights and Risk (if 'risk').
            CAGR or Risk are displayed on the x-axis.

        figsize : tuple[float, float], default None
            Figure size (width, height) in inches. If `None`, matplotlib default is used.

        Returns
        -------
        Axes
            Matplotlib axes with the plot.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.EfficientFrontier(['SPY.US', 'AGG.US', 'GLD.US'], ccy='USD', inflation=False)
        >>> x.plot_transition_map()
        >>> plt.show()

        Transition Map with default settings shows the relation between risk (standard deviation) and asset weights
        for optimized portfolios. The same relation for CAGR can be shown by setting `x_axe='cagr'`.

        >>> x.plot_transition_map(x_axe='cagr')
        >>> plt.show()
        """
        ef = self.ef_points
        linestyle = itertools.cycle(("-", "--", ":", "-."))
        if x_axe.lower() == "cagr":
            xlabel = "CAGR (Compound Annual Growth Rate)"
            x_axe = "CAGR"
        elif x_axe.lower() == "risk":
            xlabel = "Risk (volatility)"
            x_axe = "Risk"
        else:
            raise ValueError("x_axe parameter must be 'cagr' or 'risk'.")
        fig, ax = plt.subplots(figsize=figsize)
        for i in ef:
            if i not in (
                "Risk",
                "Mean return",
                "CAGR",
                "Weights",
                "iterations",
                "init_guess",
            ):  # select only columns with tickers
                ax.plot(ef[x_axe], ef.loc[:, i], linestyle=next(linestyle), label=i)
        ax.set_xlim(ef[x_axe].min(), ef[x_axe].max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Weights of assets")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        return ax
