import itertools
from typing import Optional, Tuple, Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from okama import asset_list
from okama.common.helpers import helpers


class EfficientFrontier(asset_list.AssetList):
    """
    Efficient Frontier with classic Mean-Variance optimization.

    Efficient Frontier is a set of portfolios which satisfy the condition that no other portfolio exists with a higher
    expected return but with the same risk (standard deviation of return).

    The points on the Efficient Frontier are obtained through the constrained optimization process
    (optimization with bounds). Bounds are defined with 'bounds' property.

    In classic Markowitz optimization portfolios are rebalanced monthly and has constant weights
    (single period optimization).

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
        With inflation = False some properties like real return are not available.

    n_points : int, default 20
        Number of points in the Efficient Frontier.

    full_frontier : bool, default True
        Defines whether to show the full Efficient Frontier or only its upper part.
        If 'False' Efficient Frontier has only the points with the return above Global Minimum Volatility (GMV) point.

    ticker_names : bool, default True
        Defines whether to include full names of assets in the optimization report or only tickers.
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        *,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        ccy: str = "USD",
        bounds: Optional[Tuple[Tuple[float, ...], ...]] = None,
        inflation: bool = True,
        full_frontier: bool = True,
        n_points: int = 20,
        ticker_names: bool = True,
    ):
        if len(assets) < 2:
            raise ValueError("The number of symbols cannot be less than two")
        super().__init__(
            assets,
            first_date=first_date,
            last_date=last_date,
            ccy=ccy,
            inflation=inflation,
        )

        self._bounds = None
        self.bounds = bounds
        self.full_frontier = full_frontier
        self.n_points = n_points
        self.labels_are_tickers = ticker_names
        self._ef_points = pd.DataFrame(dtype=float)
        self._mdp_points = pd.DataFrame(dtype=float)

    def __repr__(self):
        dic = {
            'symbols': self.symbols,
            'currency': self._currency.ticker,
            'first_date': self.first_date.strftime("%Y-%m"),
            'last_date': self.last_date.strftime("%Y-%m"),
            'period_length': self._pl_txt,
            'bounds': self.bounds,
            'inflation': self.inflation if hasattr(self, 'inflation') else 'None',
            'n_points': self.n_points,
        }
        return repr(pd.Series(dic))

    @property
    def bounds(self) -> Tuple[Tuple[float, ...], ...]:
        """
        Return bounds for the assets weights.

        Bounds are used in Mean-Variance optimization. Each asset can have weights limitation from 0 to 1.0.

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
    def gmv_weights(self) -> np.ndarray:
        """
        Calculate asset weights of the Global Minimum Volatility (GMV) portfolio within given bounds.

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        Along the Efficient Frontier, the left-most point is a portfolio with minimum risk when compared to
        all possible portfolios of risky assets.

        In Mean-Variance optimization risk is defined as a standard deviation of return time series.

        Bounds are defined with 'bounds' property.

        Returns
        -------
        numpy.ndarray
            GMV portfolio assets weights.

        Examples
        --------
        >>> two_assets = ok.EfficientFrontier(['SPY.US', 'AGG.US'])
        >>> two_assets.gmv_weights
        array([0.05474178, 0.94525822])
        """
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)
        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        weights = minimize(
            helpers.Frame.get_portfolio_risk,
            init_guess,
            args=(self.assets_ror,),
            method="SLSQP",
            options={"disp": False},
            constraints=(weights_sum_to_1,),
            bounds=self.bounds,
        )
        if weights.success:
            return weights.x
        else:
            raise RecursionError("No solutions where found")

    @property
    def gmv_monthly(self) -> Tuple[float, float]:
        """
        Calculate the monthly risk and return of the Global Minimum Volatility (GMV) portfolio within given bounds.

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        Along the Efficient Frontier, the left-most point is a portfolio with minimum risk when compared to
        all possible portfolios of risky assets.

        In Mean-Variance optimization risk is defined as a standard deviation of return time series.

        Bounds are defined with 'bounds' property.

        Returns
        -------
        tuple
            risk, return monthly values for GMV portfolio.

        Examples
        --------
        >>> ef = ok.EfficientFrontier(['SPY.US', 'AGG.US', 'GLD.US'])
        >>> ef.gmv_monthly
        (0.01024946425526032, 0.0036740056018316597)
        """
        return (
            helpers.Frame.get_portfolio_risk(self.gmv_weights, self.assets_ror),
            helpers.Frame.get_portfolio_mean_return(self.gmv_weights, self.assets_ror),
        )

    @property
    def gmv_annualized(self) -> Tuple[float, float]:
        """
        Calculate the annualized risk and return of the Global Minimum Volatility (GMV) portfolio within given bounds.

        Global Minimum Volatility portfolio is a portfolio with the lowest risk of all possible.
        Along the Efficient Frontier, the left-most point is a portfolio with minimum risk when compared to
        all possible portfolios of risky assets.

        In Mean-Variance optimization risk is defined as a standard deviation of return time series.

        Bounds are defined with 'bounds' property.

        Returns
        -------
        tuple
            risk, return annualized values for GMV portfolio.

        Examples
        --------
        >>> ef = ok.EfficientFrontier(['SPY.US', 'AGG.US', 'GLD.US'])
        >>> ef.gmv_annualized
        (0.03697734994430258, 0.0449899573148429)
        """
        return (
            helpers.Float.annualize_risk(self.gmv_monthly[0], self.gmv_monthly[1]),
            helpers.Float.annualize_return(self.gmv_monthly[1]),
        )

    def get_tangency_portfolio(self, rf_return: float = 0) -> dict:
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

        Returns
        -------
        dict
             Weights of assets, risk and return of the tangency portfolio.

        Examples
        --------
        >>> three_assets = ['MCFTR.INDX', 'RGBITR.INDX', 'GC.COMM']
        >>> ef = ok.EfficientFrontier(assets=three_assets, ccy='USD')
        >>> ef.get_tangency_portfolio(rf_return=0.02)  # risk free rate of return is 2%
        {'Weights': array([3.41138555e-01, 1.90819582e-17, 6.58861445e-01]), 'Mean_return': 0.13457274320732382, 'Risk': 0.19563856367290783}
        """
        ror = self.assets_ror
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        def objective_function(w):
            # Sharpe ratio
            mean_return_monthly = helpers.Frame.get_portfolio_mean_return(w, ror)
            risk_monthly = helpers.Frame.get_portfolio_risk(w, ror)
            objective_function.mean_return = helpers.Float.annualize_return(mean_return_monthly)
            objective_function.risk = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
            return -(objective_function.mean_return - rf_return) / objective_function.risk

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
                "Mean_return": objective_function.mean_return,
                "Risk": objective_function.risk,
            }
        else:
            raise RecursionError("No solutions where found")

    def get_most_diversified_portfolio(self,
                                       target_return: Optional[float] = None,
                                       monthly_return: bool = False,
                                       ) -> dict:
        """
        Calculate assets weights, risk, return and Diversification ratio for the most diversified portfolio given
        the target return within given bounds.

        The most diversified portfolio has the largest Diversification Ratio.

        The Diversification Ratio is the ratio of the weighted average of assets risks divided by the portfolio risk.
        In this case risk is the annuilized standatd deviation for the rate of return .

        Returns
        -------
        dict
             Weights of assets, risk and return of the most diversified portfolio.

        Parameters
        ----------
        target_return : float, optional
            Rate of return value. The optimization process looks for a portfolio with the target_return
            and largest Diversification ratio. If not sepcifed global most diversified portfolio is obtained.
            Target return value can be in monthly or annual values depending on 'monthly_return' option.
        monthly_return : bool, default False
            Defines whether to use rate of return monthly (True) or annual (False) values.

        Examples
        --------
        >>> ls4 = ['SPY.US', 'AGG.US', 'VNQ.US', 'GLD.US']
        >>> x = ok.EfficientFrontier(assets=ls4, ccy='USD', last_date='2021-12')
        >>> x.get_most_diversified_portfolio()  # get a global most diversified portfolio
        {'SPY.US': 0.19612726258395477,
        'AGG.US': 0.649730553241489,
        'VNQ.US': 0.020096313783052246,
        'GLD.US': 0.13404587039150392,
        'Mean return': 0.0637820844415733,
        'CAGR': 0.062355715886719176,
        'Risk': 0.05510135025563423,
        'Diversification ratio': 1.5665720501693001}

        It is possible to get the most diversified portfolio for a given target rate of return.
        Set `monthly_return=False` to use annual values for the rate of return.

        >>> x.get_most_diversified_portfolio(target_return=0.10, monthly_return=False)
        {'SPY.US': 0.3389762570274293,
        'AGG.US': 0.12915657041748244,
        'VNQ.US': 0.15083042115027034,
        'GLD.US': 0.3810367514048179,
        'Mean return': 0.10000000151051025,
        'CAGR': 0.09370688842211439,
        'Risk': 0.11725067815643951,
        'Diversification ratio': 1.4419864802150442}
        """
        if (not monthly_return) and (target_return is not None):
            target_return = helpers.Float.get_monthly_return_from_annual(target_return)
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

            portfolio_ror = helpers.Frame.get_portfolio_return_ts(w, ror)
            portfolio_mean_return_monthly = helpers.Frame.get_portfolio_mean_return(w, ror)
            portfolio_risk_monthly = portfolio_ror.std()

            objective_function.annual_risk = helpers.Float.annualize_risk(portfolio_risk_monthly, portfolio_mean_return_monthly)
            objective_function.annual_mean_return = helpers.Float.annualize_return(portfolio_mean_return_monthly)
            return - assets_sigma_weighted_sum / objective_function.annual_risk

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        return_is_target = {
            "type": "eq",
            "fun": lambda weights: target_return - helpers.Frame.get_portfolio_mean_return(weights, ror),
        }
        constraints = (weights_sum_to_1,) if target_return is None else (weights_sum_to_1, return_is_target)

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
            portfolio_return_ts = helpers.Frame.get_portfolio_return_ts(weights.x, ror)
            cagr = helpers.Frame.get_cagr(portfolio_return_ts)
            if not self.labels_are_tickers:
                asset_labels = list(self.names.values())
            else:
                asset_labels = self.symbols
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point["Mean return"] = objective_function.annual_mean_return
            point["CAGR"] = cagr
            point["Risk"] = objective_function.annual_risk
            point["Diversification ratio"] = - weights.fun
            return point
        else:
            raise RecursionError("No solutions where found")

    def optimize_return(self, option: str = "max") -> dict:
        """
        Find a portfolio with global max or min for the rate of return within given bounds.

        The objective function is an arithmetic mean of monthly Rate of return.

        Bounds are defined with 'bounds' property.

        Returns
        -------
        dict
            Weights of assets, risk and return of the portfolio.

        Parameters
        ----------
        option : {'max', 'min'}, default 'max'
            Find objective function global maximun if 'max' or global minimum if 'min'.

        Examples
        --------
        >>> ef = ok.EfficientFrontier(['SPY.US', 'AGG.US', 'GLD.US'])
        >>> ef.optimize_return(option='max')
        {'Weights': array([1.00000000e+00, 1.94289029e-16, 1.11022302e-16]), 'Mean_return_monthly': 0.009144, 'Risk_monthly': 0.041956276163975015}

        The global maximum can be found with constrained optimization using bounds.

        >>> ef.bounds = ((0, 1.), (0, 1.), (0.20, 1.))  # The portfolio should have at least 20% of GLD
        >>> ef.optimize_return(option='max')
        {'Weights': array([8.00000000e-01, 5.48172618e-16, 2.00000000e-01]), 'Mean_return_monthly': 0.008894299999999997, 'Risk_monthly': 0.035570987973869726}
        """
        ror = self.assets_ror
        n = self.assets_ror.shape[1]  # Number of assets
        init_guess = np.repeat(1 / n, n)
        # Set the objective function
        if option == "max":

            def objective_function(w):
                month_return_value = helpers.Frame.get_portfolio_mean_return(w, ror)
                return -month_return_value

        elif option == "min":

            def objective_function(w):
                month_return_value = helpers.Frame.get_portfolio_mean_return(w, ror)
                return month_return_value

        else:
            raise ValueError('option should be "max" or "min"')
        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            constraints=(weights_sum_to_1,),
            bounds=self.bounds,
            options={
                "disp": False,
                "ftol": 1e-08,
            },  # 1e-06 is not enough to optimize monthly returns
        )
        if weights.success:
            portfolio_risk = helpers.Frame.get_portfolio_risk(weights.x, ror)
            if option.lower() == "max":
                optimized_return = -weights.fun
            else:
                optimized_return = weights.fun
            point = {
                "Weights": weights.x,
                "Mean_return_monthly": optimized_return,
                "Risk_monthly": portfolio_risk,
            }
            return point
        else:
            raise RecursionError("No solutions where found")

    def minimize_risk(
        self,
        target_return: float,
        monthly_return: bool = False,
        tolerance: float = 1e-08,
    ) -> Dict[str, float]:
        """
        Find minimal risk given the target return within given bounds.

        In Mean-Variance optimization the objective function is risk (standard deviation of return time series).

        Optimization returns a "point" on the Efficient Frontier with values:

        - weights of assets
        - annualized mean rate of return
        - Compound annual growth rate (CAGR)
        - annualized risk (annualized value of standard deviation)

        Target return can have a monthly or annual value.

        Bounds are defined with 'bounds' property.

        Returns
        -------
        dict
            Point on the Efficient Frontier with assets weights, mean return, CAGR, risk.

        Parameters
        ----------
        target_return : float
            Rate of return value. The optimization process looks for a portfolio with the target_return
            and minimum risk.
            Target return value can be in monthly or annual values depending on 'monthly_return' option.
        monthly_return : bool, default False
            Defines whether to use rate of return monthly (True) or annual (False) values.
        tolerance : float, default 1e-08
            Sets the accuracy for the solver.

        Examples
        --------
        >>> ef = ok.EfficientFrontier(['SPY.US', 'AGG.US', 'GLD.US'], last_date='2021-07')
        >>> ef.minimize_risk(target_return=0.044, monthly_return=False)
        {'SPY.US': 0.03817252986735185,
        'AGG.US': 0.9618274701326482,
        'GLD.US': 0.0,
        'Mean return': 0.04400000000000004,
        'CAGR': 0.04335075344214023,
        'Risk': 0.037003608635098856}
        """
        if not monthly_return:
            target_return = helpers.Float.get_monthly_return_from_annual(target_return)
        ror = self.assets_ror
        n = ror.shape[1]  # number of assets
        init_guess = np.repeat(1 / n, n)  # initial weights

        def objective_function(w):
            return helpers.Frame.get_portfolio_risk(w, ror)

        # construct the constraints
        weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        return_is_target = {
            "type": "eq",
            "fun": lambda weights: target_return
            - helpers.Frame.get_portfolio_mean_return(weights, ror),
        }
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            constraints=(weights_sum_to_1, return_is_target),
            bounds=self.bounds,
            options={"disp": False, "ftol": tolerance},
        )
        if weights.success:
            # Calculate point of EF given optimal weights
            risk = weights.fun
            # Annualize risk and return
            a_r = helpers.Float.annualize_return(target_return)
            a_risk = helpers.Float.annualize_risk(risk=risk, mean_return=target_return)
            # CAGR calculation
            portfolio_return_ts = helpers.Frame.get_portfolio_return_ts(weights.x, ror)
            cagr = helpers.Frame.get_cagr(portfolio_return_ts)
            if not self.labels_are_tickers:
                asset_labels = list(self.names.values())
            else:
                asset_labels = self.symbols
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point["Mean return"] = a_r
            point["CAGR"] = cagr
            point["Risk"] = a_risk
        else:
            raise RecursionError("No solutions were found")
        return point

    @property
    def mean_return_range(self) -> np.ndarray:
        """
        Calculate the range of mean monthly returns (from min to max).

        Number of values in the range is defined by 'n_points'.

        Returns
        -------
        numpy.ndarray
            Monthly rate of return values from min to max.

        Examples
        --------
        >>> ef = ok.EfficientFrontier(['SPY.US', 'AGG.US', 'GLD.US'], last_date='2021-07')
        >>> ef.mean_return_range
        array([0.0033745 , 0.00367816, 0.00398182, 0.00428547, 0.00458913,
        0.00489279, 0.00519645, 0.00550011, 0.00580376, 0.00610742,
        0.00641108, 0.00671474, 0.00701839, 0.00732205, 0.00762571,
        0.00792937, 0.00823303, 0.00853668, 0.00884034, 0.009144  ])
        """
        if self.full_frontier:
            if self.bounds:
                min_return = self.optimize_return(option="min")["Mean_return_monthly"]
                max_return = self.optimize_return(option="max")["Mean_return_monthly"]
            else:
                er = self.assets_ror.mean()
                min_return = er.min()
                max_return = er.max()
        else:
            min_return = self.gmv_monthly[1]
            if self.bounds:
                max_return = self.optimize_return(option="max")["Mean_return_monthly"]
            else:
                er = self.assets_ror.mean()
                max_return = er.max()
        return np.linspace(min_return, max_return, self.n_points)

    @property
    def ef_points(self) -> pd.DataFrame:
        """
        Generate single period Efficient Frontier.

        Each point on the Efficient Frontier is a portfolio with optimized risk for a given return.

        The points are obtained through the constrained optimization process (optimization with bounds).
        Bounds are defined with 'bounds' property.

        Returns
        -------
        DataFrame
            Table of weights and risk/return values for the Efficient Frontier.
            The columns:

            - assets weights
            - CAGR (geometric mean)
            - Mean return (arithmetic mean)
            - Risk (standard deviation)

            All the values are annualized.

        Examples
        --------
        >>> assets = ['SPY.US', 'AGG.US', 'GLD.US']
        >>> last_date='2021-07'
        >>> y = ok.EfficientFrontier(assets, last_date=last_date)
        >>> y.ef_points
                Risk  Mean return      CAGR        AGG.US        GLD.US        SPY.US
        0   0.037707     0.041254  0.040579  1.000000e+00  9.278755e-09  2.220446e-16
        1   0.036979     0.045042  0.044394  9.473684e-01  0.000000e+00  5.263158e-02
        2   0.038027     0.048842  0.048157  8.947368e-01  0.000000e+00  1.052632e-01
        3   0.040517     0.052655  0.051879  8.376442e-01  2.061543e-02  1.417404e-01
        4   0.043944     0.056481  0.055569  7.801725e-01  4.298194e-02  1.768455e-01
        5   0.048125     0.060320  0.059229  7.227015e-01  6.534570e-02  2.119528e-01
        6   0.052902     0.064171  0.062856  6.652318e-01  8.770367e-02  2.470646e-01
        7   0.058144     0.068035  0.066451  6.077632e-01  1.100558e-01  2.821809e-01
        8   0.063753     0.071912  0.070014  5.502956e-01  1.324040e-01  3.173004e-01
        9   0.069655     0.075802  0.073543  4.928283e-01  1.547504e-01  3.524213e-01
        10  0.075796     0.079704  0.077039  4.353613e-01  1.770958e-01  3.875429e-01
        11  0.082136     0.083620  0.080501  3.778987e-01  1.994207e-01  4.226806e-01
        12  0.088645     0.087549  0.083928  3.204253e-01  2.217953e-01  4.577794e-01
        13  0.095300     0.091491  0.087321  2.629559e-01  2.441514e-01  4.928926e-01
        14  0.102084     0.095446  0.090678  2.054869e-01  2.665062e-01  5.280069e-01
        15  0.108984     0.099414  0.093999  1.480175e-01  2.888623e-01  5.631202e-01
        16  0.115991     0.103395  0.097284  9.054789e-02  3.112196e-01  5.982325e-01
        17  0.123096     0.107389  0.100533  3.307805e-02  3.335779e-01  6.333441e-01
        18  0.132674     0.111397  0.103452  0.000000e+00  2.432182e-01  7.567818e-01
        19  0.161413     0.115418  0.103704  1.110223e-16  1.036379e-09  1.000000e+00

        To plot the Efficient Frontier use the DataFrame with the points data. Additionaly 'Plot.plot_assets()'
        can be used to show the assets in the chart.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> # Plot the assets points
        >>> y.plot_assets(kind='cagr')  # kind should be set to "cagr" as we take "CAGR" column from the ef_points.
        >>> ax = plt.gca()
        >>> # Plot the Efficient Frontier
        >>> df = y.ef_points
        >>> ax.plot(df['Risk'], df['CAGR'])  # we chose to plot CAGR which is geometric mean of return series
        >>> # Set the axis labels and the title
        >>> ax.set_title('Single period Efficient Frontier')
        >>> ax.set_xlabel('Risk (Standard Deviation)')
        >>> ax.set_ylabel('Return (CAGR)')
        >>> ax.legend()
        >>> plt.show()
        """
        if self._ef_points.empty:
            target_rs = self.mean_return_range
            df = pd.DataFrame(dtype="float")
            for x in target_rs:
                row = self.minimize_risk(x, monthly_return=True)
                df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
            df = helpers.Frame.change_columns_order(df, ["Risk", "Mean return", "CAGR"])
            self._ef_points = df
        return self._ef_points

    @property
    def mdp_points(self) -> pd.DataFrame:
        """
        Generate Most diversified portfolios line.

        Each point on the Most diversified portfolios line is a portfolio with optimized
        Diversification ratio for a given return.

        The points are obtained through the constrained optimization process (optimization with bounds).
        Bounds are defined with 'bounds' property.

        Returns
        -------
        DataFrame
            Table of weights and risk/return values for the Efficient Frontier.
            The columns:

            - assets weights
            - CAGR (geometric mean)
            - Mean return (arithmetic mean)
            - Risk (standard deviation)
            - Diversification ratio

            All the values are annualized.

        Examples
        --------
        >>> ls4 = ['SP500TR.INDX', 'MCFTR.INDX', 'RGBITR.INDX', 'GC.COMM']
        >>> y = ok.EfficientFrontier(assets=ls4, ccy='RUB', last_date='2021-12', n_points=100)
        >>> y.mdp_points  # print mdp weights, risk, mean return, CAGR and Diversification ratio
                Risk  Mean return      CAGR  ...    MCFTR.INDX   RGBITR.INDX  SP500TR.INDX
        0   0.066040     0.094216  0.092220  ...  2.081668e-16  1.000000e+00  0.000000e+00
        1   0.064299     0.095342  0.093451  ...  0.000000e+00  9.844942e-01  5.828671e-16
        2   0.062761     0.096468  0.094670  ...  0.000000e+00  9.689885e-01  1.110223e-16
        3   0.061445     0.097595  0.095874  ...  5.828671e-16  9.534827e-01  0.000000e+00
        4   0.060364     0.098724  0.097065  ...  3.191891e-16  9.379769e-01  0.000000e+00
        ..       ...          ...       ...  ...           ...           ...           ...
        95  0.258857     0.205984  0.178346  ...  8.840844e-01  1.387779e-17  0.000000e+00
        96  0.266583     0.207214  0.177941  ...  9.130633e-01  3.469447e-18  0.000000e+00
        97  0.274594     0.208446  0.177432  ...  9.420422e-01  0.000000e+00  1.075529e-16
        98  0.282873     0.209678  0.176820  ...  9.710211e-01  2.428613e-17  6.938894e-18
        99  0.291402     0.210912  0.176103  ...  1.000000e+00  2.775558e-16  3.951094e-09
        [100 rows x 8 columns]

        To plot the Most diversification portfolios line use the DataFrame with the points data.
        Additionaly 'Plot.plot_assets()' can be used to show the assets in the chart.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> # Plot the assets points
        >>> y.plot_assets(kind='cagr')  # kind should be set to "cagr" as we take "CAGR" column from the ef_points.
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
            target_rs = self.mean_return_range
            df = pd.DataFrame(dtype="float")
            for x in target_rs:
                row = self.get_most_diversified_portfolio(target_return=x, monthly_return=True)
                df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
            df = helpers.Frame.change_columns_order(df, ["Risk", "Mean return", "CAGR"])
            self._mdp_points = df
        return self._mdp_points

    def get_monte_carlo(self, n: int = 100, kind: str = "mean") -> pd.DataFrame:
        """
        Generate N random portfolios with Monte Carlo simulation.

        Risk (annualized standard deviation) and Return (CAGR) are calculated for a set of random weights.

        Returns
        -------
        DataFrame
            Table with Return and Risk values for random portfolios.

        Parameters
        ----------
        n : int, default 100
            Number of random portfolios to generate with Monte Carlo simulation.
        kind : {'mean','cagr'}, default 'mean'
            Use CAGR for return if kind='cagr', or annualized arithmetic mean if kind='mean'.

        Examples
        --------
        >>> assets = ['SPY.US', 'AGG.US', 'GLD.US']
        >>> last_date='2021-07'
        >>> base_currency = 'EUR'
        >>> y = ok.EfficientFrontier(assets, ccy=base_currency, last_date=last_date)
        >>> y.get_monte_carlo(n=10)  # generate 10 random portfolios
             Return      Risk
        0  0.090393  0.101900
        1  0.075611  0.087561
        2  0.100580  0.151436
        3  0.109584  0.108251
        4  0.092985  0.092296
        5  0.086165  0.108419
        6  0.116168  0.141825
        7  0.079040  0.090309
        8  0.093917  0.092967
        9  0.102236  0.115301

        To plot Monte Carlo simulation result it's useful to combine in with the Efficien Frontier chart.
        Additionaly assets points could be plotted with 'Plot.plot_assets()'.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> # Plot the assets points (optional).
        >>> # The same first and last dates, base currency and return type should be used.
        >>> y.plot_assets(kind='cagr')
        >>> ax = plt.gca()
        >>> # Plot random portfolios risk-return points.
        >>> mc = y.get_monte_carlo(n=1000, kind='cagr')
        >>> ax.scatter(mc.Risk, mc.CAGR, linewidth=0, color='green')
        >>> # Plot the Efficient (optional)
        >>> df = y.ef_points
        >>> ax.plot(df['Risk'], df['CAGR'], color='black', linestyle='dashed', linewidth=3)
        >>> # Set the title and axis labels
        >>> ax.set_title('Single period Efficient Frontier & Monte Carlo simulation')
        >>> ax.set_xlabel('Risk (Standard Deviation)')
        >>> ax.set_ylabel('CAGR')
        >>> ax.legend()
        >>> plt.show()
        """
        weights_series = helpers.Float.get_random_weights(n, self.assets_ror.shape[1])

        # Portfolio risk and return for each set of weights
        random_portfolios = pd.DataFrame(dtype=float)
        for weights in weights_series:
            risk_monthly = helpers.Frame.get_portfolio_risk(weights, self.assets_ror)
            mean_return_monthly = helpers.Frame.get_portfolio_mean_return(weights, self.assets_ror)
            risk = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
            mean_return = helpers.Float.annualize_return(mean_return_monthly)
            if kind.lower() == "cagr":
                cagr = helpers.Float.approx_return_risk_adjusted(mean_return, risk)
                row = dict(Risk=risk, CAGR=cagr)
            elif kind.lower() == "mean":
                row = dict(Risk=risk, Return=mean_return)
            else:
                raise ValueError('kind should be "mean" or "cagr"')
            random_portfolios = pd.concat([random_portfolios, pd.DataFrame(row, index=[0])], ignore_index=True)
        return random_portfolios

    def plot_transition_map(self, cagr: bool = True, figsize: Optional[tuple] = None) -> plt.axes:
        """
        Plot Transition Map for optimized portfolios on the single period Efficient Frontier.

        Transition Map shows the relation between asset weights and optimized portfolios properties:

        - CAGR (Compound annual growth rate)
        - Risk (annualized standard deviation of return)

        Wights are displayed on the y-axis.
        CAGR or Risk - on the x-axis.

        Constrained optimization with weights bounds is available.

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Parameters
        ----------
        bounds: tuple of ((float, float),...)
            Bounds for the assets weights. Each asset can have weights limitation from 0 to 1.0.
            If an asset has limitation for 10 to 20%, bounds are defined as (0.1, 0.2).
            bounds = ((0, .5), (0, 1)) shows that in Portfolio with two assets first one has weight limitations
            from 0 to 50%. The second asset has no limitations.

        cagr : bool, default True
            Show the relation between weights and CAGR (if True) or between weights and Risk (if False).
            of - sets X axe to CAGR (if true) or to risk (if false).
            CAGR or Risk are displayed on the x-axis.

        figsize: (float, float), optional
            Figure size: width, height in inches.
            If None default matplotlib size is taken: [6.4, 4.8]

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.EfficientFrontier(['SPY.US', 'AGG.US', 'GLD.US'], ccy='USD', inflation=False)
        >>> x.plot_transition_map()
        >>> plt.show()

        Transition Map with default setting show the relation between Return (CAGR) and assets weights for optimized portfolios.
        The same relation for Risk can be shown setting cagr=False.

        >>> x.plot_transition_map(cagr=False)
        >>> plt.show()
        """
        ef = self.ef_points
        linestyle = itertools.cycle(("-", "--", ":", "-."))
        x_axe = "CAGR" if cagr else "Risk"
        fig, ax = plt.subplots(figsize=figsize)
        for i in ef:
            if i not in (
                "Risk",
                "Mean return",
                "CAGR",
            ):  # select only columns with tickers
                ax.plot(
                    ef[x_axe], ef.loc[:, i], linestyle=next(linestyle), label=i
                )
        ax.set_xlim(ef[x_axe].min(), ef[x_axe].max())
        if cagr:
            ax.set_xlabel("CAGR (Compound Annual Growth Rate)")
        else:
            ax.set_xlabel("Risk (volatility)")
        ax.set_ylabel("Weights of assets")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        return ax

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
        >>> last_date = '07-2021'
        >>> ef = ok.EfficientFrontier(ls4, ccy=curr, last_date=last_date)
        >>> ef.plot_pair_ef()
        >>> plt.show()

        It can be useful to plot the full Efficent Frontier (EF) with optimized 4 assets portfolios
        together with the EFs for each pair of assets.

        >>> ef4 = ok.EfficientFrontier(assets=ls4, ccy=curr, n_points=100)
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
        bool_inflation = bool(self.inflation)
        fig, ax = plt.subplots(figsize=figsize)
        for i in itertools.combinations(self.symbols, 2):
            sym_pair = list(i)
            index0 = self.symbols.index(sym_pair[0])
            index1 = self.symbols.index(sym_pair[1])
            bounds_pair = (self.bounds[index0], self.bounds[index1])
            ef = EfficientFrontier(
                assets=sym_pair,
                ccy=self.currency,
                first_date=self.first_date,
                last_date=self.last_date,
                inflation=bool_inflation,
                full_frontier=True,
                bounds=bounds_pair,
            ).ef_points
            ax.plot(ef["Risk"], ef["Mean return"])
        self.plot_assets(kind="mean", tickers=tickers)
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

        figsize: (float, float), optional
            Figure size: width, height in inches.
            If None default matplotlib size is taken: [6.4, 4.8]

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> three_assets = ['MCFTR.INDX', 'RGBITR.INDX', 'GC.COMM']
        >>> ef = ok.EfficientFrontier(assets=three_assets, ccy='USD', full_frontier=True)
        >>> ef.plot_cml(rf_return=0.05)  # Risk-Free return is 5%
        >>> plt.show
        """
        ef = self.ef_points
        tg = self.get_tangency_portfolio(rf_return)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ef.Risk, ef['Mean return'], color='black')
        ax.scatter(tg['Risk'], tg['Mean_return'], linewidth=0, color='green')
        ax.annotate("MSR",
                    (tg['Risk'], tg['Mean_return']),
                    textcoords="offset points",  # how to position the text
                    xytext=(-10, 10),  # distance from text to points (x,y)
                    ha="center",  # horizontal alignment can be left, right or center
                    )
        # plot the line
        x, y = [0, tg['Risk']], [rf_return, tg['Mean_return']]
        ax.plot(x, y, linewidth=1)
        # set the axis size
        risk_monthly = self.assets_ror.std()
        mean_return_monthly = self.assets_ror.mean()
        risks = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
        returns = helpers.Float.annualize_return(self.assets_ror.mean())
        ax.set_ylim(0, max(returns) * 1.1)  # height is 10% more than max return
        ax.set_xlim(0, max(risks) * 1.1)  # width is 10% more than max risk
        # plot the assets
        self.plot_assets(kind='mean')
        return ax
