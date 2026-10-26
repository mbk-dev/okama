"""Original GMV properties from b33ca12; independent of Git history at test time."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from okama.common.helpers import helpers


class ScalarGMV:
    def __init__(self, ror: pd.DataFrame, bounds: tuple[tuple[float, float], ...]) -> None:
        self.assets_ror, self.bounds = ror, bounds

    @property
    def gmv_monthly_weights(self) -> np.ndarray:
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
        >>> two_assets = ok.EfficientFrontierSingle(["SPY.US", "AGG.US"])
        >>> two_assets.gmv_monthly_weights
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
            raise RuntimeError("No solutions were found")

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
        >>> ef = ok.EfficientFrontierSingle(["SPY.US", "AGG.US"])
        >>> ef.gmv_annual_weights
        array([0.05474178, 0.94525822])
        """
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        # Set the objective function
        def objective_function(w: np.ndarray) -> float:
            risk = helpers.Frame.get_portfolio_risk(w, self.assets_ror)
            mean_return = helpers.Frame.get_portfolio_mean_return(w, self.assets_ror)
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
        if weights.success:
            return weights.x
        else:
            raise RuntimeError("No solutions were found")

    @property
    def gmv_monthly(self) -> tuple[float, float]:
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
        >>> ef = ok.EfficientFrontierSingle(["SPY.US", "AGG.US", "GLD.US"])
        >>> ef.gmv_monthly
        (0.01024946425526032, 0.0036740056018316597)
        """
        return (
            helpers.Frame.get_portfolio_risk(self.gmv_monthly_weights, self.assets_ror),
            helpers.Frame.get_portfolio_mean_return(self.gmv_monthly_weights, self.assets_ror),
        )

    @property
    def gmv_annualized(self) -> tuple[float, float]:
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
        >>> ef = ok.EfficientFrontierSingle(["SPY.US", "AGG.US", "GLD.US"])
        >>> ef.gmv_annualized
        (0.03697734994430258, 0.0449899573148429)
        """
        return (
            helpers.Float.annualize_risk(self.gmv_monthly[0], self.gmv_monthly[1]),
            helpers.Float.annualize_return(self.gmv_monthly[1]),
        )
