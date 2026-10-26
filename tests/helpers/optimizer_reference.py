"""Original remaining optimizer methods; prior performance improvements are shared."""

import numpy as np
from scipy.optimize import minimize
from okama.common.helpers import helpers
from okama.frontier.single_period import EfficientFrontierSingle


class ScalarOptimizer(EfficientFrontierSingle):
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
        rate_of_return : {cagr, mean_return}, default cagr
            Use CAGR (Compound annual growth rate) or arithmetic mean of return to calculate Sharpe Ratio.

        rf_return : float, default 0
            Risk-free rate of return.

        Returns
        -------
        dict
             Weights of assets, risk and return of the tangency portfolio.

        Examples
        --------
        >>> three_assets = ["SPY.US", "AGG.US", "GLD.US"]
        >>> ef = ok.EfficientFrontierSingle(assets=three_assets, ccy="USD", last_date="2022-06")
        >>> msr = ef.get_tangency_portfolio(rf_return=0.03)  # risk free rate of return is 3%
        >>> msr
        {'Weights': array([0.49179786, 0.20135428, 0.30684787]),
        'Rate_of_return': 0.07986836255902685,
        'Risk': 0.1026393969626808}

        To calculate tangency portfolio parameters for arithmetic mean set `rate_of_return="mean_return"`:

        >>> msr_mean = ef.get_tangency_portfolio(rate_of_return="mean_return", rf_return=0.03)
        >>> msr_mean
        {'Weights': array([0.60878298, 0.        , 0.39121702]),
        'Rate_of_return': 0.09835416896227,
        'Risk': 0.1267417312486844}
        """
        assets_ror = self.assets_ror
        n = self.assets_ror.shape[1]
        init_guess = np.repeat(1 / n, n)

        def of_arithmetic_mean(w):
            # Sharpe ratio
            mean_return_monthly = helpers.Frame.get_portfolio_mean_return(w, assets_ror)
            risk_monthly = helpers.Frame.get_portfolio_risk(w, assets_ror)
            of_arithmetic_mean.rate_of_return = helpers.Float.annualize_return(mean_return_monthly)
            of_arithmetic_mean.risk = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
            return -(of_arithmetic_mean.rate_of_return - rf_return) / of_arithmetic_mean.risk

        def of_geometric_mean(w):
            portfolio_ror = helpers.Frame.get_portfolio_return_ts(w, assets_ror)
            # Rate of return
            mean_return_monthly = helpers.Frame.get_portfolio_mean_return(w, assets_ror)
            of_geometric_mean.rate_of_return = helpers.Frame.get_cagr(portfolio_ror)
            # Risk
            risk_monthly = portfolio_ror.std()
            of_geometric_mean.risk = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
            return -(of_geometric_mean.rate_of_return - rf_return) / of_geometric_mean.risk

        if rate_of_return.lower() in {"cagr", "mean_return"}:
            rate_of_return = rate_of_return.lower()
        else:
            raise ValueError("rate_of_return must be 'cagr or 'mean_return'")

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
            raise RuntimeError("No solutions were found")

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
        >>> ef = ok.EfficientFrontierSingle(["SPY.US", "AGG.US", "GLD.US"])
        >>> ef.optimize_return(option="max")
        {'Weights': array([1.00000000e+00, 1.94289029e-16, 1.11022302e-16]), 'Mean_return_monthly': 0.009144, 'Risk_monthly': 0.041956276163975015}

        The global maximum can be found with constrained optimization using bounds.

        >>> ef.bounds = ((0, 1.0), (0, 1.0), (0.20, 1.0))  # The portfolio should have at least 20% of GLD
        >>> ef.optimize_return(option="max")
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
            raise RuntimeError("No solutions were found")

    def minimize_risk(
        self,
        target_return: float,
        monthly_return: bool = False,
        tolerance: float = 1e-08,
    ) -> dict[str, float]:
        """
        Find minimal risk given the target return within given bounds.

        In Mean-Variance optimization the objective function is risk (standard deviation of return time series).

        Optimization returns a "point" on the Efficient Frontier with annualized values:

        - weights of assets
        - annualized mean rate of return
        - Compound annual growth rate (CAGR)
        - annualized risk (annualized value of standard deviation)

        Target return can have a monthly or annual value.

        Bounds are defined with 'bounds' property.

        Returns
        -------
        dict
            Point on the Efficient Frontier with assets weights and annualized values for mean return, CAGR, risk.

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
        >>> ef = ok.EfficientFrontierSingle(["SPY.US", "AGG.US", "GLD.US"], last_date="2021-07")
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
            "fun": lambda weights: target_return - helpers.Frame.get_portfolio_mean_return(weights, ror),
        }
        weights = minimize(
            objective_function,
            init_guess,
            method="SLSQP",
            constraints=(weights_sum_to_1, return_is_target),
            bounds=self.bounds,
            options={"disp": False, "ftol": tolerance},
        )
        best_x = weights.x if weights.success else None
        best_risk = weights.fun if weights.success else None

        # Corner guard: when the target return equals a single asset's own mean return
        # (e.g. the max-return frontier point), the 100% single-asset portfolio is feasible
        # and is a valid minimum-risk candidate, while SLSQP from the single equal-weights
        # start can fail to converge to that vertex of the bounds (scipy 1.18 reports
        # "Inequality constraints incompatible" there). Mirrors the guard in the rebalanced
        # EfficientFrontier.minimize_risk.
        corner_x = self._single_asset_corner_weights(target_return)
        if corner_x is not None:
            corner_risk = objective_function(corner_x)
            if best_x is None or corner_risk < best_risk:
                best_x, best_risk = corner_x, corner_risk

        if best_x is None:
            raise RuntimeError("No solutions were found")

        # Calculate point of EF given optimal weights
        a_r = helpers.Float.annualize_return(target_return)
        a_risk = helpers.Float.annualize_risk(risk=best_risk, mean_return=target_return)
        # CAGR calculation
        portfolio_return_ts = helpers.Frame.get_portfolio_return_ts(best_x, ror)
        cagr = helpers.Frame.get_cagr(portfolio_return_ts)
        asset_labels = self.get_assets_tickers()
        point = dict(zip(asset_labels, best_x, strict=True))
        point["Mean return"] = a_r
        point["CAGR"] = cagr
        point["Risk"] = a_risk
        return point
