import math

import pandas as pd
import numpy as np

class Float:
    """
    Group of methods using float values inputs.
    Some of them can take DataFrane also.
    """
    @staticmethod
    def annualize_return(rate_of_return: float, periods_per_year=12) -> float:
        """
        Annualizes a return.
        Default annualization is from month to year.
        """
        return (rate_of_return + 1.) ** periods_per_year - 1.

    @staticmethod
    def annualize_risk(risk: float, mean_return: float, periods_per_year=12) -> float:
        """
        Annualizes Risk.
        Annualization from month to year (from standard deviation) is by default. Mean return is also required.
        Works with DataFrame inputs (in math.sqrt is not used)
        """
        annualized_std = ((risk ** 2 + (1 + mean_return) ** 2) ** 12 - (1 + mean_return) ** 24) ** 0.5
        return annualized_std

    @staticmethod
    def approx_return_risk_adjusted(mean_return: float, std: float) -> float:
        """
        Approximates geometric mean return given mean return and std.
        """
        return np.exp(np.log(1. + mean_return) - 0.5 * std ** 2 / (1. + mean_return) ** 2) - 1.


class Frame:
    """
    Group of methods using
    """
    @staticmethod
    def get_portfolio_return_ts(weights: list, ror: pd.DataFrame) -> pd.Series:
        """
        Returns the mean return time series given portfolio weights and the DataFrame of assets mean returns.
        """
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror
        return_ts = ror @ weights
        return return_ts

    @staticmethod
    def get_portfolio_mean_return(weights: list, ror: pd.DataFrame) -> float:
        """
        Computes mean return of a portfolio (month scale). Returns a single float number.
        """
        weights = np.asarray(weights)
        if isinstance(ror.mean(), float):  # required for a single asset portfolio
            return ror.mean()
        return weights.T @ ror.mean()

    @staticmethod
    def get_portfolio_risk(weights: list, ror: pd.DataFrame) -> float:
        """
        Computes the std of portfolio returns.
        """
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror.std()
        weights = np.array(weights)
        covmat = ror.cov()
        return math.sqrt(weights.T @ covmat @ weights)

    @staticmethod
    def drawdowns(ror: pd.DataFrame) -> pd.DataFrame:
        """
        From returns time series gets drawdowns
        """
        wealth_index = 1000 * (1 + ror).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns

class Rebalance:
    """
    Methods for rebalancing portfolio.
    """
    @staticmethod
    def rebalanced_portfolio_return_ts(weights: list, ror: pd.DataFrame, *, period='Y'):
        """
        Returns the mean return time series of rebalanced portfolio. Can be used to calculate returns geometric mean for the
        rebalanced portfolio.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'N'
        """
        initial_inv = 1000
        if period == 'N':  # Not rebalanced portfolio
            inv_period = initial_inv
            inv_period_spread = np.asarray(weights) * inv_period
            assets_wealth_indexes = inv_period_spread * (1 + ror).cumprod()
            wealth_index = assets_wealth_indexes.sum(axis=1)
            ror = wealth_index.pct_change()
            ror = ror.iloc[1:]
            return ror
        grouped = ror.resample(period)
        for i, x in enumerate(grouped):
            if i == 0:
                inv_period = 1000
                wealth_index = pd.Series(dtype='float64')
                # wealth_index_local = pd.Series(dtype='float64')
            df = x[1]
            inv_period_spread = np.asarray(weights) * inv_period  # rebalancing
            assets_wealth_indexes = inv_period_spread * (1 + df).cumprod()
            wealth_index_local = assets_wealth_indexes.sum(axis=1)
            wealth_index = pd.concat([wealth_index, wealth_index_local], verify_integrity=True, sort=True)
            inv_period = wealth_index.iloc[-1]
        ror = wealth_index.pct_change()
        ror = ror.iloc[1:]
        return ror