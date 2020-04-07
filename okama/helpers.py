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
    # Rate of return metrics

    @staticmethod
    def _weights_sum_is_one(weights: list):
        if np.around(np.sum(weights), decimals=4) != 1.:
            raise ValueError('Weights sum is not equal to one.')
        if any(x < 0 for x in weights):
            raise ValueError('Negative weights are detected.')
    pass

    @staticmethod
    def get_portfolio_return_ts(weights: list, ror: pd.DataFrame) -> pd.Series:
        """
        Returns the mean return time series given portfolio weights and the DataFrame of assets mean returns.
        """
        Frame._weights_sum_is_one(weights)
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror
        return_ts = ror @ weights
        return return_ts

    @staticmethod
    def get_portfolio_mean_return(weights: list, ror: pd.DataFrame) -> float:
        """
        Computes mean return of a portfolio (month scale). Returns a single float number.
        """
        Frame._weights_sum_is_one(weights)
        weights = np.asarray(weights)
        if isinstance(ror.mean(), float):  # required for a single asset portfolio
            return ror.mean()
        return weights.T @ ror.mean()

    @staticmethod
    def get_cagr(ror: pd.DataFrame) -> pd.DataFrame:
        """
        Return Compound Annual Rate of Return (CAGR) for each asset given returns time series DataFrame.
        TODO: update the method to calculate CAGR for 1, 2, 3, 5, 10 years.
        """
        if ror.shape[0] < 12:
            return np.nan  # CAGR is not defined for time periods < 1 year
        return ((ror + 1.).prod()) ** (12 / ror.shape[0]) - 1

    # Risk metrics

    @staticmethod
    def get_portfolio_risk(weights: list, ror: pd.DataFrame) -> float:
        """
        Computes the std of portfolio returns.
        """
        Frame._weights_sum_is_one(weights)
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror.std()
        weights = np.array(weights)
        covmat = ror.cov()
        return math.sqrt(weights.T @ covmat @ weights)

    @staticmethod
    def get_semideviation(ror: pd.DataFrame) -> pd.Series:
        """
        Returns semideviation for each asset given returns time series.
        """
        is_negative = ror < 0
        return ror[is_negative].std(ddof=0)

    @staticmethod
    def get_var_historic(ror, level=5):
        """
        Returns the historic Value at Risk (VaR) at a specified level
        """
        if isinstance(ror, pd.DataFrame) or isinstance(ror, pd.Series):
            return -ror.quantile(level / 100)
        else:
            raise TypeError("Expected ror to be a pd.Series or pd.DataFrame")

    @staticmethod
    def get_cvar_historic(ror: pd.DataFrame, level=5):
        """
        Computes the Conditional VaR (CVaR) of Series or DataFrame at a specified level.
        """
        if isinstance(ror, pd.Series) or isinstance(ror, pd.DataFrame):
            is_beyond = ror <= ror.quantile(level / 100)  # mask: return is less than quantile
            return -ror[is_beyond].mean()
        else:
            raise TypeError("Expected ror to be a pd.Series or pd.DataFrame")

    @staticmethod
    def get_drawdowns(ror: pd.DataFrame) -> pd.DataFrame:
        """
        From returns time series gets drawdowns.
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
    def rebalanced_portfolio_return_ts(weights: list, ror: pd.DataFrame, *, period='Y') -> pd.Series:
        """
        Returns the mean return time series of rebalanced portfolio. Can be used to calculate returns geometric mean for the
        rebalanced portfolio.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'N'
        """
        Frame._weights_sum_is_one(weights)
        initial_inv = 1000

        # define data of the first period
        first_date = ror.index[0]
        returns = ror @ weights
        return_first_period = returns[0]

        if period == 'N':  # Not rebalanced portfolio
            inv_period = initial_inv
            inv_period_spread = np.asarray(weights) * inv_period
            assets_wealth_indexes = inv_period_spread * (1 + ror).cumprod()
            wealth_index = assets_wealth_indexes.sum(axis=1)
            ror = wealth_index.pct_change()
            # ror = ror.iloc[1:] #  drops NaN
            ror.loc[first_date] = return_first_period #  replaces NaN with the first period return
            ror.sort_index(ascending = True, inplace=True)
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
        # ror = ror.iloc[1:]
        ror.loc[first_date] = return_first_period  # replaces NaN with the first period return
        ror.sort_index(ascending=True, inplace=True)
        return ror
