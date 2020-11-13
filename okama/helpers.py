import math
from typing import Union

import pandas as pd
import numpy as np

_MONTHS_PER_YEAR = 12


class Float:
    """
    Group of methods using float values inputs.
    Some of them can take DataFrane also.
    """

    @staticmethod
    def get_monthly_return_from_annual(annual_return: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Gets value of monthly return for a given annual return.
        """
        periods_per_year = 12
        return (1. + annual_return)**(1/periods_per_year) - 1.

    @staticmethod
    def annualize_return(rate_of_return: Union[float, pd.Series], periods_per_year: int = 12) -> Union[float, pd.Series]:
        """
        Annualizes a return.
        Default annualization is from month to year.
        """
        return (rate_of_return + 1.) ** periods_per_year - 1.

    @staticmethod
    def annualize_risk(risk: Union[float, pd.Series], mean_return: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Annualizes Risk.
        Annualization from month to year (from standard deviation) is by default. Monthly mean return is also required.
        Works with DataFrame inputs (in math.sqrt is not used).
        """
        return ((risk ** 2 + (1 + mean_return) ** 2) ** 12 - (1 + mean_return) ** 24) ** 0.5

    @staticmethod
    def approx_return_risk_adjusted(mean_return: float, std: float) -> float:
        """
        Approximates geometric mean return given mean return and std.
        """
        return np.exp(np.log(1. + mean_return) - 0.5 * std ** 2 / (1. + mean_return) ** 2) - 1.


class Frame:
    """
    Group of methods using DataFrame (or Series) input.
    """
    # Rate of return metrics

    @staticmethod
    def weights_sum_is_one(weights: list) -> None:
        if np.around(np.sum(weights), decimals=3) != 1.:
            raise ValueError('Weights sum is not equal to one.')
        if any(x < 0 for x in weights):
            raise ValueError('Negative weights are detected.')

    @classmethod
    def get_portfolio_return_ts(cls, weights: list, ror: pd.DataFrame) -> pd.Series:
        """
        Returns the mean return time series given portfolio weights and the DataFrame of assets mean returns.
        """
        cls.weights_sum_is_one(weights)
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror
        return_ts = ror @ weights
        return return_ts

    @classmethod
    def get_portfolio_mean_return(cls, weights: list, ror: pd.DataFrame) -> float:
        """
        Computes mean return of a portfolio (month scale).
        """
        # cls.weights_sum_is_one(weights)
        weights = np.asarray(weights)
        if isinstance(ror.mean(), float):  # required for a single asset portfolio
            return ror.mean()
        return weights.T @ ror.mean()

    @staticmethod
    def get_ror(close_ts: pd.Series, period: str = 'M') -> pd.Series:
        """
        Calculates rate of return time series given a close ts.
        Periods:
        'D' - daily return
        'M' - monthly return
        """
        if period == 'D':
            ror = close_ts.pct_change().iloc[1:]
            return ror
        if period == 'M':
            close_ts = close_ts.resample('M').last()
            # Replacing zeroes by NaN and padding
            # TODO: replace with pd .where(condition, value, inplace=True)
            if (close_ts == 0).any():
                toxic = close_ts[close_ts == 0]
                for i in toxic.index:
                    close_ts[i] = None
                close_ts.fillna(method='backfill', inplace=True, limit=3)
                if close_ts.isna().any():
                    raise Exception("Too many NaN or zeros in data. Can't pad the data.")
            ror = close_ts.pct_change().iloc[1:]
            return ror
        else:
            raise TypeError(f"{period} is not a supported period")

    @staticmethod
    def get_cagr(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
        """
        Return Compound Annual Rate of Return (CAGR) for each asset given returns time series DataFrame.
        """
        if ror.shape[0] < 12:
            return pd.Series({x: None for x in ror.columns})  # CAGR is not defined for time periods < 1 year
        return ((ror + 1.).prod()) ** (12 / ror.shape[0]) - 1

    @staticmethod
    def get_compound_return(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
        """
        Return Compound Return for time series of return (one or several).
        """
        return (ror + 1.).prod() - 1.

    @staticmethod
    def get_annual_return_ts_from_monthly(ror_monthly: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Annual Rate of Returns time series from monthly data.
        """
        return ror_monthly.resample('A').apply(lambda x: np.prod(x + 1.) - 1)

    @staticmethod
    def get_wealth_indexes(ror: pd.DataFrame, first_value: float = 1000.) -> pd.Series:
        """
        Returns wealth indexes for a list of assets (or for portfolio).
        Works also with pd.Series inputs.
        """
        initial_investments = first_value
        first_date = ror.index[0]
        wealth_index = initial_investments * (ror + 1).cumprod()
        wealth_index.loc[first_date] = initial_investments  # replaces NaN with the first period return
        wealth_index.sort_index(ascending=True, inplace=True)
        return wealth_index

    @staticmethod
    def _adjust_rates(rates: pd.Series, symbol: str) -> pd.Series:
        """
        Makes several adjustment to the rates before OKID index is computed:
        - penalty for bank license revocation (except TOP 10 banks)
        - monthly interest capitalization adjustment
        """
        if symbol != 'RUS_RUB_TOP10.RATE':
            #  license revocation does not happen among TOP 10 banks
            penalty1 = 14 / 365 / 2   # penalty for license revocation (period 1 - frequent revocation)
            penalty2 = (14 / 365) * (30 / 440)  # penalty for license revocation (period 2 - rare revocation)
            rates['28.09.2018':] = rates['28.09.2018':] * (1 - penalty1)
            rates[:'28.10.2018'] = rates[:'28.10.2018'] * (1 - penalty2)
        # compensate monthly interest capitalization (okid index has it)
        rates = ((rates + 1.) ** (1 / _MONTHS_PER_YEAR) - 1) * 12.
        return rates

    @staticmethod
    def get_okid_index(rates: pd.Series, symbol: str) -> pd.Series:
        """
        Computes OKID index from bank rates time series.
        The index is a basket of 12 one year bank deposits (1 month shift).
        symbol is required for rates time series adjustments (for 'RUS_RUB_TOP10.RATE' it's a different adjustment)
        """
        index_total = None
        start_period = pd.Period(rates.index[0], freq='M')
        end_period = pd.Period(rates.index[-1], freq='M')
        rates = Frame._adjust_rates(rates, symbol)
        for month_idx in range(_MONTHS_PER_YEAR):
            rates_yearly = rates.values[month_idx::_MONTHS_PER_YEAR]
            index_part = np.repeat(rates_yearly, _MONTHS_PER_YEAR)[:len(rates) - month_idx]
            index_part = (1 + index_part / _MONTHS_PER_YEAR).cumprod()
            index_total = index_part if index_total is None else index_total[1:] + index_part
            start_period += 1
        if index_total is None:
            raise Exception('`index_total` should not be `None`')
        result = index_total
        result = (np.diff(result) / result[:-1] + 1.).cumprod() - 1.
        result = [0] + result
        result = (result + 1) * 100
        result = pd.Series(result, index=pd.period_range(start_period, end_period, freq='M'))
        result.loc[start_period - 1] = 100.
        result.sort_index(ascending=True, inplace=True)
        return result

    # Risk metrics

    @classmethod
    def get_portfolio_risk(cls, weights: list, ror: pd.DataFrame) -> float:
        """
        Computes the std of portfolio returns.
        """
        # cls.weights_sum_is_one(weights)
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror.std()
        weights = np.array(weights)
        covmat = ror.cov()
        return math.sqrt(weights.T @ covmat @ weights)

    @staticmethod
    def get_semideviation(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
        """
        Returns semideviation for each asset given returns time series.
        """
        is_negative = ror < 0
        return ror[is_negative].std(ddof=0)

    @staticmethod
    def get_var_historic(ror: Union[pd.DataFrame, pd.Series], level: int = 5) -> Union[pd.Series, float]:
        """
        Returns the historic Value at Risk (VaR) at a specified level
        """
        if isinstance(ror, pd.DataFrame) or isinstance(ror, pd.Series):
            return -ror.quantile(level / 100)
        else:
            raise TypeError("Expected ror to be a pd.Series or pd.DataFrame")

    @staticmethod
    def get_cvar_historic(ror: Union[pd.DataFrame, pd.Series], level: int = 5) -> Union[pd.Series, float]:
        """
        Computes the Conditional VaR (CVaR) of Series or DataFrame at a specified level.
        """
        if isinstance(ror, pd.Series) or isinstance(ror, pd.DataFrame):
            is_beyond = ror <= ror.quantile(level / 100)  # mask: return is less than quantile
            return -ror[is_beyond].mean()
        else:
            raise TypeError("Expected ror to be a pd.Series or pd.DataFrame")

    @staticmethod
    def get_drawdowns(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        From returns time series gets drawdowns.
        """
        wealth_index = 1000 * (1 + ror).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns

    @staticmethod
    def change_columns_order(df: pd.DataFrame, selected_columns: list, position: str ='first') -> pd.DataFrame:
        """
        Places selected_columns on the first position (position='first') or last position (position='last').
        """
        cols = list(df.columns.values)  # Make a list of all of the columns in the df
        def condition(y): return y in selected_columns
        cols = [x for x in cols if not condition(x)]  # Remove from list
        # Create new DataFrame with columns in the right order
        if position == 'first':
            df = df[selected_columns + cols]
        elif position == 'last':
            df = df[cols + selected_columns]
        return df


class Rebalance:
    """
    Methods for rebalancing portfolio.
    """
    @staticmethod
    def rebalanced_portfolio_wealth_ts(weights: list, ror: pd.DataFrame, *, period: str = 'Y') -> pd.Series:
        """
        Returns wealth index time series of rebalanced portfolio given returns time series of the assets.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'N'
        """
        # Frame.weights_sum_is_one(weights)
        initial_inv = 1000
        wealth_index = pd.Series(dtype='float64')
        if period == 'N':  # Not rebalanced portfolio
            inv_period_spread = np.asarray(weights) * initial_inv
            assets_wealth_indexes = inv_period_spread * (1 + ror).cumprod()
            wealth_index = assets_wealth_indexes.sum(axis=1)
        else:
            for x in ror.resample(period):
                df = x[1]  # select ror part of the grouped data
                inv_period_spread = np.asarray(weights) * initial_inv  # rebalancing
                assets_wealth_indexes = inv_period_spread * (1 + df).cumprod()
                wealth_index_local = assets_wealth_indexes.sum(axis=1)
                wealth_index = pd.concat([wealth_index, wealth_index_local], verify_integrity=True, sort=True)
                initial_inv = wealth_index.iloc[-1]
        return wealth_index

    @staticmethod
    def rebalanced_portfolio_return_ts(weights: list, ror: pd.DataFrame, *, period: str = 'Y') -> pd.Series:
        """
        Returns mean return time series of rebalanced portfolio given returns time series of the assets.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'N'
        """
        # define data of the first period
        first_date = ror.index[0]
        return_first_period = ror.iloc[0] @ weights

        wealth_index = Rebalance.rebalanced_portfolio_wealth_ts(weights, ror, period=period)
        ror = wealth_index.pct_change()
        ror.loc[first_date] = return_first_period  # replaces NaN with the first period return
        # ror.sort_index(ascending=True, inplace=True)
        return ror


class Date:
    @staticmethod
    def subtract_years(dt: pd.Timestamp, years: int) -> pd.Timestamp:
        """
        Subtracts N years (integer) from a date. Used for time series.
        First month is +1 (if today is August the series should start at September to give 12 months).
        """
        if isinstance(years, int):
            if dt.month == 12:
                dt = dt.replace(year=dt.year - years, month=1)  # for December
            else:
                dt = dt.replace(year=dt.year - years, month=dt.month + 1)
        else:
            raise TypeError('The period should be integer')
        return dt

