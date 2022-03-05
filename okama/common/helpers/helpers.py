import math
from typing import Union, Callable

import pandas as pd
import numpy as np
import scipy.stats

from okama import settings


def check_rolling_window(
    window: int, ror: Union[pd.Series, pd.DataFrame], window_below_year: bool = False
):
    if not window_below_year and window < settings._MONTHS_PER_YEAR:
        raise ValueError("window size should be at least 1 year")
    if window > ror.shape[0]:
        raise ValueError("window size is more than data history depth")


class Float:
    """
    Group of methods using float values inputs.
    Some of them can take DataFrame also.
    """

    @staticmethod
    def get_monthly_return_from_annual(
        annual_return: Union[float, pd.Series]
    ) -> Union[float, pd.Series]:
        """
        Gets value of monthly return for a given annual return.
        """
        return (1.0 + annual_return) ** (1 / settings._MONTHS_PER_YEAR) - 1.0

    @staticmethod
    def annualize_return(
        rate_of_return: Union[float, pd.Series],
        periods_per_year: int = settings._MONTHS_PER_YEAR,
    ) -> Union[float, pd.Series]:
        """
        Annualizes a return.
        Default annualization is from month to year.
        """
        return (rate_of_return + 1.0) ** periods_per_year - 1.0

    @staticmethod
    def annualize_risk(
        risk: Union[float, pd.Series], mean_return: Union[float, pd.Series]
    ) -> Union[float, pd.Series]:
        """
        Annualizes Risk.
        Annualization from month to year (from standard deviation) is by default. Monthly mean return is also required.
        Works with DataFrame inputs (in math.sqrt is not used).
        """
        return (
            (risk ** 2 + (1 + mean_return) ** 2) ** settings._MONTHS_PER_YEAR
            - (1 + mean_return) ** (settings._MONTHS_PER_YEAR * 2)
        ) ** 0.5

    @staticmethod
    def approx_return_risk_adjusted(mean_return: float, std: float) -> float:
        """
        Approximates geometric mean return given mean return and std.
        """
        return (
            np.exp(
                np.log(1.0 + mean_return) - 0.5 * std ** 2 / (1.0 + mean_return) ** 2
            )
            - 1.0
        )

    @staticmethod
    def get_random_weights(n: int, w_shape: int) -> pd.Series:
        """
        Produce N random normalized weights of a given shape.
        """
        # Random weights
        rand_nos = np.random.rand(n, w_shape)
        weights_transposed = rand_nos.transpose() / rand_nos.sum(axis=1)
        weights = weights_transposed.transpose()
        weights_df = pd.DataFrame(weights)
        return weights_df.aggregate(
            np.array, axis=1
        )  # Converts df to DataFrame of np.array

    @staticmethod
    def get_purchasing_power(inflation: float, value: float = 1000.0):
        return value / (1.0 + inflation)


class Frame:
    """
    Group of methods using DataFrame (or Series) inputs.
    """

    # Rate of return metrics

    @staticmethod
    def weights_sum_is_one(weights: list) -> None:
        if np.around(np.sum(weights), decimals=3) != 1.0:
            raise ValueError("Weights sum is not equal to one.")
        if any(x < 0 for x in weights):
            raise ValueError("Negative weights are not allowed.")

    @staticmethod
    def change_period_to_month(ts: pd.Series) -> pd.Series:
        """
        Change time series period from day to month.
        """
        return ts.resample('M').last()

    @classmethod
    def get_portfolio_return_ts(cls, weights: list, ror: pd.DataFrame) -> pd.Series:
        """
        Returns the mean return time series given portfolio weights and the DataFrame of assets mean returns.
        """
        cls.weights_sum_is_one(weights)
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror
        return ror @ weights

    @classmethod
    def get_portfolio_mean_return(cls, weights: Union[list, np.array], ror: pd.DataFrame) -> float:
        """
        Computes mean return of a portfolio (monthly).
        """
        # cls.weights_sum_is_one(weights)
        weights = np.asarray(weights)
        if isinstance(ror.mean(), float):  # required for a single asset portfolio
            return ror.mean()
        return weights.T @ ror.mean()

    @staticmethod
    def get_cagr(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
        """
        Return Compound Annual Rate of Return (CAGR) for each asset given returns time series DataFrame.
        """
        if ror.shape[0] < 12:
            return pd.Series(
                {x: None for x in ror.columns}
            )  # CAGR is not defined for time periods < 1 year
        return ((ror + 1.0).prod()) ** (settings._MONTHS_PER_YEAR / ror.shape[0]) - 1.0

    @staticmethod
    def get_cumulative_return(
        ror: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.Series, float]:
        """
        Return Compound Return for time series of return (one or several).
        """
        return (ror + 1.0).prod() - 1.0

    @staticmethod
    def get_rolling_fn(
        ror: Union[pd.DataFrame, pd.Series],
        window: int,
        fn: Callable,
        window_below_year: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        check_rolling_window(
            window=window, ror=ror, window_below_year=window_below_year
        )
        x = ror.rolling(window).apply(fn)
        return x.dropna()

    @staticmethod
    def get_annual_return_ts_from_monthly(
        ror_monthly: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Annual Rate of Returns time series from monthly data.
        """
        ts = ror_monthly.resample("A").apply(lambda x: np.prod(x + 1.0) - 1)
        if isinstance(ts, pd.Series):
            ts.rename(ror_monthly.name, inplace=True)
        return ts

    @staticmethod
    def get_wealth_indexes(
        ror: Union[pd.Series, pd.DataFrame], first_value: float = 1000.0
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns wealth indexes for a list of assets (or for portfolio).
        Works also with pd.Series inputs.
        """
        initial_investments = first_value
        first_date = ror.index[0]
        wealth_index = initial_investments * (ror + 1).cumprod()
        wealth_index.loc[
            first_date
        ] = initial_investments  # replaces NaN with the first period return
        wealth_index.sort_index(ascending=True, inplace=True)
        return wealth_index

    @staticmethod
    def _adjust_rates(rates: pd.Series, symbol: str) -> pd.Series:
        """
        Makes several adjustment to the rates before OKID index is computed:
        - penalty for bank license revocation (except TOP 10 banks)
        - monthly interest capitalization adjustment
        """
        if symbol != "RUS_RUB_TOP10.RATE":
            #  license revocation does not happen among TOP 10 banks
            penalty1 = (
                14 / 365 / 2
            )  # penalty for license revocation (period 1 - frequent revocation)
            penalty2 = (14 / 365) * (
                30 / 440
            )  # penalty for license revocation (period 2 - rare revocation)
            rates["28.09.2018":] = rates["28.09.2018":] * (1 - penalty1)
            rates[:"28.10.2018"] = rates[:"28.10.2018"] * (1 - penalty2)
        # compensate monthly interest capitalization (okid index has it)
        rates = ((rates + 1.0) ** (1 / settings._MONTHS_PER_YEAR) - 1) * 12.0
        return rates

    @staticmethod
    def get_okid_index(rates: pd.Series, symbol: str) -> pd.Series:
        """
        Computes OKID index from bank rates time series.
        The index is a basket of 12 one year bank deposits (1 month shift).
        symbol is required for rates time series adjustments (for 'RUS_RUB_TOP10.RATE' it's a different adjustment)
        """
        index_total = None
        start_period = pd.Period(rates.index[0], freq="M")
        end_period = pd.Period(rates.index[-1], freq="M")
        rates = Frame._adjust_rates(rates, symbol)
        for month_idx in range(settings._MONTHS_PER_YEAR):
            rates_yearly = rates.values[month_idx::settings._MONTHS_PER_YEAR]
            index_part = np.repeat(rates_yearly, settings._MONTHS_PER_YEAR)[
                : len(rates) - month_idx
            ]
            index_part = (1 + index_part / settings._MONTHS_PER_YEAR).cumprod()
            index_total = (
                index_part if index_total is None else index_total[1:] + index_part
            )
            start_period += 1
        if index_total is None:
            raise ValueError("`index_total` should not be `None`")
        result = index_total
        result = (np.diff(result) / result[:-1] + 1.0).cumprod() - 1.0
        result = [0] + result
        result = (result + 1) * 100
        result = pd.Series(
            result, index=pd.period_range(start_period, end_period, freq="M")
        )
        result.loc[start_period - 1] = 100.0
        result.sort_index(ascending=True, inplace=True)
        return result

    # Risk metrics

    @classmethod
    def get_portfolio_risk(cls, weights: Union[list, np.array], assets_ror: pd.DataFrame) -> float:
        """
        Compute the standard deviation of return for monthly rebalanced portfolio.
        """
        # cls.weights_sum_is_one(weights)
        if isinstance(assets_ror, pd.Series):  # required for a single asset portfolio
            return assets_ror.std()
        weights = np.array(weights)
        covmat = assets_ror.cov()
        return math.sqrt(weights.T @ covmat @ weights)

    @staticmethod
    def get_semideviation(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
        """
        Returns semideviation.
        """
        below_mean = ror < ror.std(ddof=0)
        return ror[below_mean].std(ddof=0)

    @staticmethod
    def get_below_target_semideviation(ror: Union[pd.DataFrame, pd.Series], t_return: float = 0) -> Union[pd.Series, float]:
        """
        Returns below target semideviation.
        """
        below_target = ror < t_return
        return ror[below_target].std(ddof=0)

    @staticmethod
    def get_var_historic(
        ror: Union[pd.DataFrame, pd.Series], level: int = 5
    ) -> Union[pd.Series, float]:
        """
        Compute monthly historic Value at Risk (VaR) at a specified level.
        """
        s = -ror.quantile(level / 100)
        if isinstance(s, pd.Series):
            s.name = "VaR"
        return s

    @staticmethod
    def get_cvar_historic(
        ror: Union[pd.DataFrame, pd.Series], level: int = 5
    ) -> Union[pd.Series, float]:
        """
        Compute the Conditional VaR (CVaR) at a specified level.
        """
        is_beyond = ror <= ror.quantile(
            level / 100
        )  # mask: return is less than quantile
        s = -ror[is_beyond].mean()
        if isinstance(s, pd.Series):
            s.name = "CVaR"
        return s

    @staticmethod
    def get_drawdowns(
        ror: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Get drawdowns from return time series.
        """
        wealth_index = 1000 * (1 + ror).cumprod()
        previous_peaks = wealth_index.cummax()
        return (wealth_index - previous_peaks) / previous_peaks

    @staticmethod
    def change_columns_order(
        df: pd.DataFrame, selected_columns: list, position: str = "first"
    ) -> pd.DataFrame:
        """
        Places selected_columns on the first position (position='first') or last position (position='last').
        """
        cols = list(df.columns.values)  # Make a list of all of the columns in the df

        def condition(y):
            return y in selected_columns

        cols = [x for x in cols if not condition(x)]  # Remove from list
        # Create new DataFrame with columns in the right order
        if position == "first":
            df = df[selected_columns + cols]
        elif position == "last":
            df = df[cols + selected_columns]
        return df

    @staticmethod
    def skewness(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
        """
        Calculate expanding skewness.
        The shape of time series should be at least 12. In the opposite case empty time series is returned.
        """
        # TODO: implement skewtest (from scipy)
        sk = ror.expanding(min_periods=1).skew()
        return sk.iloc[settings._MONTHS_PER_YEAR:]

    @staticmethod
    def skewness_rolling(
        ror: Union[pd.DataFrame, pd.Series], window: int = 60
    ) -> Union[pd.Series, float]:
        """
        Calculate rolling skewness.
        Window size should be at least 12 months.
        """
        check_rolling_window(window, ror)
        sk = ror.rolling(window=window).skew()
        sk.dropna(inplace=True)
        return sk

    @staticmethod
    def kurtosis(ror: Union[pd.Series, pd.DataFrame]):
        """
        Calculate expanding Fisher (normalized) kurtosis time series.
        Kurtosis should be close to zero for normal distribution.
        """
        kt = ror.expanding(min_periods=1).kurt()
        return kt.iloc[settings._MONTHS_PER_YEAR:]

    @staticmethod
    def kurtosis_rolling(ror: Union[pd.Series, pd.DataFrame], window: int = 60):
        """
        Calculate rolling Fisher (normalized) kurtosis time series.
        Kurtosis should be close to zero for normal distribution.
        Window should be at least 12 months.
        """
        check_rolling_window(window, ror)
        kt = ror.rolling(window=window).kurt()
        return kt.dropna()

    @staticmethod
    def jarque_bera_series(ror: pd.Series) -> dict:
        """
        Jarque-Bera goodness of fit test on a single time series (Pandas Series).
        The Jarque-Bera test shows whether the returns have the skewness and kurtosis matching a normal distribution.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
            Low statistic numbers correspond to normal distribution.
        """
        result_tuple = scipy.stats.jarque_bera(ror)[0], scipy.stats.jarque_bera(ror)[1]
        return {"statistic": result_tuple[0], "p-value": result_tuple[1]}

    @staticmethod
    def jarque_bera_dataframe(ror: pd.DataFrame) -> pd.DataFrame:
        """
        Jarque-Bera goodness of fit test on time series in form of Pandas DataFrame.
        The Jarque-Bera test shows whether the returns have the skewness and kurtosis matching a normal distribution.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
            Low statistic numbers correspond to normal distribution.
        """
        result = ror.apply(scipy.stats.jarque_bera, axis=0)
        return result.set_index(pd.Index(["statistic", "p-value"]))

    @staticmethod
    def kstest_series(ror: pd.Series, distr: str = "norm") -> dict:
        """
        Kolmogorov-Smirnov test goodness of fit test on a single time series (Pandas Series).

        Returns:
            {'statistics': The test statistic, 'p-value': The p-value for the hypothesis test}
        """
        if distr == "norm":
            kstest = scipy.stats.kstest(ror, distr, scipy.stats.norm.fit(ror))
        elif distr == "lognorm":
            kstest = scipy.stats.kstest(ror, distr, scipy.stats.lognorm.fit(ror))
        else:
            raise ValueError('distr should be "norm" (default) or "lognormal".')
        return {"statistic": kstest[0], "p-value": kstest[1]}

    @staticmethod
    def kstest_dataframe(ror: pd.DataFrame, distr: str = "norm") -> pd.DataFrame:
        """
        Kolmogorov-Smirnov test for goodness of fit test on time series in form of Pandas DataFrame.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
        """
        test_dict = {}
        for label, content in ror.items():
            test_values = Frame.kstest_series(content, distr=distr)
            test_dict.update({label: test_values})
        return pd.DataFrame.from_dict(test_dict, orient="columns")


class Rebalance:
    """
    Methods for rebalancing portfolio.
    """

    @staticmethod
    def wealth_ts(
        weights: list, ror: pd.DataFrame, *, period: str = "year"
    ) -> pd.Series:
        """
        Calculate wealth index time series of rebalanced portfolio given returns time series of the assets.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'none'
        """
        # Frame.weights_sum_is_one(weights)
        initial_inv = 1000
        wealth_index = pd.Series(dtype="float64")
        if period == "none":  # Not rebalanced portfolio
            inv_period_spread = np.asarray(weights) * initial_inv
            assets_wealth_indexes = inv_period_spread * (1 + ror).cumprod()
            wealth_index = assets_wealth_indexes.sum(axis=1)
        else:
            for x in ror.resample("Y"):
                df = x[1]  # select ror part of the grouped data
                inv_period_spread = np.asarray(weights) * initial_inv  # rebalancing
                assets_wealth_indexes = inv_period_spread * (1 + df).cumprod()
                wealth_index_local = assets_wealth_indexes.sum(axis=1)
                wealth_index = pd.concat(
                    [wealth_index, wealth_index_local], verify_integrity=True, sort=True
                )
                initial_inv = wealth_index.iloc[-1]
        return wealth_index

    @staticmethod
    def assets_wealth_ts(
            weights: list, ror: pd.DataFrame, *, period: str = "year"
    ) -> pd.DataFrame:
        """
        Calculate ASSETS wealth indexes time series of rebalanced portfolio given returns time series of the assets.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'none'
        """
        # Frame.weights_sum_is_one(weights)
        initial_inv = 1000
        assets_wealth_indexes = pd.DataFrame(dtype="float64")
        wealth_index = pd.Series(dtype="float64")
        if period == "none":  # Not rebalanced portfolio
            inv_period_spread = np.asarray(weights) * initial_inv
            assets_wealth_indexes = inv_period_spread * (1 + ror).cumprod()
        else:
            for x in ror.resample("Y"):
                df = x[1]  # select ror part of the grouped data
                inv_period_spread = np.asarray(weights) * initial_inv  # rebalancing
                assets_wealth_indexes_local = inv_period_spread * (1 + df).cumprod()
                assets_wealth_indexes = pd.concat(
                    [assets_wealth_indexes, assets_wealth_indexes_local],
                    verify_integrity=True, sort=True
                )
                wealth_index_local = assets_wealth_indexes_local.sum(axis=1)
                wealth_index = pd.concat(
                    [wealth_index, wealth_index_local], verify_integrity=True, sort=True
                )
                initial_inv = wealth_index.iloc[-1]
        return assets_wealth_indexes

    @staticmethod
    def assets_weights_ts(
            weights: list, ror: pd.DataFrame, *, period: str = "year"
    ) -> pd.DataFrame:
        """
        Calculate assets weights monthly time series for rebalanced portfolio.

        Parameters
        ----------
        weights
        ror
        period

        Returns
        -------

        """
        assets_wealth_indexes = Rebalance.assets_wealth_ts(weights=weights, ror=ror, period=period)
        portfolio_wealth_index = Rebalance.wealth_ts(weights=weights, ror=ror, period=period)
        return assets_wealth_indexes.divide(portfolio_wealth_index, axis=0)

    @staticmethod
    def return_ts(
        weights: Union[list, np.ndarray], ror: pd.DataFrame, *, period: str = "year"
    ) -> pd.Series:
        """
        Return monthly rate of return time series of rebalanced portfolio given returns time series of the assets.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'none'
        """
        # define data of the first period
        first_date = ror.index[0]
        return_first_period = ror.iloc[0] @ weights

        wealth_index = Rebalance.wealth_ts(
            weights, ror, period=period
        )
        ror = wealth_index.pct_change()
        ror.loc[
            first_date
        ] = return_first_period  # replaces NaN with the first period return
        return ror


class Date:
    @staticmethod
    def subtract_years(dt: pd.Timestamp, years: int) -> pd.Timestamp:
        """
        Subtract N years (integer) from a date. Used for time series.
        First month is +1 (if today is August the series should start at September to give 12 months).
        """
        if isinstance(years, int):
            if dt.month == 12:
                new_dt = dt.replace(year=dt.year - years + 1, month=1)  # for December
            else:
                new_dt = dt.replace(year=dt.year - years, month=dt.month + 1)
        else:
            raise TypeError("The period should be integer")
        return new_dt

    @staticmethod
    def get_difference_in_months(last_day: pd.Timestamp, first_day: pd.Timestamp):
        return last_day.to_period('M') - first_day.to_period('M')


class Index:
    @staticmethod
    def tracking_difference(accumulated_return: pd.DataFrame) -> pd.DataFrame:
        """
        Returns tracking difference for a rate of return time series.
        Assets are compared with the index or another benchmark.
        Index should be in the first position (first column).
        """
        if accumulated_return.shape[1] < 2:
            raise ValueError(
                "At least 2 symbols should be provided to calculate Tracking Difference."
            )
        initial_value = accumulated_return.iloc[0]
        difference = (
            accumulated_return.subtract(accumulated_return.iloc[:, 0], axis=0)
            / initial_value
        )
        difference.drop(
            difference.columns[0], axis=1, inplace=True
        )  # drop the first column (stock index data)
        return difference

    @staticmethod
    def tracking_difference_annualized(tracking_diff: pd.DataFrame) -> pd.DataFrame:
        """
        Annualize the values of tracking difference time series.
        Annual values are available for periods of more than 12 months.
        Returns for less than 12 months can't be annualized.
        """
        pwr = 12 / (1.0 + np.arange(tracking_diff.shape[0]))
        y = abs(tracking_diff)
        diff = (y + 1.0).pow(pwr, axis=0) - 1.0
        diff = np.sign(tracking_diff) * diff
        return diff.iloc[
            settings._MONTHS_PER_YEAR - 1 :
        ]  # returns for the first 11 months can't be annualized

    @staticmethod
    def tracking_error(ror: pd.DataFrame) -> pd.DataFrame:
        """
        Returns tracking error for a rate of return time series.
        Assets are compared with the index or another benchmark.
        Index should be in the first position (first column).
        """
        if ror.shape[1] < 2:
            raise ValueError(
                "At least 2 symbols should be provided to calculate Tracking Error."
            )
        cumsum = ror.subtract(ror.iloc[:, 0], axis=0).pow(2, axis=0).cumsum()
        cumsum.drop(
            cumsum.columns[0], axis=1, inplace=True
        )  # drop the first column (stock index data)
        tracking_error = cumsum.divide((1.0 + np.arange(ror.shape[0])), axis=0).pow(
            0.5, axis=0
        )
        return tracking_error * np.sqrt(12)

    @staticmethod
    def cov_cor(ror: pd.DataFrame, fn: str) -> pd.DataFrame:
        """
        Returns the accumulated correlation or covariance time series.
        The period should be at least 12 months.
        """
        if ror.shape[1] < 2:
            raise ValueError("At least 2 symbols should be provided.")
        if fn not in ["cov", "corr"]:
            raise ValueError("fn should be corr or cov")
        cov_matrix_ts = getattr(ror.expanding(), fn)()
        cov_matrix_ts = cov_matrix_ts.drop(index=ror.columns[1:], level=1).droplevel(1)
        cov_matrix_ts.drop(columns=ror.columns[0], inplace=True)
        return cov_matrix_ts.iloc[settings._MONTHS_PER_YEAR:]

    @staticmethod
    def rolling_cov_cor(
        ror: pd.DataFrame, window: int = 60, fn: str = "corr"
    ) -> pd.DataFrame:
        """
        Returns the rolling correlation (or covariance) time series.
        The history period should be at least 12 months.
        """
        if ror.shape[1] < 2:
            raise ValueError("At least 2 symbols should be provided.")
        if fn not in ["cov", "corr"]:
            raise ValueError("fn should be corr or cov")
        check_rolling_window(window, ror)
        cov_matrix_ts = getattr(ror.rolling(window=window), fn)()
        cov_matrix_ts = cov_matrix_ts.drop(index=ror.columns[1:], level=1).droplevel(1)
        cov_matrix_ts.drop(columns=ror.columns[0], inplace=True)
        cov_matrix_ts.dropna(inplace=True)
        return cov_matrix_ts

    @staticmethod
    def beta(ror: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate beta coefficient time series.
        Index (or benchmark) should be in the first position (first column).
        The period should be at least 12 months.
        """
        cov = Index.cov_cor(ror, fn="cov")
        var = ror.expanding().var().drop(columns=ror.columns[0])
        var = var[settings._MONTHS_PER_YEAR:]
        return cov / var
