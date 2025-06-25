import math
from typing import Union, Callable, Optional, Tuple
from functools import singledispatchmethod

import pandas as pd
import numpy as np
import scipy.stats

from okama.common.validators import validate_integer, validate_distribution
from okama.common.error import (
    LongRollingWindowLengthError,
    RollingWindowLengthBelowOneYearError,
    ShortPeriodLengthError,
)
from okama.portfolio import CashFlow
from okama import settings


def check_rolling_window(window: int, ror: Union[pd.Series, pd.DataFrame], window_below_year: bool = False):
    validate_integer(arg_name="window", arg_value=window, min_value=0, inclusive=False)
    if not window_below_year and window < settings._MONTHS_PER_YEAR:
        raise RollingWindowLengthBelowOneYearError("window size must be at least 1 year")
    if window > ror.shape[0]:
        raise LongRollingWindowLengthError("window size is more than data history depth: 13 months")


class Float:
    """
    Group of methods using float values inputs.
    Some of them can take DataFrame.
    """

    @staticmethod
    def get_monthly_return_from_annual(annual_return: Union[float, pd.Series]) -> Union[float, pd.Series]:
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
        risk: Union[float, pd.Series, pd.DataFrame], mean_return: Union[float, pd.Series, pd.DataFrame]
    ) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Annualizes Risk.
        Annualization from month to year (from standard deviation) is by default. Monthly mean return is also required.
        Works with DataFrame inputs (in math.sqrt is not used).
        """
        return (
            (risk**2 + (1 + mean_return) ** 2) ** settings._MONTHS_PER_YEAR
            - (1 + mean_return) ** (settings._MONTHS_PER_YEAR * 2)
        ) ** 0.5

    @staticmethod
    def approx_return_risk_adjusted(mean_return: float, std: float) -> float:
        """
        Approximates geometric mean return given mean return and std.
        """
        return np.exp(np.log(1.0 + mean_return) - 0.5 * std**2 / (1.0 + mean_return) ** 2) - 1.0

    @staticmethod
    def get_random_weights(n: int, w_shape: int, bounds: Optional[Tuple[Tuple[float, float], ...]] = None) -> pd.Series:
        """
        Produce N random normalized weights of a given shape using sequential generation.
        bounds : tuple of tuples, optional.
        Constraints for each asset's weight, e.g., ((0, 1), (0, 0.5), (0.5, 1), ...).
        If None, default constraints are applied.
        """
        # Case 1: default bounds
        if bounds is None:
            random_numbers = np.random.rand(n, w_shape)
            # keepdims instead of transpose
            weights = random_numbers / random_numbers.sum(axis=1, keepdims=True)

        # Case 2: custom bounds
        else:
            bounds_arr = np.array(bounds)
            mins = bounds_arr[:, 0]
            maxs = bounds_arr[:, 1]

            weights = []
            batch_size = min(1000, n)

            while len(weights) < n:

                remaining = np.ones(batch_size)
                indices = np.arange(w_shape)
                batch_w = np.zeros((batch_size, w_shape))
                valid_mask = np.ones(batch_size, dtype=bool)

                shuffled_indices = np.tile(indices, (batch_size, 1))
                for i in range(batch_size):
                    np.random.shuffle(shuffled_indices[i])

                for i in range(w_shape - 1):

                    idx = shuffled_indices[:, i]
                    low = mins[idx]
                    high = maxs[idx]

                    future_mins = np.sum(mins[shuffled_indices[:, i + 1 :]], axis=1)
                    future_maxs = np.sum(maxs[shuffled_indices[:, i + 1 :]], axis=1)

                    adjusted_low = np.maximum(low, remaining - future_maxs)
                    adjusted_high = np.minimum(high, remaining - future_mins)

                    rand_vals = np.random.uniform(adjusted_low, adjusted_high)

                    batch_w[np.arange(batch_size), idx] = rand_vals
                    remaining -= rand_vals

                    valid_mask &= adjusted_low <= adjusted_high

                last_idx = shuffled_indices[:, -1]
                batch_w[np.arange(batch_size), last_idx] = remaining
                valid_mask &= (mins[last_idx] <= remaining) & (remaining <= maxs[last_idx])
                valid_mask &= np.all(batch_w >= 0, axis=1)

                valid_weights = batch_w[valid_mask]
                weights.extend(valid_weights.tolist())

                if len(weights) >= n:
                    break

        return pd.Series([np.array(w) for w in weights[:n]])

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
        return ts.resample("M").last()

    @classmethod
    def get_portfolio_return_ts(cls, weights: list, ror: pd.DataFrame) -> pd.Series:
        # sourcery skip: assign-if-exp, reintroduce-else
        """
        Returns the mean return time series given portfolio weights and the DataFrame of assets mean returns.
        """
        # cls.weights_sum_is_one(weights)
        if isinstance(ror, pd.Series):  # required for a single asset portfolio
            return ror
        return ror @ weights

    @classmethod
    def get_portfolio_mean_return(cls, weights: Union[list, np.array], ror: pd.DataFrame) -> float:
        # sourcery skip: assign-if-exp, reintroduce-else
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
            return pd.Series({x: None for x in ror.columns})  # CAGR is not defined for periods < 1 year
        return ((ror + 1.0).prod()) ** (settings._MONTHS_PER_YEAR / ror.shape[0]) - 1.0

    @staticmethod
    def get_cumulative_return(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
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
        """
        Calculate rolling function with a given window.

        'window_below_year' allows to use window size below 12 monthes (periods).
        """
        check_rolling_window(window=window, ror=ror, window_below_year=window_below_year)
        x = ror.rolling(window).apply(fn)
        return x.dropna()

    @staticmethod
    def get_annual_return_ts_from_monthly(
        ror_monthly: Union[pd.DataFrame, pd.Series],
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
        ror: Union[pd.Series, pd.DataFrame], initial_amount: float = 1000.0
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns wealth indexes for a list of assets (or for portfolio).
        Works also with pd.Series inputs.

        The values of the wealth index correspond to the begin of the month.
        """
        initial_investments = initial_amount
        first_date = ror.index[0]
        wealth_index = initial_investments * (ror + 1).cumprod()
        first_wealth_index_date = first_date - 1  # set 1000 to one month earlie
        wealth_index.loc[first_wealth_index_date] = initial_investments
        wealth_index.sort_index(ascending=True, inplace=True)
        return wealth_index

    @staticmethod
    def get_wealth_indexes_with_cashflow(
        ror: Union[pd.Series, pd.DataFrame],
        portfolio_symbol: Optional[str],
        inflation_symbol: Optional[str],
        cashflow_parameters: type[CashFlow],
        use_discounted_values: bool,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns wealth index for a series of returns with cash flows (withdrawals/contributions).

        Values of the wealth index correspond to the beginning of the month.
        """
        pf_object = cashflow_parameters.parent
        dcf_object = cashflow_parameters.parent.dcf
        dcf_object.cashflow_parameters = cashflow_parameters
        amount = getattr(cashflow_parameters, "amount", None)
        period_initial_amount = (
            dcf_object.initial_investment_pv if use_discounted_values else cashflow_parameters.initial_investment
        )
        period_initial_amount_cached = period_initial_amount
        if amount == 0:
            wealth_index = Frame.get_wealth_indexes(ror, period_initial_amount)
        else:
            try:
                # amount is not defined in TimeSeriesStrategy & PercentageStrategy
                amount = dcf_object.cashflow_pv if use_discounted_values else cashflow_parameters.amount
            except AttributeError:
                pass
            if isinstance(ror, pd.DataFrame):
                portfolio_position = ror.columns.get_loc(portfolio_symbol)
            else:
                # for Series
                portfolio_position = 0
                ror = ror.to_frame()
            periods_per_year = settings.frequency_periods_per_year[cashflow_parameters.frequency]
            if cashflow_parameters.frequency == "month" or cashflow_parameters.NAME == "time_series":
                s = pd.Series(dtype=float, name=portfolio_symbol)
                for n, row in enumerate(ror.itertuples()):
                    date = row[0]
                    r = row[portfolio_position + 1]
                    if cashflow_parameters.NAME == "fixed_amount":
                        cashflow = amount * (1 + cashflow_parameters.indexation / settings._MONTHS_PER_YEAR) ** n
                    elif cashflow_parameters.NAME == "fixed_percentage":
                        cashflow = cashflow_parameters.percentage / periods_per_year * period_initial_amount
                    elif cashflow_parameters.NAME == "time_series":
                        try:
                            cashflow = cashflow_parameters.time_series[date]
                            if use_discounted_values:
                                last_date = pf_object.last_date
                                first_date = date.to_timestamp(how="End")
                                period_length = Date.get_period_length(last_date, first_date)
                                rate = dcf_object.discount_rate
                                cashflow = cashflow / (1 + rate) ** period_length
                        except KeyError:
                            cashflow = 0
                    else:
                        raise ValueError("Wrong cashflow strategy name value.")
                    period_initial_amount = period_initial_amount * (r + 1) + cashflow
                    date = row[0]
                    s[date] = period_initial_amount
            else:
                pandas_frequency = settings.frequency_mapping[cashflow_parameters.frequency]
                wealth_df = pd.DataFrame(dtype=float, columns=[portfolio_symbol])
                for n, x in enumerate(ror.resample(rule=pandas_frequency, convention="start")):
                    ror_df = x[1].iloc[:, portfolio_position]  # select ror part of the grouped data
                    period_wealth_index = period_initial_amount * (1 + ror_df).cumprod()
                    if cashflow_parameters.NAME == "fixed_amount":
                        cashflow_value = amount * (1 + cashflow_parameters.indexation / periods_per_year) ** n
                    elif cashflow_parameters.NAME == "fixed_percentage":
                        cashflow_value = cashflow_parameters.percentage / periods_per_year * period_initial_amount
                    else:
                        raise ValueError("Wrong cashflow_method value.")
                    period_final_balance = period_wealth_index.iloc[-1] + cashflow_value
                    period_wealth_index.iloc[-1] = period_final_balance
                    period_initial_amount = period_final_balance
                    wealth_df = pd.concat([None if wealth_df.empty else wealth_df, period_wealth_index], sort=False)
                s = wealth_df.squeeze()
            first_date = s.index[0]
            first_wealth_index_date = first_date - 1  # set first date to one month earlie
            s.loc[first_wealth_index_date] = period_initial_amount_cached
            if inflation_symbol:
                cum_inflation = Frame.get_wealth_indexes(
                    ror=ror.loc[:, inflation_symbol], initial_amount=period_initial_amount_cached
                )
                wealth_index = pd.concat([s, cum_inflation], axis="columns")
            else:
                wealth_index = s
        wealth_index.sort_index(ascending=True, inplace=True)
        return wealth_index

    @singledispatchmethod
    @staticmethod
    def get_survival_date(wealth_series, discount_rate: float, threshold: float = 0):
        raise TypeError("wealth_series must be a pd.Series or pd.DataFrame.")

    @get_survival_date.register
    def _(wealth_series: pd.Series, discount_rate: float, threshold: float) -> pd.Timestamp:
        if threshold > 1 or threshold < 0:
            raise ValueError("threshold must be in range from 0 to 1.")
        if threshold:
            fv = (
                wealth_series.iloc[0]
                * pd.Series(1.0 + discount_rate / 12, index=wealth_series.index).shift(1).cumprod()
            )
            fv.iloc[0] = wealth_series.iloc[0]
            condition = wealth_series <= fv * threshold
        else:
            condition = wealth_series <= 0
        try:
            survival_date = wealth_series[condition].index[0]
        except IndexError:
            survival_date = wealth_series.index[-1]
        return survival_date.to_timestamp(freq="M")

    @get_survival_date.register
    def _(wealth: pd.DataFrame, discount_rate: float, threshold: float = 0) -> pd.Timestamp:
        return wealth.apply(func=Frame.get_survival_date, axis=0, args=(discount_rate, threshold))

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
    def get_below_target_semideviation(
        ror: Union[pd.DataFrame, pd.Series], t_return: float = 0
    ) -> Union[pd.Series, float]:
        """
        Returns below target semideviation.
        """
        below_target = ror < t_return
        return ror[below_target].std(ddof=0)

    @staticmethod
    def get_var_historic(ror: Union[pd.DataFrame, pd.Series], level: int = 5) -> Union[pd.Series, float]:
        """
        Compute monthly historic Value at Risk (VaR) at a specified level.
        """
        s = -ror.quantile(level / 100)
        if isinstance(s, pd.Series):
            s.name = "VaR"
        return s

    @staticmethod
    def get_cvar_historic(ror: Union[pd.DataFrame, pd.Series], level: int = 5) -> Union[pd.Series, float]:
        """
        Compute the Conditional VaR (CVaR) at a specified level.
        """
        is_beyond = ror <= ror.quantile(level / 100)  # mask: return is less than quantile
        s = -ror[is_beyond].mean()
        if isinstance(s, pd.Series):
            s.name = "CVaR"
        return s

    @staticmethod
    def get_drawdowns(ror: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Get drawdowns from return time series.
        """
        wealth_index = 1000 * (1 + ror).cumprod()
        previous_peaks = wealth_index.cummax()
        return (wealth_index - previous_peaks) / previous_peaks

    @staticmethod
    def change_columns_order(df: pd.DataFrame, selected_columns: list, position: str = "first") -> pd.DataFrame:
        """
        Places selected_columns on the first position (position='first') or last position (position='last').
        """
        cols = list(df.columns.values)  # Make a list of all columns in the df

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
        sk = ror.expanding(min_periods=1).skew()
        return sk.iloc[settings._MONTHS_PER_YEAR :]

    @staticmethod
    def skewness_rolling(ror: Union[pd.DataFrame, pd.Series], window: int = 60) -> Union[pd.Series, float]:
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
        return kt.iloc[settings._MONTHS_PER_YEAR :]

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
            kstest = scipy.stats.kstest(ror, distr, args=scipy.stats.norm.fit(ror))
        elif distr == "lognorm":
            kstest = scipy.stats.kstest(ror, distr, args=scipy.stats.lognorm.fit(ror))
        elif distr == "t":
            kstest = scipy.stats.kstest(ror, distr, args=scipy.stats.t.fit(ror))
        else:
            raise ValueError('distr should be "norm" (default), "lognormal" or "t".')
        return {"statistic": kstest[0], "p-value": kstest[1]}

    @staticmethod
    def kstest_dataframe(ror: pd.DataFrame, distr: str = "norm") -> pd.DataFrame:
        """
        Kolmogorov-Smirnov test for goodness of fit test on time series in form of Pandas DataFrame.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
        """
        validate_distribution(distr)
        test_dict = {}
        for label, content in ror.items():
            test_values = Frame.kstest_series(content, distr=distr)
            test_dict[label] = test_values
        return pd.DataFrame.from_dict(test_dict, orient="columns")


class Date:
    @staticmethod
    def subtract_years(dt: pd.Timestamp, years: int) -> pd.Timestamp:
        """
        Subtract N years (integer) from a date. Used for time series.
        First month is +1 (if today is August the series should start at September to give 12 months).
        """
        if not isinstance(years, int):
            raise TypeError("The period should be integer")
        return (
            dt.replace(year=dt.year - years + 1, month=1)
            if dt.month == 12
            else dt.replace(year=dt.year - years, month=dt.month + 1)
        )

    @staticmethod
    def subtract_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
        """
        Subtract N months (integer) from a date. Used for time series.
        First month is +1 (if today is August the series should start at September to give 12 months).
        """
        if not isinstance(months, int):
            raise TypeError("The period should be integer")
        return (
            dt.replace(year=dt.year, month=dt.month - months)
            if dt.month > months
            else dt.replace(year=dt.year - 1, month=12 - (months - dt.month))
        )

    @staticmethod
    def get_difference_in_months(last_day: pd.Timestamp, first_day: pd.Timestamp) -> pd.DateOffset:
        return last_day.to_period("M") - first_day.to_period("M")

    @staticmethod
    def get_period_length(last_date: pd.Timestamp, first_date: pd.Timestamp) -> float:
        return round((last_date - first_date) / np.timedelta64(365, "D"), ndigits=1)


class Index:
    @staticmethod
    def tracking_difference(accumulated_return: pd.DataFrame) -> pd.DataFrame:
        """
        Returns tracking difference for a rate of return time series.
        Assets are compared with the index or another benchmark.
        Index should be in the first position (first column).
        """
        if accumulated_return.shape[1] < 2:
            raise ValueError("At least 2 symbols must be provided to calculate Tracking Difference.")
        initial_value = accumulated_return.iloc[0]
        difference = accumulated_return.subtract(accumulated_return.iloc[:, 0], axis=0) / initial_value
        difference.drop(difference.columns[0], axis=1, inplace=True)  # drop the first column (stock index data)
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
        return diff.iloc[settings._MONTHS_PER_YEAR - 1 :]  # returns for the first 11 months can't be annualized

    @staticmethod
    def tracking_error(ror: pd.DataFrame) -> pd.DataFrame:
        """
        Returns tracking error for a rate of return time series.
        Assets are compared with the index or another benchmark.
        Index should be in the first position (first column).
        """
        if ror.shape[1] < 2:
            raise ValueError("At least 2 symbols should be provided to calculate Tracking Error.")
        if ror.shape[0] < 12:
            raise ShortPeriodLengthError("Tracking Error is not defined for time periods < 1 year")
        cumsum = ror.subtract(ror.iloc[:, 0], axis=0).pow(2, axis=0).cumsum()
        cumsum.drop(cumsum.columns[0], axis=1, inplace=True)  # drop the first column (stock index data)
        tracking_error = cumsum.divide((1.0 + np.arange(ror.shape[0])), axis=0).pow(0.5, axis=0)
        return tracking_error * np.sqrt(12)

    @staticmethod
    def expanding_cov_cor(ror: pd.DataFrame, fn: str) -> pd.DataFrame:
        """
        Returns the accumulated expanding correlation or covariance time series.
        The period should be at least 12 months.
        """
        if ror.shape[1] < 2:
            raise ValueError("At least 2 symbols should be provided.")
        if fn not in ["cov", "corr"]:
            raise ValueError("fn should be corr or cov")
        cov_matrix_ts = getattr(ror.expanding(), fn)()
        cov_matrix_ts = cov_matrix_ts.drop(index=ror.columns[1:], level=1).droplevel(1)
        cov_matrix_ts.drop(columns=ror.columns[0], inplace=True)
        return cov_matrix_ts.iloc[settings._MONTHS_PER_YEAR :]

    @staticmethod
    def rolling_cov_cor(ror: pd.DataFrame, window: int = 60, fn: str = "corr") -> pd.DataFrame:
        """
        Returns the rolling correlation (or covariance) time series.
        The history period should be at least 12 months.
        """
        if ror.shape[1] < 2:
            raise ValueError("At least 2 symbols should be provided.")
        if fn not in ["cov", "corr"]:
            raise ValueError("fn should be corr or cov")
        check_rolling_window(window=window, ror=ror, window_below_year=False)
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
        if ror.shape[1] < 2:
            raise ValueError("At least 2 symbols should be provided to calculate beta coefficient.")
        if ror.shape[0] < 12:
            raise ShortPeriodLengthError("Beta coefficient is not defined for time periods < 1 year")
        cov = Index.expanding_cov_cor(ror, fn="cov")
        benchmark_var = ror.loc[:, ror.columns[0]].expanding().var()
        benchmark_var = benchmark_var.iloc[settings._MONTHS_PER_YEAR :]
        return cov.divide(benchmark_var, axis=0)

    @staticmethod
    def rolling_fn(df: pd.DataFrame, window: int, fn: Callable, window_below_year: bool = False) -> pd.DataFrame:
        """
        Calculate the rolling custom function.

        Apply a function to time series DataFrame with the rolling window.
        The window should be in months.
        """
        check_rolling_window(window=window, ror=df, window_below_year=window_below_year)
        output = pd.DataFrame()
        for start_date in df.index:
            end_date = start_date + window
            df_window = df.loc[start_date:end_date, :]
            end_date = df_window.index[-1]
            period_length = end_date - start_date
            if period_length.n < window:
                break
            windows_result = fn(df_window).iloc[-1, :]
            output = pd.concat([output, windows_result.to_frame().T], copy=False)
        return output
