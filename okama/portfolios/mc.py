from __future__ import annotations

from typing import Optional
import logging

import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
from matplotlib import pyplot as plt

from okama import settings
from okama.common import validators
from okama.portfolios import dcf as dcf
from okama.common.helpers import tails, helpers

# Module-level logger
logger = logging.getLogger(__name__)


class MonteCarlo:
    """
    Monte Carlo simulation parameters for an investment portfolio.

    Parameters
    ----------
    parent : PortfolioDCF
        Parent `PortfolioDCF` instance.

    distribution : {'norm', 'lognorm', 't'}, default 'norm'
        Distribution used to generate random monthly rates of return.

        - 'norm' : Normal distribution.
        - 'lognorm' : Lognormal distribution.
        - 't' : Student's t-distribution.

    distribution_parameters : tuple or None, default None
        Parameters for the selected distribution. The expected tuple structure depends on
        `distribution`:

        - 'norm' : (mu, sigma)
        - 'lognorm' : (shape, loc, scale)
        - 't' : (df, loc, scale)

        Any element can be `None` to indicate it should be inferred from the historical
        returns (for example, `(3, None, None)` fixes `df=3` for the t-distribution and
        estimates `loc` and `scale`).

    period : int, default 25
        Forecast horizon in years.

    mc_number : int, default 100
        Number of random scenarios to generate.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pf = ok.Portfolio(first_date='2015-01', last_date='2024-10')
    >>> pf.dcf.set_mc_parameters(distribution='t', period=10, mc_number=100)
    >>> ind = ok.IndexationStrategy(pf)
    >>> ind.initial_investment = 10_000
    >>> ind.frequency = 'year'
    >>> ind.amount = -1_500
    >>> ind.indexation = 'inflation'
    >>> pf.dcf.cashflow_parameters = ind
    >>> pf.dcf.wealth_index(discounting='fv', include_negative_values=False).plot()
    >>> plt.show()
    """
    # TODO: Change example
    def __init__(
            self,
            parent: dcf.PortfolioDCF,
            distribution: str = 'norm',
            distribution_parameters: Optional[tuple] = None,
            period: int = 25,
            mc_number: int = 100,
    ):
        self.parent = parent
        self._distribution = distribution
        self._distribution_parameters = distribution_parameters
        self._period = period
        self._mc_number = mc_number
        self.ror = self.parent.parent.ror

    def __repr__(self):
        """
        Representation of the MonteCarlo configuration.

        Returns
        -------
        str
            A stringified pandas Series containing key Monte Carlo settings such as
            distribution type, parameters (raw and processed), forecast period, and
            number of scenarios.
        """
        # Limit precision of resolved distribution parameters to two decimals for representation purposes
        resolved_params = tuple(round(p, 2) for p in self.get_parameters_for_distribution())
        dic = {
            "Portfolio symbol": self.parent.parent.symbol,
            "Monte Carlo distribution": self.distribution,
            "Distribution parameters": self.distribution_parameters,
            "Distribution parameters after resolving": resolved_params,
            "Monte Carlo period": self.period,
            "Monte Carlo number": self.mc_number,
        }
        return repr(pd.Series(dic))

    @property
    def distribution(self) -> str:
        """
        Distribution used to generate random monthly rates of return.

        Allowed values:

        - 'norm' : Normal distribution.
        - 'lognorm' : Lognormal distribution.
        - 't' : Student's t-distribution.

        Returns
        -------
        str
        """
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        validators.validate_distribution(distribution)
        self._clear_cf_cache()
        self._distribution = distribution

    @property
    def distribution_parameters(self) -> tuple:
        """
        Distribution parameters provided by the user for the selected distribution.

        The expected tuple structure depends on the current distribution:
        - 'norm' : (mu, sigma)
        - 'lognorm' : (shape, loc, scale)
        - 't' : (df, loc, scale)

        Any element can be `None` to indicate it should be inferred from historical returns.

        Returns
        -------
        tuple or None
            Raw distribution parameters as configured. May contain `None` values.
        """
        return self._distribution_parameters

    @distribution_parameters.setter
    def distribution_parameters(self, parameters):
        validators.validate_distribution_parameters(self.distribution, parameters)
        self._clear_cf_cache()
        self._distribution_parameters = parameters

    @property
    def period(self) -> int:
        """
        Forecast period in years for portfolio wealth index time series.

        Returns
        -------
        int
        """
        return self._period

    @period.setter
    def period(self, period):
        validators.validate_integer("period", period)
        self._clear_cf_cache()
        self._period = period

    @property
    def mc_number(self) -> int:
        """
        Number of random scenarios to generate with Monte Carlo simulation.

        Returns
        -------
        int
        """
        return self._mc_number

    @mc_number.setter
    def mc_number(self, mc_number):
        validators.validate_integer("mc_number", mc_number)
        self._clear_cf_cache()
        self._mc_number = mc_number

    def _clear_cf_cache(self):
        self.parent._monte_carlo_wealth_fv = pd.DataFrame(dtype=float)
        self.parent._monte_carlo_cash_flow_fv = pd.DataFrame(dtype=float)

    def backtesting_error(self, var_level: int = 5) -> dict:
        """
        Calculate Backtesting Error as the difference between empirical and theoretical
        risk measures (VaR, CVaR) and arithmetic mean.

        Parameters
        ----------
        var_level : int, default 5
            Confidence level in percent for Value-at-Risk (VaR) and Conditional
            Value-at-Risk (CVaR). For example, 5 corresponds to 5% left tail.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - 'delta_arithmetic_mean': float
                Difference between the theoretical and empirical arithmetic mean
                of returns.
            - 'delta_var': float
                Difference between empirical and theoretical Value-at-Risk at the
                specified level.
            - 'delta_cvar': float
                Difference between empirical and theoretical Conditional
                Value-at-Risk at the specified level.
        """
        parameters = self.get_parameters_for_distribution()
        # Var CVaR
        var_theor = tails.var_theoretical(distr=self.distribution, alpha=var_level / 100, args=parameters)
        cvar_theor = tails.cvar_theoretical(distr=self.distribution, alpha=var_level / 100, args=parameters)
        var_emp = - helpers.Frame.get_var_historic(ror=self.ror, level=var_level)
        cvar_emp = - helpers.Frame.get_cvar_historic(ror=self.ror, level=var_level)
        delta_var = var_emp - var_theor
        delta_cvar = cvar_emp - cvar_theor
        # Arithmetic mean
        parameters = self.get_parameters_for_distribution()
        match self.distribution:
            case "norm":
                mean_t = parameters[0]
            case "lognorm":
                mean_t = scipy.stats.lognorm.mean(*parameters)
            case "t":
                mean_t = parameters[1]
            case _:
                raise ValueError("Unknown distribution: " + self.distribution)
        delta_arithmetic_mean = mean_t - self.ror.mean()
        return {
            "delta_arithmetic_mean": float(delta_arithmetic_mean),
            "delta_var": float(delta_var),
            "delta_cvar": float(delta_cvar)
        }

    def optimize_df_for_students(self, var_level: int) -> float:
        """
        Find degrees of freedom for the t-distribution that best match empirical VaR and CVaR.

        The method minimizes the squared error between theoretical and empirical VaR/CVaR
        using `scipy.optimize.minimize_scalar` with bounds (2.1, 50).

        Parameters
        ----------
        var_level : int
            Confidence level in percent for Value-at-Risk (VaR) and Conditional Value-at-Risk
            (CVaR). Must be in [1, 99].

        Returns
        -------
        float
            Estimated degrees of freedom for Student's t-distribution.

        Raises
        ------
        ValueError
            If `var_level` is outside [1, 99].

        """
        if not var_level in range(1, 100):
            raise ValueError("var_level must be in [1, 99]")
        _, loc, scale = self._get_params_for_t()
        var_emp = - helpers.Frame.get_var_historic(ror=self.ror, level=var_level)
        cvar_emp = - helpers.Frame.get_cvar_historic(ror=self.ror, level=var_level)
        def loss(df):
            var_theor = tails.var_t(alpha=var_level / 100, v=df, loc=loc, scale=scale)
            cvar_theor = tails.cvar_t(alpha=var_level / 100, v=df, loc=loc, scale=scale)
            return (var_theor - var_emp) ** 2 + (cvar_theor - cvar_emp) ** 2
        res = scipy.optimize.minimize_scalar(loss, bounds=(2.1, 50), method="bounded")
        return float(res.x)

    def _get_params_for_t(self) -> tuple[float, float, float]:
        parameters = self.distribution_parameters
        if parameters is None or all(x is None for x in parameters):
            v, loc, scale = scipy.stats.t.fit(self.ror)
        else:
            if None in parameters:
                v, loc, scale = scipy.stats.t.fit(self.ror)
                v = parameters[0] if parameters[0] is not None else v
                loc = parameters[1] if parameters[1] is not None else loc
                scale = parameters[2] if parameters[2] is not None else scale
            else:
                v, loc, scale = parameters
        return float(v), float(loc), float(scale)

    def _get_params_for_lognormal(self) -> tuple[float, float, float]:
        parameters = self.distribution_parameters
        if parameters is None or all(x is None for x in parameters):
            # Fit lognormal to r with loc fixed at -1 so support is (-1, infinity)
            shape, _, scale = scipy.stats.lognorm.fit(self.ror, floc=-1.0)
        else:
            if None in parameters:
                shape, _, scale = scipy.stats.lognorm.fit(self.ror, floc=-1.0)
                shape = parameters[0] if parameters[0] is not None else shape
                scale = parameters[2] if parameters[2] is not None else scale
            else:
                shape, loc, scale = parameters
        return float(shape), -1.0, float(scale)

    def _get_params_for_normal(self) -> tuple[float, float]:
        parameters = self.distribution_parameters
        if parameters is None or all(x is None for x in parameters):
            mu, std = self.ror.mean(), self.ror.std()
        else:
            if None in parameters:
                mu, std = self.ror.mean(), self.ror.std()
                mu = parameters[0] if parameters[0] is not None else mu
                std = parameters[1] if parameters[1] is not None else std
            else:
                mu, std = parameters
        return float(mu), float(std)

    def get_parameters_for_distribution(self) -> tuple[float, ...]:
        """
        Resolve and return fully specified parameters for the current distribution.

        This method combines user-provided parameters (which may contain None values)
        with parameters estimated from the historical returns to produce a complete
        set of arguments for the selected distribution.

        Returns
        -------
        tuple
            A tuple of finalized parameters suitable for random variate generation
            and density/quantile calculations. The structure depends on the
            distribution:

            - 'norm': (mu, sigma)
            - 'lognorm': (shape, loc, scale) where loc is fixed at -1
            - 't': (df, loc, scale)

        Raises
        ------
        ValueError
            If the distribution is unknown.
        """
        match self.distribution:
            case "norm":
                return self._get_params_for_normal()
            case "lognorm":
                return self._get_params_for_lognormal()
            case "t":
                return self._get_params_for_t()
            case _:
                raise ValueError("Unknown distribution: " + self.distribution)

    @property
    def monte_carlo_returns_ts(self) -> pd.DataFrame:
        """
        Generate portfolio monthly rate of return time series with Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series with a given distribution.
        Forecast period should not exceed 1/2 of portfolio history period length.

        First date of forecaseted returns is portfolio last_date.

        Returns
        -------
        DataFrame
            Table with n random rate of return monthly time series.

        Examples
        --------
        >>> pf = ok.Portfolio(
        ...     ['SPY.US', 'AGG.US', 'GLD.US'],
        ...     weights=[.60, .35, .05],
        ...     rebalancing_strategy=ok.Rebalance(period="month"),
        ... )
        >>> pf.dcf.set_mc_parameters(period=8, mc_number=5000)
        >>> pf.dcf.mc.monte_carlo_returns_ts
                         0         1         2     ...      4997      4998      4999
        2021-07 -0.008383 -0.013167 -0.031659  ...  0.046717  0.065675  0.017933
        2021-08  0.038773 -0.023627  0.039208  ... -0.016075  0.034439  0.001856
        2021-09  0.005026 -0.007195 -0.003300  ... -0.041591  0.021173  0.114225
        2021-10 -0.007257  0.003013 -0.004958  ...  0.037057 -0.009689 -0.003242
        2021-11 -0.005006  0.007090  0.020741  ...  0.026509 -0.023554  0.010271
                   ...       ...       ...  ...       ...       ...       ...
        2029-02 -0.065898 -0.003673  0.001198  ...  0.039293  0.015963 -0.050704
        2029-03  0.021215  0.008783 -0.017003  ...  0.035144  0.002169  0.015055
        2029-04  0.002454 -0.016281  0.017004  ...  0.032535  0.027196 -0.029475
        2029-05  0.011206  0.023396 -0.013757  ... -0.044717 -0.025613 -0.002066
        2029-06 -0.016740 -0.007955  0.002862  ... -0.027956 -0.012339  0.048974
        [96 rows x 5000 columns]
        """
        period_months, ts_index = self._forecast_preparation()
        parameters = self.get_parameters_for_distribution()
        match self.distribution:
            case "norm":
                random_returns = np.random.normal(parameters[0], parameters[1], (period_months, self.mc_number))
            case "lognorm":
                random_returns = scipy.stats.lognorm(parameters[0], loc=parameters[1], scale=parameters[2]).rvs(size=[period_months, self.mc_number])
            case "t":
                random_returns = scipy.stats.t(df=parameters[0], loc=parameters[1], scale=parameters[2]).rvs(size=[period_months, self.mc_number])
            case _:
                raise ValueError('Unknown distribution type.')
        return pd.DataFrame(data=random_returns, index=ts_index)

    def _forecast_preparation(self) -> tuple[int, pd.DatetimeIndex]:
        period_months = self.period * settings._MONTHS_PER_YEAR
        # make periods index where the shape is max_period
        start_period = self.parent.parent.last_date.to_period("M")
        end_period = self.parent.parent.last_date.to_period("M") + period_months - 1
        ts_index = pd.period_range(start_period, end_period, freq="M")
        return period_months, ts_index

    def _get_cagr_distribution(self) -> pd.Series:
        """
        Generate CAGR distribution for the rate of return distribution.

        CAGR is calculated for each of n random returns time series.
        """
        return_ts = self.monte_carlo_returns_ts
        return helpers.Frame.get_cagr(return_ts)

    def percentile_distribution_cagr(
        self,
        percentiles: list[int] = [10, 50, 90],
    ) -> dict[int, float]:
        """
        Calculate percentiles for the simulated CAGR distribution.

        CAGR (Compound Annual Growth Rate) is calculated for each Monte Carlo return path.

        Parameters
        ----------
        percentiles : list[int], default [10, 50, 90]
            Percentiles to compute (0-100).

        Returns
        -------
        dict[int, float]
            Mapping `{percentile: value}`.

        Examples
        --------
        >>> pf = ok.Portfolio(
        ...     ['SPY.US', 'AGG.US', 'GLD.US'],
        ...     weights=[.60, .35, .05],
        ...     rebalancing_strategy=ok.Rebalance(period="year"),
        ... )
        >>> pf.dcf.set_mc_parameters(distribution='norm', period=1)
        >>> pf.dcf.mc.percentile_distribution_cagr()
        {10: ..., 50: ..., 90: ...}
        >>> pf.dcf.set_mc_parameters(period=5)
        >>> pf.dcf.mc.percentile_distribution_cagr([5, 10, 20])
        {5: ..., 10: ..., 20: ...}
        """
        cagr_distr = self._get_cagr_distribution()
        results = {}
        for percentile in percentiles:
            value = cagr_distr.quantile(percentile / 100)
            results.update({percentile: value})
        return results

    def percentile_inverse_cagr(
        self,
        score: float = 0,
    ) -> float:
        """
        Compute the percentile rank of a CAGR value within the simulated distribution.

        The percentile rank is calculated from the Monte Carlo CAGR distribution produced by
        the current Monte Carlo settings.

        For example, if the percentile rank for `score=0` is 8 for a 1-year horizon, it means
        that 8% of simulated CAGR values are negative over 1-year periods.

        Parameters
        ----------
        score : float, default 0
            CAGR value to evaluate.

        Returns
        -------
        float
            Percentile rank (0-100).

        Examples
        --------
        >>> pf = ok.Portfolio(
        ...     ['SPY.US', 'AGG.US', 'GLD.US'],
        ...     weights=[.60, .35, .05],
        ...     rebalancing_strategy=ok.Rebalance(period="year"),
        ... )
        >>> pf.dcf.set_mc_parameters(distribution='lognorm', period=1, mc_number=5000)
        >>> pf.dcf.mc.percentile_inverse_cagr(score=0)
        ...
        The probability of getting negative result (score=0) in 1 year period for lognormal distribution.
        """
        cagr_distr = self._get_cagr_distribution()
        return scipy.stats.percentileofscore(cagr_distr, score, kind="rank")

    # historical distribution properties
    @property
    def skewness(self) -> pd.Series:
        """
        Compute expanding skewness time series for portfolio rate of return.

        For normally distributed data, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        Returns
        -------
        Series
            Rolling skewness time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.dcf.mc.skewness
        Date
        2008-05   -0.134193
        2008-06   -0.022349
        2008-07    0.081412
        2008-08   -0.020978
                    ...
        2021-04    0.441430
        2021-05    0.445772
        2021-06    0.437383
        2021-07    0.425247
        Freq: M, Name: portfolio_8378.PF, Length: 159, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.dcf.mc.skewness.plot()
        >>> plt.show()
        """
        return helpers.Frame.skewness(self.ror)

    def skewness_rolling(self, window: int = 60):
        """
        Compute rolling skewness of the return time series.

        For normally distributed rate of return, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        Parameters
        ----------
        window : int, default 60
            Size of the moving window in months.
            The window size should be at least 12 months.

        Returns
        -------
        Series
            Expanding skewness time series

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.dcf.mc.skewness_rolling(window=12*10)
        Date
        2017-04    0.464916
        2017-05    0.446095
        2017-06    0.441211
        2017-07    0.453947
        2017-08    0.464805
        ...
        2021-02    0.007622
        2021-03    0.000775
        2021-04    0.002308
        2021-05    0.022543
        2021-06   -0.006534
        2021-07   -0.012192
        Freq: M, Name: portfolio_8378.PF, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.dcf.mc.skewness_rolling(window=12*10).plot()
        >>> plt.show()
        """
        return helpers.Frame.skewness_rolling(self.ror, window=window)

    @property
    def kurtosis(self):
        """
        Calculate expanding Fisher (normalized) kurtosis time series for portfolio rate of return.

        Kurtosis is a measure of whether the rate of return are heavy-tailed or light-tailed
        relative to a normal distribution.
        It should be close to zero for normally distributed rate of return.
        Kurtosis is the fourth central moment divided by the square of the variance.

        Returns
        -------
        Series
            Expanding kurtosis time series

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.dcf.mc.kurtosis
        Date
        2008-05   -0.815206
        2008-06   -0.718330
        2008-07   -0.610741
        2008-08   -0.534105
                    ...
        2021-04    2.821322
        2021-05    2.855267
        2021-06    2.864717
        2021-07    2.850407
        Freq: M, Name: portfolio_4411.PF, Length: 159, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.dcf.mc.kurtosis.plot()
        >>> plt.show()
        """
        return helpers.Frame.kurtosis(self.ror)

    def kurtosis_rolling(self, window: int = 60):
        """
        Calculate rolling Fisher (normalized) kurtosis time series for portfolio rate of return.

        Kurtosis is a measure of whether the rate of return are heavy-tailed or light-tailed
        relative to a normal distribution.
        It should be close to zero for normally distributed rate of return.
        Kurtosis is the fourth central moment divided by the square of the variance.

        Parameters
        ----------
        window : int, default 60
            Size of the moving window in months.
            The window size should be at least 12 months.

        Returns
        -------
        Series
            Rolling kurtosis time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.dcf.mc.kurtosis_rolling(window=12*10)
        Date
        2017-04    4.041599
        2017-05    4.133518
        2017-06    4.165099
        2017-07    4.205125
        2017-08    4.313773
        ...
        2021-03    0.362184
        2021-04    0.409680
        2021-05    0.455760
        2021-06    0.457315
        2021-07    0.496168
        Freq: M, Name: portfolio_4411.PF, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.dcf.mc.kurtosis_rolling(window=12*10).plot()
        >>> plt.show()
        """
        return helpers.Frame.kurtosis_rolling(self.ror, window=window)

    @property
    def jarque_bera(self) -> dict[str, float]:
        """
        Perform Jarque-Bera test for normality of portfolio returns time series.

        Jarque-Bera shows whether the returns have the skewness and kurtosis
        matching a normal distribution (null hypothesis or H0).

        Returns
        -------
        dict
            Jarque-Bera test statistics and p-value.

        Notes
        -----
        Test returns statistics (first row) and p-value (second row).
        p-value is the probability of obtaining test results, under the assumption that the null hypothesis is correct.
        In general, a large Jarque-Bera statistics and tiny p-value indicate that null hypothesis is rejected
        and the time series are not normally distributed.

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.dcf.mc.jarque_bera
        {'statistic': 58.27670538027455, 'p-value': 2.2148949341271873e-13}
        """
        return helpers.Frame.jarque_bera_series(self.ror)

    @property
    def kstest(self) -> dict[str, float]:
        """
        Perform one sample Kolmogorov-Smirnov test on portfolio returns and evaluate goodness of fit
        for a given distribution.

        The one-sample Kolmogorov-Smirnov test compares the rate of return time series against a given distribution.

        Returns
        -------
        dict
            Kolmogorov-Smirnov test statistics and p-value.

        Notes
        -----
        Like in Jarque-Bera test returns statistic (first row) and p-value (second row).
        Null hypotesis (two distributions are similar) is not rejected when p-value is high enough.
        5% threshold can be used.

        Examples
        --------
        >>> pf = ok.Portfolio(['GLD.US'])
        >>> pf.dcf.set_mc_parameters(distribution='lognorm')
        >>> pf.dcf.mc.kstest
        {'statistic': 0.05001344986084533, 'p-value': 0.6799422889377373}

        >>> pf.dcf.set_mc_parameters(distribution='norm')
        >>> pf.dcf.mc.kstest
        {'statistic': 0.09528000069992831, 'p-value': 0.047761781235967415}

        Kolmogorov-Smirnov test shows that GLD rate of return time series fits lognormal distribution
        better than normal one.
        """
        return helpers.Frame.kstest_series(self.ror, distr=self.distribution)

    @property
    def kstest_for_all_distributions(self) -> pd.DataFrame:
        """
        Run Kolmogorov-Smirnov goodness-of-fit tests for all configured distributions.

        This property evaluates the KS test of the instance's return series against
        each distribution defined in the project's configured list of distributions
        and aggregates the results into a single DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame where the index contains distribution names and each row
            holds the KS test results for that distribution (e.g., test statistic
            and p-value). The exact column labels depend on the underlying test
            implementation.

        See Also
        --------
        kstest : Run the KS test for a single distribution.
        """
        ks_results = []
        for d in settings.distributions:
            ks_results.append(helpers.Frame.kstest_series(self.ror, distr=d))
        return pd.DataFrame(ks_results, index=settings.distributions)

    # Plots
    def plot_qq(
            self,
            var_level: int = 5,
            bootstrap_size_var: int = 2000,
            zoom_to_left_tail: int = 20,
            figsize: Optional[tuple] = None
    ) -> None:
        """
        Generate a quantile-quantile (Q-Q) plot of portfolio monthly rate of return against quantiles of a given
        theoretical distribution.

        A q-q plot is a plot of the quantiles of the portfolio rate of return historical data
        against the quantiles of a given theoretical distribution.

        Bootstrap bands in a Q–Q plot are bootstrap-based confidence envelopes around quantiles that show
        the amount of random sample-to-sample variation one would expect. They bands built by repeatedly resampling
        dataset of a given size and recomputing the Q–Q points.

        Parameters
        ----------
        var_level : int, default 5
            Confidence level in percent for VaR and CVaR.

        bootstrap_size_var : int, default 2000
            Number of bootstrap resamples used to compute confidence intervals for empirical
            VaR and CVaR. If 0, the bootstrap stripe is not drawn. A larger number provides a
            smoother estimate of the confidence bands at the cost of computation time.

        zoom_to_left_tail : int or None, default 20
            Zoom the plot to the left tail by limiting the view to the
            [0.1%, `zoom_to_left_tail`%] percentile range. Must be in [1, 98]. Use `None`
            to show the full range.

        figsize : tuple[float, float], default None
            Figure size in inches (width, height). If `None`, matplotlib default is used.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(
        ...     ['SPY.US', 'AGG.US', 'GLD.US'],
        ...     weights=[.60, .35, .05],
        ...     rebalancing_strategy=ok.Rebalance(period="year"),
        ... )
        >>> pf.dcf.set_mc_parameters(distribution="t")
        >>> pf.dcf.mc.plot_qq(bootstrap_size_var=2000, zoom_to_left_tail=50, figsize=(10, 10))
        >>> plt.show()
        """
        distr = self.distribution
        parameters = self.get_parameters_for_distribution()
        fig, ax = plt.subplots(figsize=figsize)
        alpha = var_level / 100
        if zoom_to_left_tail in range(1, 99):
            p_zoom = zoom_to_left_tail / 100
        elif zoom_to_left_tail is None:
            p_zoom = 1
        else:
            raise ValueError("Zoom level must be between 1 and 99 (or None).")
        var_emp = - helpers.Frame.get_var_historic(ror=self.ror, level=int(alpha * 100))
        cvar_emp = - helpers.Frame.get_cvar_historic(ror=self.ror, level=int(alpha * 100))
        if distr == "norm":
            distargs = ()
            distribution = scipy.stats.norm
            x10_theor = scipy.stats.norm.ppf(p_zoom, loc=parameters[0], scale=parameters[1])
            x001_theor = scipy.stats.norm.ppf(0.001, loc=parameters[0], scale=parameters[1])
            title = f"QQ-plot: Normal mu={parameters[0]:.3f}, sigma={parameters[1]:.3f}"
        elif distr == "lognorm":
            distargs = (parameters[0],)
            distribution = scipy.stats.lognorm
            x10_theor = scipy.stats.lognorm.ppf(p_zoom, s=parameters[0], loc=parameters[1], scale=parameters[2])
            x001_theor = scipy.stats.lognorm.ppf(0.001, s=parameters[0], loc=parameters[1], scale=parameters[2])
            title = f"QQ-plot: Lognormal shape={parameters[0]:.3f}, loc={parameters[1]:.3f}, scale={parameters[2]:.3f}"
        elif distr == "t":
            distargs = (parameters[0],)
            distribution = scipy.stats.t
            x10_theor = scipy.stats.t.ppf(p_zoom, df=parameters[0], loc=parameters[1], scale=parameters[2])
            x001_theor = scipy.stats.t.ppf(0.001, df=parameters[0], loc=parameters[1], scale=parameters[2])
            title = f"QQ-plot: Student's t df={parameters[0]:.3f}, loc={parameters[1]:.3f}, scale={parameters[2]:.3f}"
        else:
            raise ValueError('distr should be "norm" (default), "lognorm" or "t".')
        var_theor = tails.var_theoretical(distr=distr, alpha=alpha, args=parameters)
        cvar_theor = tails.cvar_theoretical(distr=distr, alpha=alpha, args=parameters)
        delta_var = var_emp - var_theor
        delta_cvar = cvar_emp - cvar_theor

        # VaR points
        ax.scatter([var_theor], [var_emp], color='tab:red', s=40, zorder=3, label=f'VaR {alpha:.0%}')
        ax.axvline(var_theor, color='tab:red', ls=':', lw=1.2)
        ax.axhline(var_emp, color='tab:red', ls='--', lw=1.2)

        ax.annotate(f'delta var={delta_var:.4f}',
                    xy=(var_theor, var_emp),
                    xytext=(10, -15),
                    textcoords='offset points',
                    color='tab:red',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='tab:red', alpha=0.7))

        # CVAR points
        ax.scatter([cvar_theor], [cvar_emp], color='yellow', s=40, zorder=3, label=f'CVAR {alpha:.0%}')
        ax.axvline(cvar_theor, color='tab:blue', ls=':', lw=1.2)
        ax.axhline(cvar_emp, color='tab:blue', ls='--', lw=1.2)

        ax.annotate(f'delta cvar={delta_cvar:.4f}',
                    xy=(cvar_theor, cvar_emp),
                    xytext=(10, -15),
                    textcoords='offset points',
                    color='tab:red',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='tab:red', alpha=0.7))

        sm.qqplot(
            self.ror,
            dist=distribution,
            distargs=distargs,
            loc=parameters[0] if distr == "norm" else parameters[1],
            scale=parameters[1] if distr == "norm" else parameters[2],
            fit=False,
            line='45',
            markersize=3,
            alpha=0.7,
            ax=ax
        )
        # BOOTSTRAP VaR & CVaR
        if bootstrap_size_var:
            bootsrap_number = bootstrap_size_var
            rng = np.random.default_rng(seed=11)
            samples = rng.choice(self.ror, size=(bootsrap_number, self.ror.size), replace=True)
            boot_var = np.quantile(samples, alpha, axis=1)
            boot_cvar = np.apply_along_axis(tails.cvar_of_sample, 1, samples, alpha)
            var_ci = np.quantile(boot_var, [0.025, 0.975])
            cvar_ci = np.quantile(boot_cvar, [0.025, 0.975])
            # Stripe 95% CI emp VaR & CVar
            ax.axhspan(var_ci[0], var_ci[1], color='tab:blue', alpha=0.08, label='95% CI Emp-VaR (bootstrap)')
            ax.axhspan(cvar_ci[0], cvar_ci[1], color='tab:green', alpha=0.08, label='95% CI Emp-CVaR (bootstrap)')

        # Zoom to the left tail (large drawdowns)
        if zoom_to_left_tail:
            y10_emp = float(np.quantile(self.ror, p_zoom))
            y001_emp = float(np.quantile(self.ror, 0.001))
            ax.set_xlim(x001_theor, x10_theor)
            ax.set_ylim(y001_emp, y10_emp)


        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.title(title)
        plt.show()
        # Log metrics using f-strings
        logger.info(
            f"VaR  {alpha:.0%}: theor={var_theor:.6f}, emp={var_emp:.6f}, delta(emp-theor)={(var_emp - var_theor):.6f}"
        )
        logger.info(
            f"CVAR {alpha:.0%}: theor={cvar_theor:.6f}, emp={cvar_emp:.6f}, delta(emp-theor)={(cvar_emp - cvar_theor):.6f}"
        )
        # Log bootstrap confidence intervals only if they were computed
        if bootstrap_size_var:
            logger.info(
                f"95% CI empiric VaR (bootstrap): [{var_ci[0]:.6f}, {var_ci[1]:.6f}]"
            )
            logger.info(
                f"95% CI empiric CVaR (bootstrap): [{cvar_ci[0]:.6f}, {cvar_ci[1]:.6f}]"
            )

    def plot_hist_fit(self, bins: int = None) -> None:
        """
        Plot a histogram of historical monthly returns and overlay the fitted theoretical PDF.

        Uses the currently selected distribution (self.distribution) and its resolved
        parameters to draw the probability density function.

        Parameters
        ----------
        bins : int, default None
            Number of histogram bins. If None, matplotlib will choose automatically.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SP500TR.INDX'])
        >>> pf.dcf.set_mc_parameters(distribution='norm')
        >>> pf.dcf.mc.plot_hist_fit()
        >>> plt.show()
        """
        data = self.ror
        # Plot the histogram
        plt.hist(data, bins=bins, density=True, alpha=0.6, color="g")
        # Plot the PDF.Probability Density Function
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        distr = self.distribution
        if distr == "norm":  # Generate PDF
            mu, std = self.get_parameters_for_distribution()
            p = scipy.stats.norm.pdf(x, loc=mu, scale=std)
            title = f"Fit results: mu = {mu:.3f}, std = {std:.3f}"
        elif distr == "lognorm":
            shape, loc, scale = self.get_parameters_for_distribution()  # shape, loc, scale
            mu = np.log(scale)
            p = scipy.stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
            title = f"Fit results: shape = {shape:.3f}, mu = {mu:.3f}, loc = {loc:.3f}"
        elif distr == "t":
            df, loc, scale = self.get_parameters_for_distribution()
            p = scipy.stats.t.pdf(x, loc=loc, scale=scale, df=df)
            title = f"Fit results: df = {df:.3f}, loc = {loc:.3f}, scale = {scale:.3f}"
        else:
            raise ValueError('distr must be "norm" (default) or "lognorm".')
        plt.plot(x, p, "k", linewidth=2)
        plt.title(title)
        plt.show()

