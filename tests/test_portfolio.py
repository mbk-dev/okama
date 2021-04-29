import numpy as np
import pandas as pd
import pytest
from pytest import approx
from pytest import mark
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal

import okama as ok

from .conftest import data_folder


@mark.portfolio
def test_init_portfolio_failing(_init_portfolio_values):
    with pytest.raises(ValueError, match=r'Number of tickers \(2\) should be equal to the weights number \(3\)'):
        _init_portfolio_values['weights'] = [0.1, 0.2, 0.7]
        ok.Portfolio(**_init_portfolio_values)
    with pytest.raises(ValueError, match='Weights sum is not equal to one.'):
        _init_portfolio_values['weights'] = [0.1, 0.2]
        ok.Portfolio(**_init_portfolio_values)


@mark.portfolio
@mark.usefixtures('_init_portfolio')
class TestPortfolio:

    def test_ror(self):
        portfolio_sample = pd.read_pickle(data_folder / 'portfolio.pkl')
        assert_series_equal(self.portfolio.get_returns_ts(), portfolio_sample)

    def test_weights(self):
        assert self.portfolio.weights == [0.5, 0.5]

    @pytest.mark.parametrize(
        "period, expected", [('none', 0.0122), ('month', 0.0108), ('year', 0.0112)]
    )
    def test_get_returns_ts(self, period, expected):
        assert self.portfolio.get_returns_ts(rebalancing_period=period).mean() == approx(expected, rel=1e-2)

    def test_mean_return(self):
        assert self.portfolio.mean_return_monthly == approx(0.010854, rel=1e-2)
        assert self.portfolio.mean_return_annual == approx(0.138324, rel=1e-2)

    def test_real_mean_return(self):
        assert self.portfolio.real_mean_return == approx(0.05286, rel=1e-2)

    def test_get_rolling_cumulative_return(self):
        assert self.portfolio.get_rolling_cumulative_return(window=12).iloc[-1] == approx(0.1226, rel=1e-2)

    def test_dividend_yield(self):
        assert self.portfolio.dividend_yield.iloc[-1, :].sum() == 0

    def test_risk(self):
        assert self.portfolio.risk_monthly == approx(0.035718, rel=1e-2)
        assert self.portfolio.risk_annual == approx(0.139814, rel=1e-2)

    def test_get_var_historic(self):
        assert self.portfolio.get_var_historic(time_frame=1, level=5) == approx(0.03815, rel=1e-2)
        assert self.portfolio.get_var_historic(time_frame=5, level=1) == approx(0.0969, rel=1e-2)

    def test_get_cvar_historic(self):
        assert self.portfolio.get_cvar_historic(time_frame=1, level=5) == approx(0.05016, rel=1e-2)
        assert self.portfolio.get_cvar_historic(time_frame=5, level=1) == approx(0.10762, rel=1e-2)

    def test_get_cagr(self):
        values = pd.Series({'portfolio': 0.1303543, 'RUB.INFL': 0.05548082428015655})
        assert_series_equal(self.portfolio.get_cagr(), values, rtol=1e-4)
        with pytest.raises(TypeError):
            self.portfolio.get_cagr(period='one year')

    cagr_testdata1 = [
        (1, 0.0778),
        (None, 0.0710),
    ]

    @mark.parametrize(
        "input_data,expected",
        cagr_testdata1,
        ids=["1 year", "full period"],
    )
    def test_get_cagr_real(self, input_data, expected):
        assert self.portfolio.get_cagr(period=input_data, real=True).values[0] == approx(expected, rel=1e-2)

    def test_get_cagr_real_no_inflation_exception(self):
        with pytest.raises(Exception):
            self.portfolio_no_inflation.get_cagr(period=1, real=True)

    @mark.parametrize("period, real, expected", [('YTD', False, 0.01505), (1, False, 0.12269), (2, True, 0.1532)])
    def test_cumulative_return(self, period, real, expected):
        assert self.portfolio.get_cumulative_return(period=period, real=real)['portfolio'] == approx(expected, rel=1e-2)

    cumulative_return_fail = [(1.5, False, TypeError), (-1, False, ValueError), (1, True, Exception)]

    @pytest.mark.parametrize("period, real, exception", cumulative_return_fail)
    def test_cumulative_return_error(self, period, real, exception):
        with pytest.raises(exception):
            self.portfolio_no_inflation.get_cumulative_return(period=period, real=real)

    def test_describe_inflation(self):
        description = self.portfolio.describe()
        description_sample = pd.read_pickle(data_folder / 'portfolio_description.pkl')
        assert_frame_equal(description, description_sample)

    def test_describe_no_inflation(self):
        self.portfolio_no_inflation.describe([5, 10])  # one limit should exceed the history

    def test_percentile_from_history(self):
        assert self.portfolio.percentile_from_history(years=1).iloc[-1, :].sum() == approx(0.29723, rel=1e-2)
        with pytest.raises(Exception, match="Time series does not have enough history to forecast. "
                                            "Period length is 0.90 years. At least 2 years are required."):
            self.portfolio_short_history.percentile_from_history(years=1)

    def test_table(self):
        assert_array_equal(self.portfolio.table['ticker'].values, np.array(['RUB.FX', 'MCFTR.INDX']))

    def test_get_rolling_return(self):
        assert self.portfolio.get_rolling_cagr(years=1).iloc[-1] == approx(0.122696, rel=1e-2)

    def test_forecast_monte_carlo_norm_wealth_indexes(self):
        assert self.portfolio.forecast_monte_carlo_wealth_indexes(years=1, n=1000).iloc[-1, :].mean() == approx(2121, rel=1e-1)

    def test_forecast_monte_carlo_percentile_wealth_indexes(self):
        dic = self.portfolio.forecast_wealth(years=1, n=100, percentiles=[50])
        assert dic[50] == approx(2121, rel=1e-1)

    def test_skewness(self):
        assert self.portfolio.skewness.iloc[-1] == approx(2.39731, rel=1e-2)

    def test_rolling_skewness(self):
        assert self.portfolio.skewness_rolling(window=24).iloc[-1] == approx(0.82381, rel=1e-2)

    def test_kurtosis(self):
        assert self.portfolio.kurtosis.iloc[-1] == approx(13.19578, rel=1e-2)

    def test_kurtosis_rolling(self):
        assert self.portfolio.kurtosis_rolling(window=24).iloc[-1] == approx(1.58931, rel=1e-2)

    def test_jarque_bera(self):
        assert self.portfolio.jarque_bera['statistic'] == approx(424.1337, rel=1e-2)
