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
def test_init_portfolio_failing():
    with pytest.raises(Exception, match=r'Number of tickers \(2\) should be equal to the weights number \(3\)'):
        ok.Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], weights=[0.1, 0.2, 0.7]).symbols
    with pytest.raises(Exception, match='Weights sum is not equal to one.'):
        ok.Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], weights=[0.1, 0.2]).symbols


@mark.portfolio
@mark.usefixtures('_init_portfolio')
class TestPortfolio:

    def test_ror(self):
        portfolio_sample = pd.read_pickle(data_folder / 'portfolio.pkl')
        assert_series_equal(self.portfolio.returns_ts, portfolio_sample)

    def test_weights(self):
        assert self.portfolio.weights == [0.5, 0.5]

    def test_mean_return(self):
        assert self.portfolio.mean_return_monthly == approx(0.010854, rel=1e-2)
        assert self.portfolio.mean_return_annual == approx(0.138324, rel=1e-2)

    def test_real_mean_return(self):
        assert self.portfolio.real_mean_return == approx(0.05286, rel=1e-2)

    def test_real_cagr(self):
        assert self.portfolio.real_cagr == approx(0.045684, rel=1e-2)
        with pytest.raises(Exception, match="Real Return is not defined. Set inflation=True to calculate."):
            self.portfolio_no_inflation.real_cagr

    testdata = [
        ('YTD', 0.01505),
        (1, 0.1226),
        (None, 0.8642),
    ]

    # input_data - period (tuple[0]), expected - expected value (tuple[1])
    @mark.parametrize("input_data,expected", testdata, ids=["YTD", "1 year", "full period"])
    def test_get_cumulative_return(self, input_data, expected):
        assert self.portfolio.get_cumulative_return(period=input_data) == approx(expected, rel=1e-2)

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

    def test_rebalanced_portfolio_return(self):
        assert self.portfolio.get_rebalanced_portfolio_return_ts().mean() == approx(0.011220, rel=1e-2)
        assert self.portfolio.get_rebalanced_portfolio_return_ts(period='none').mean() == \
               approx(0.01221789515271935, rel=1e-2)

    def test_get_cagr(self):
        values = pd.Series({'portfolio': 0.1303543, 'RUB.INFL': 0.05548082428015655})
        assert_series_equal(self.portfolio.get_cagr(), values, rtol=1e-4)
        assert self.portfolio.get_cagr('YTD').iloc[0] == approx(0.01505, rel=1e-2)

    def test_describe(self):
        description = self.portfolio.describe()
        description_sample = pd.read_pickle(data_folder / 'portfolio_description.pkl')
        assert_frame_equal(description, description_sample)

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