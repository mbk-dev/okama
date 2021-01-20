import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pytest import approx
from pytest import mark
from okama.assets import AssetList
from okama import Portfolio


@mark.asset
@mark.usefixtures('_init_asset')
class TestAsset:

    @mark.smoke
    def test_get_symbol_data(self):
        assert self.spy.name == 'SPDR S&P 500 ETF Trust'
        assert self.spy.country == 'USA'
        assert self.spy.currency == 'USD'
        assert self.spy.type == 'ETF'
        assert self.spy.inflation == 'USD.INFL'
        assert self.spy.first_date == pd.to_datetime('1993-02')

    def test_price(self):
        assert type(self.spy.price) == float

    def test_dividends(self):
        assert self.spy.dividends['2019'].sum() == 5.6183

    def test_nav_ts(self):
        assert self.otkr.nav_ts[0] == 101820352.18


@mark.asset_list
def test_asset_list_init_failing():
    with pytest.raises(Exception, match=r'Symbols should be a list of string values.'):
        AssetList(symbols=('RUB.FX', 'MCFTR.INDX'))


@mark.asset_list
@mark.usefixtures('_init_asset_list')
class TestAssetList:

    def test_ror(self):
        asset_list_sample = pd.read_pickle('data/asset_list.pkl')
        asset_list_lt_sample = pd.read_pickle('data/asset_list_lt.pkl')
        currencies_sample = pd.read_pickle('data/currencies.pkl')
        real_estate_sample = pd.read_pickle('data/real_estate.pkl')
        spy_sample = pd.read_pickle('data/spy.pkl')
        assert_frame_equal(self.asset_list.ror, asset_list_sample)
        assert_frame_equal(self.asset_list_lt.ror, asset_list_lt_sample)
        assert_frame_equal(self.currencies.ror, currencies_sample)
        assert_frame_equal(self.real_estate.ror, real_estate_sample)
        assert_frame_equal(self.spy.ror, spy_sample)

    def test_currencies(self):
        assert self.currencies.pl.years == 1
        assert self.currencies.first_date == pd.to_datetime('2019-01')
        assert self.currencies.currencies == \
               {'RUBUSD.FX': 'USD', 'EURUSD.FX': 'USD', 'CNYUSD.FX': 'USD', 'asset list': 'USD'}
        assert self.currencies.names == {'RUBUSD.FX': 'RUBUSD', 'EURUSD.FX': 'EURUSD', 'CNYUSD.FX': 'CNYUSD'}
        assert self.currencies.describe().iloc[1, -1] == approx(0.02485, rel=1e-2)

    @mark.smoke
    def test_make_asset_list(self):
        assert self.asset_list.last_date == pd.to_datetime('2020-01')
        assert list(self.asset_list.ror) == ['RUB.FX', 'MCFTR.INDX']

    def test_calculate_wealth_indexes(self):
        assert self.asset_list.wealth_indexes.sum(axis=1)[-1] == \
               approx(3339.677963676333, rel=1e-2)  # last month indexes sum

    def test_risk(self):
        assert self.asset_list.risk_monthly['RUB.FX'] == approx(0.0258, rel=1e-2)
        assert self.asset_list.risk_monthly['MCFTR.INDX'] == approx(0.0264, rel=1e-2)
        assert self.asset_list.risk_annual['RUB.FX'] == approx(0.0825, rel=1e-2)
        assert self.asset_list.risk_annual['MCFTR.INDX'] == approx(0.1222, rel=1e-2)

    def test_semideviation_monthly(self):
        assert self.asset_list.semideviation_monthly.sum() == approx(0.015614, rel=1e-2)

    def test_get_var(self):
        assert self.asset_list.get_var_historic(level=5).sum() == approx(0.04664, rel=1e-2)

    def test_get_cvar(self):
        assert self.asset_list.get_cvar_historic(level=5).sum() == approx(0.0659, rel=1e-2)

    def test_drawdowns(self):
        assert self.asset_list.drawdowns.min().sum() == approx(-0.082932, rel=1e-2)

    testdata = [
        ('YTD', 0.0183, 0.0118, 0.0405),
        (1, -0.0463, 0.3131, 0.2975),
        (None, -0.0888, 0.3651, 0.1617),
    ]

    # input_data - period (tuple[0]), expected1 - expected value 1 (tuple[1]), expected2 - expected value 2(tuple[2])
    @mark.parametrize("input_data,expected1,expected2,expected3", testdata, ids=["YTD", "1 year", "full period"])
    def test_get_cagr(self, input_data, expected1, expected2, expected3):
        assert self.asset_list.get_cagr(period=input_data)['RUB.FX'] == approx(expected1, rel=1e-2)
        assert self.asset_list.get_cagr(period=input_data)['MCFTR.INDX'] == approx(expected2, rel=1e-2)
        assert self.real_estate.get_cagr(period=input_data).sum() == approx(expected3, rel=1e-2)

    def test_mean_return(self):
        assert self.asset_list.mean_return['RUB.FX'] == approx(-0.0854, rel=1e-2)
        assert self.asset_list.mean_return['MCFTR.INDX'] == approx(0.3701, rel=1e-2)
        assert self.asset_list.mean_return['RUB.INFL'] == approx(0.0319, rel=1e-2)

    def test_real_return(self):
        assert self.asset_list.real_mean_return['RUB.FX'] == approx(-0.15402, rel=1e-2)
        assert self.asset_list.real_mean_return['MCFTR.INDX'] == approx(0.2671, rel=1e-2)

    def test_annual_return_ts(self):
        assert self.asset_list.annual_return_ts.iloc[-1, 0] == approx(0.01829, rel=1e-2)
        assert self.asset_list.annual_return_ts.iloc[-1, 1] == approx(0.01180, rel=1e-2)

    def test_describe(self):
        description = self.asset_list.describe(tickers=False)
        description_sample = pd.read_pickle('data/asset_list_describe.pkl')
        assert_frame_equal(description, description_sample)

    def test_dividend_yield(self):
        assert list(self.spy.names.values()) == ['SPDR S&P 500 ETF Trust']
        assert self.spy.dividend_yield.iloc[-1, 0] == approx(0.012541968545679447, rel=1e-2)
        assert self.asset_list.dividend_yield.iloc[:, 0].sum() == 0

    def test_dividends_annual(self):
        assert self.spy.dividends_annual.iloc[-2, 0] == approx(1.4194999999999998, rel=1e-2)
        assert self.asset_list.dividends_annual.iloc[:, 0].sum() == 0

    def test_growing_dividend_years(self):
        assert self.spy.dividend_growing_years.iloc[-1, 0] == 0

    def test_paying_dividend_years(self):
        assert self.spy.dividend_paying_years.iloc[-2, 0] == 2

    def test_tracking_difference_failing(self):
        with pytest.raises(Exception, match='At least 2 symbols should be provided to calculate Tracking Difference.'):
            self.spy.tracking_difference

    def test_tracking_difference(self):
        assert self.asset_list.tracking_difference['MCFTR.INDX'].iloc[-1] == approx(0.4967, rel=1e-2)

    def test_tracking_difference_annualized(self):
        assert self.asset_list.tracking_difference_annualized.iloc[-1, 0] == approx(0.451000, rel=1e-2)

    def test_tracking_error(self):
        assert self.asset_list.tracking_error.iloc[-1, 0] == approx(0.19399, rel=1e-2)

    def test_index_corr(self):
        assert self.asset_list.index_corr.iloc[-1, 0] == approx(-0.57634, rel=1e-2)

    def test_index_beta(self):
        assert self.asset_list.index_beta.iloc[-1, 0] == approx(-0.563714, rel=1e-2)

    def test_skewness(self):
        assert self.asset_list.skewness['RUB.FX'].iloc[-1] == approx(0.444343, rel=1e-2)
        assert self.asset_list.skewness['MCFTR.INDX'].iloc[-1] == approx(0.24876, rel=1e-2)

    def test_rolling_skewness_failing(self):
        with pytest.raises(Exception, match=r'window size is less than data history depth'):
            self.asset_list.skewness_rolling(window=24)

    def test_kurtosis(self):
        assert self.asset_list.kurtosis['RUB.FX'].iloc[-1] == approx(0.89810, rel=1e-2)
        assert self.asset_list.kurtosis['MCFTR.INDX'].iloc[-1] == approx(-1.32129, rel=1e-2)

    def test_kurtosis_rolling(self):
        assert self.asset_list_lt.kurtosis_rolling(window=24)['RUB.FX'].iloc[-1] == approx(1.4425, rel=1e-2)
        assert self.asset_list_lt.kurtosis_rolling(window=24)['MCFTR.INDX'].iloc[-1] == approx(-0.11495, rel=1e-2)

    def test_jarque_bera(self):
        assert self.asset_list.jarque_bera['RUB.FX'].iloc[-1] == approx(0.84131, rel=1e-2)
        assert self.asset_list.jarque_bera['MCFTR.INDX'].iloc[-1] == approx(0.60333, rel=1e-2)


@mark.portfolio
def test_init_portfolio_failing():
    with pytest.raises(Exception, match=r'Number of tickers \(2\) should be equal to the weights number \(3\)'):
        Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], weights=[0.1, 0.2, 0.7]).symbols
    with pytest.raises(Exception, match='Weights sum is not equal to one.'):
        Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], weights=[0.1, 0.2]).symbols


@mark.portfolio
@mark.usefixtures('_init_portfolio')
class TestPortfolio:

    def test_ror(self):
        portfolio_sample = pd.read_pickle('data/portfolio.pkl')
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

    def test_dividend_yield(self):
        assert self.portfolio.dividend_yield.iloc[-1, :].sum() == 0

    def test_risk(self):
        assert self.portfolio.risk_monthly == approx(0.035718, rel=1e-2)
        assert self.portfolio.risk_annual == approx(0.139814, rel=1e-2)

    def test_rebalanced_portfolio_return(self):
        assert self.portfolio.get_rebalanced_portfolio_return_ts().mean() == approx(0.011220, rel=1e-2)
        assert self.portfolio.get_rebalanced_portfolio_return_ts(period='none').mean() == \
               approx(0.01221789515271935, rel=1e-2)

    def test_cagr(self):
        values = pd.Series({'portfolio': 0.1303543, 'RUB.INFL': 0.05548082428015655})
        assert_series_equal(self.portfolio.cagr, values, rtol=1e-4)

    @mark.test
    def test_describe(self):
        description = self.portfolio.describe()
        description_sample = pd.read_pickle('data/portfolio_description.pkl')
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
