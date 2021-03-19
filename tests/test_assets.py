import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pytest import approx
from pytest import mark

import okama as ok

from .conftest import data_folder


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
        assert self.spy.dividends['2019'].iloc[-1] == 1.57

    def test_nav_ts(self):
        assert self.otkr.nav_ts[0] == 101820352.18


@mark.asset_list
def test_asset_list_init_failing():
    with pytest.raises(Exception, match=r'Symbols should be a list of string values.'):
        ok.AssetList(symbols=('RUB.FX', 'MCFTR.INDX'))


@mark.asset_list
@mark.usefixtures('_init_asset_list')
class TestAssetList:

    def test_ror(self):
        asset_list_sample = pd.read_pickle(data_folder / 'asset_list.pkl')
        asset_list_lt_sample = pd.read_pickle(data_folder / 'asset_list_lt.pkl')
        currencies_sample = pd.read_pickle(data_folder / 'currencies.pkl')
        real_estate_sample = pd.read_pickle(data_folder / 'real_estate.pkl')
        spy_sample = pd.read_pickle(data_folder / 'spy.pkl')
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
        assert self.asset_list.semideviation_monthly[0] == approx(0.015614, rel=1e-2)
        assert self.asset_list.semideviation_monthly[1] == approx(0, abs=1e-2)

    def test_semideviation_annual(self):
        assert self.asset_list.semideviation_annual[0] == approx(0.05408, rel=1e-2)
        assert self.asset_list.semideviation_annual[1] == approx(0, abs=1e-2)

    def test_get_var_historic(self):
        assert self.asset_list.get_var_historic(time_frame=1, level=5)['RUB.FX'] == approx(0.0411, rel=1e-2)
        assert self.asset_list.get_var_historic(time_frame=5, level=1)['MCFTR.INDX'] == approx(-0.1048, rel=1e-2)

    @mark.test
    def test_get_cvar_historic(self):
        assert self.asset_list.get_cvar_historic(level=5, time_frame=12)['RUB.FX'] == approx(0.1120, rel=1e-2)
        assert self.asset_list.get_cvar_historic(level=5, time_frame=12)['MCFTR.INDX'] == approx(-0.3130, rel=1e-2)

    def test_drawdowns(self):
        assert self.asset_list.drawdowns.min().sum() == approx(-0.082932, rel=1e-2)

    testdata = [
        (1, -0.0463, 0.3131, 0.0242),
        (None, -0.0888, 0.3651, 0.0318),
    ]

    # input_data - period (tuple[0]), expected1 - expected value 1 (tuple[1]), expected2 - expected value 2(tuple[2])
    @mark.parametrize("input_data,expected1,expected2,expected3", testdata, ids=["1 year", "full period"])
    def test_get_cagr(self, input_data, expected1, expected2, expected3):
        assert self.asset_list.get_cagr(period=input_data)['RUB.FX'] == approx(expected1, rel=1e-2)
        assert self.asset_list.get_cagr(period=input_data)['MCFTR.INDX'] == approx(expected2, rel=1e-2)
        assert self.asset_list.get_cagr(period=input_data)['RUB.INFL'] == approx(expected3, rel=1e-2)

    def test_get_rolling_cagr(self):
        assert self.asset_list_lt.get_rolling_cagr(window=24)['RUB.FX'].iloc[-1] == approx(0.05822, rel=1e-2)
        assert self.asset_list_lt.get_rolling_cagr(window=24)['MCFTR.INDX'].iloc[-1] == approx(0.2393, rel=1e-2)

    testdata = [
        ('YTD', 0.0182, 0.0118, 0.0040),
        (1, -0.0463, 0.3131, 0.0242),
        (None, -0.0957, 0.4009,  0.0345),
    ]

    # input_data - period (tuple[0]), expected1 - expected value 1 (tuple[1]), expected2 - expected value 2(tuple[2])
    @mark.parametrize("input_data,expected1,expected2,expected3", testdata, ids=["YTD", "1 year", "full period"])
    def test_get_cumulative_return(self, input_data, expected1, expected2, expected3):
        assert self.asset_list.get_cumulative_return(period=input_data)['RUB.FX'] == approx(expected1, rel=1e-2)
        assert self.asset_list.get_cumulative_return(period=input_data)['MCFTR.INDX'] == approx(expected2, rel=1e-2)
        assert self.asset_list.get_cumulative_return(period=input_data)['RUB.INFL'] == approx(expected3, rel=1e-2)

    def test_get_rolling_cumulative_return(self):
        assert self.asset_list_lt.get_rolling_cumulative_return(window=12)['RUB.FX'].iloc[-1] == approx(-0.0462, rel=1e-2)
        assert self.asset_list_lt.get_rolling_cumulative_return(window=12)['MCFTR.INDX'].iloc[-1] == approx(0.3130, rel=1e-2)

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
        description = self.asset_list.describe(tickers=False).iloc[:-2, :]  # last 2 rows are fresh lastdate
        description_sample = pd.read_pickle(data_folder / 'asset_list_describe.pkl').iloc[:-2, :]
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
