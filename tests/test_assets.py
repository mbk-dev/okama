import pandas as pd
import pytest
from pytest import approx

from okama.assets import Asset, AssetList, Portfolio


class TestAsset:
    @pytest.fixture(autouse=True)
    def _init_asset(self):
        self.spy = Asset(symbol='SPY.US')

    def test_get_monthly_ror(self):
        assert isinstance(self.spy._get_monthly_ror('SPY.US', check_zeros=False), pd.Series)

    @pytest.mark.xfail
    def test_define_market_returns(self):
        assert self.spy._define_market('SBER.MCX') == 'MCX'
        assert self.spy._define_market('VFIAX') == 'US'

    def test_define_currency(self):
        assert self.spy._define_currency() == 'USD'


class TestPortfolio:
    @pytest.fixture(autouse=True)
    def _init_portfolio(self):
        self.portfolio = Portfolio(symbols=['VNQ.US', 'SNGSP.MCX'], curr='USD', first_date='2015-01', last_date='2020-01')

    def test_weights(self):
        assert self.portfolio.weights == [0.5, 0.5]

    def test_mean_return(self):
        assert self.portfolio.mean_return_monthly == approx(0.012189530337703269, rel=1e-4)
        assert self.portfolio.mean_return_annual == approx(0.1564905544168027, rel=1e-4)

    def test_risk(self):
        assert self.portfolio.risk_monthly == approx(0.04394340007565, rel=1e-4)
        assert self.portfolio.risk_annual == approx(0.1748308321768538, rel=1e-4)

    @pytest.mark.xfail
    def test_rebalanced_portfolio_return(self):
        assert self.portfolio.get_rebalanced_portfolio_return_ts().mean() == approx(0.01030113749004299, rel=1e-4)
        assert self.portfolio.get_rebalanced_portfolio_return_ts(period='N').mean() == \
               approx(0.010543594224881905, rel=1e-4)


class TestAssetList:
    @pytest.fixture(autouse=True)
    def _init_asset_list(self) -> None:
        self.asset_list = AssetList(symbols=['SPY.US', 'GAZP.MCX'], curr='RUB', first_date='2019-01', last_date='2020-01', check_zeros=False)

    def test_make_asset_list(self):
        assert self.asset_list.last_date == pd.to_datetime('2020-01')
        assert list(self.asset_list.ror) == ['SPY.US', 'GAZP.MCX']

    def test_calculate_wealth_indexes(self):
        assert self.asset_list.wealth_indexes.sum(axis=1)[-1] == \
               approx(2801.0512421559833, rel=1e-4)  # last month indexes sum

    def test_risk(self):
        assert self.asset_list.calculate_risk().sum() == approx(0.13611625845630504, rel=1e-4)
        assert self.asset_list.calculate_risk(annualize=True).sum() == approx(0.723380689819134, rel=1e-4)

    def test_semideviation(self):
        assert self.asset_list.semideviation.sum() == approx(0.05592602298844028, rel=1e-4)  # default is 7

    def test_drawdowns(self):
        assert self.asset_list.drawdowns.min().sum() == approx(-0.1798727778213174, rel=1e-4)

    def test_cagr(self):
        assert self.asset_list.cagr.sum() == approx(0.727483108320252, rel=1e-4)
