import unittest

import pandas as pd
from okama.assets import Asset, AssetList, Portfolio


class TestAsset(unittest.TestCase):
    def setUp(self):
        self.spy = Asset('SPY.US')

    def test_get_monthly_ror(self):
        self.assertIsInstance(self.spy._get_monthly_ror('SPY.US'), pd.Series)

    def test_define_market_returns(self):
        self.assertEqual(self.spy._define_market('SBER.MCX'), 'MCX')
        self.assertEqual(self.spy._define_market('VFIAX'), 'US')

    def test_define_currency(self):
        self.assertEqual(self.spy._define_currency(), 'USD')


class TestAssetList(unittest.TestCase):
    def setUp(self) -> None:
        self.ls = ['SPY.US', 'GAZP.MCX']
        self.x = AssetList(symbols=self.ls, curr='RUB', first_date='2019-01', last_date='2020-01')

    def test_make_asset_list(self):
        self.assertEqual(self.x.last_date, pd.to_datetime('2020-01'))
        self.assertEqual(list(self.x.ror), self.ls)

    def test__calculate_wealth_indexes(self):
        self.assertEqual(self.x.wealth_indexes.sum(axis=1)[-1], 2801.0512421559833)  # last month indexes sum

    def test_risk(self):
        self.assertEqual(self.x.calculate_risk().sum(), 0.13611625845630504)
        self.assertEqual(self.x.calculate_risk(annualize=True).sum(), 0.723380689819134)

    def test_semideviation(self):
        self.assertEqual(self.x.semideviation.sum(), 0.05592602298844028)

    def test_drawdowns(self):
        self.assertEqual(self.x.drawdowns.min().sum(), -0.1798727778213174)

    def test_cagr(self):
        self.assertEqual(self.x.cagr.sum(), 0.727483108320252)


class TestPortfolio(unittest.TestCase):
    def setUp(self) -> None:
        ls = ['VNQ.US', 'SNGSP.MCX']
        self.x = Portfolio(symbols=ls, curr='USD', first_date='2015-01', last_date='2020-01')

    def test_portfolio(self):
        self.assertListEqual(self.x.weights, [0.5, 0.5])

        self.assertEqual(self.x.mean_return_monthly, 0.012189530337703269)
        self.assertEqual(self.x.mean_return_annual, 0.1564905544168027)

        self.assertEqual(self.x.risk_monthly, 0.04394340007565)
        self.assertEqual(self.x.risk_annual, 0.17483083217685386)

        self.assertEqual(self.x.get_rebalanced_portfolio_return_ts().mean(), 0.01030113749004299)
        self.assertEqual(self.x.get_rebalanced_portfolio_return_ts(period='N').mean(), 0.010543594224881905)

if __name__ == '__main__':
    unittest.main()

