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

    def test_cagr(self):
        self.assertEqual(self.x.cagr.sum(), 0.727483108320252)


class TestPortfolio(unittest.TestCase):
    def setUp(self) -> None:
        ls = ['VNQ.US', 'SNGSP.MCX']
        self.x = Portfolio(symbols=ls, curr='USD', first_date='2015-01', last_date='2020-01')

    def test_portfolio(self):
        self.assertListEqual(self.x.weights, [0.5, 0.5])

        self.assertEqual(self.x.mean_return_monthly, 0.012178819876164489)
        self.assertEqual(self.x.mean_return_annual, 0.1563437144065083)

        self.assertEqual(self.x.risk_monthly, 0.04356365834071058)
        self.assertEqual(self.x.risk_annual, 0.17328436789926255)

        self.assertEqual(self.x.get_rebalanced_portfolio_return_ts().mean(), 0.010292714584833771)
        self.assertEqual(self.x.get_rebalanced_portfolio_return_ts(period='N').mean(), 0.010525105802228808)

if __name__ == '__main__':
    unittest.main()

