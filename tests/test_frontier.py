import unittest
from okama.frontier import EfficientFrontierReb, EfficientFrontier


class TestEfficientFrontier(unittest.TestCase):
    def setUp(self) -> None:
        ls = ['SPY.US', 'SBMX.MCX']
        self.ef = EfficientFrontier(symbols=ls, curr='RUB', first_date='2018-11', last_date='2020-02', n=2)

    def test_ef_points(self):
        self.assertEqual(self.ef.ef_points['Return'].iloc[-1], 0.20007879286573038)


class TestEfficientFrontierReb(unittest.TestCase):
    def setUp(self) -> None:
        ls = ['SPY.US', 'GLD.US']
        self.ef = EfficientFrontierReb(symbols=ls, curr='RUB', first_date='2019-01', last_date='2020-02', n=2)

    def test_ef_points_reb(self):
        self.assertEqual(self.ef.ef_points['GLD.US'].iloc[-1], 0.9988175545446986)


if __name__ == '__main__':
    unittest.main()