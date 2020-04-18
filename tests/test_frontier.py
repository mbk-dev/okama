from okama.frontier import EfficientFrontierReb, EfficientFrontier
from pytest import approx
import pytest


class TestEfficientFrontier:
    @pytest.fixture(autouse=True)
    def _init_efficient_frontier(self):
        ls = ['SPY.US', 'SBMX.MCX']
        self.ef = EfficientFrontier(symbols=ls, curr='RUB', first_date='2018-11', last_date='2020-02', n=2)

    def test_ef_points(self):
        assert self.ef.ef_points['Return'].iloc[-1] == approx(0.20007879286573038, rel=1e-4)


class TestEfficientFrontierReb:
    @pytest.fixture(autouse=True)
    def _init_efficient_frontier_reb(self):
        ls = ['SPY.US', 'GLD.US']
        self.ef = EfficientFrontierReb(symbols=ls, curr='RUB', first_date='2019-01', last_date='2020-02', n=2)

    @pytest.mark.xfail
    def test_ef_points_reb(self):
        assert self.ef.ef_points['GLD.US'].iloc[-1] == approx(0.9988175545446986, rel=1e-4)
