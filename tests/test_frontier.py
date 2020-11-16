import pytest
from pytest import approx
from pytest import mark

import numpy as np
from numpy.testing import assert_allclose

from okama import EfficientFrontier


@mark.portfolio
def test_init_efficient_frontier():
    with pytest.raises(Exception, match=r'The number of symbols cannot be less than two'):
        EfficientFrontier(symbols=['MCFTR.INDX'])


@mark.frontier
def test_bounds_setter_failing(init_efficient_frontier):
    with pytest.raises(Exception, match=r'The number of symbols \(2\) and the length of bounds \(3\) should be equal.'):
        init_efficient_frontier.bounds = ((0, 1.), (0.5, 1.), (0, 0.5))


@mark.frontier
def test_gmv(init_efficient_frontier):
    assert_allclose(init_efficient_frontier.gmv_weights, np.array([0.67501259, 0.32498741]), rtol=1e-1, atol=1e-1)


@mark.frontier
def test_gmv_monthly(init_efficient_frontier):
    assert init_efficient_frontier.gmv_monthly[0] == approx(0.027662483460172523, rel=1e-2)


@mark.frontier
def test_gmv_annualized(init_efficient_frontier):
    assert init_efficient_frontier.gmv_annualized[0] == approx(0.10819898056182026, rel=1e-2)


@mark.frontier
def test_optimize_return(init_efficient_frontier):
    assert init_efficient_frontier.optimize_return(option='max')['Mean_return_monthly'] == approx(0.015324, rel=1e-2)
    assert init_efficient_frontier.optimize_return(option='min')['Mean_return_monthly'] == approx(0.008522, rel=1e-2)


@mark.frontier
def test_minimize_risk(init_efficient_frontier):
    assert init_efficient_frontier.minimize_risk(target_return=0.015324, monthly_return=True)['SBMX.MOEX'] == approx(1, rel=1e-2)
    assert init_efficient_frontier.minimize_risk(target_return=0.139241, monthly_return=False)['SBMX.MOEX'] == approx(0.35287, rel=1e-2)


@mark.frontier
def test_minimize_risk_bounds(init_efficient_frontier_bounds):
    assert init_efficient_frontier_bounds.minimize_risk(target_return=0.015324, monthly_return=True)['SBMX.MOEX'] == approx(1, rel=1e-2)
    assert init_efficient_frontier_bounds.minimize_risk(target_return=0.1548, monthly_return=False)['SBMX.MOEX'] == approx(0.52095, rel=1e-2)


@mark.frontier
def test_mean_return_range(init_efficient_frontier):
    assert_allclose(init_efficient_frontier.mean_return_range, np.array([0.008522, 0.015325]), rtol=1e-2)


@mark.frontier
def test_mean_return_range_bounds(init_efficient_frontier_bounds):
    assert_allclose(init_efficient_frontier_bounds.mean_return_range, np.array([0.011924, 0.015325]), rtol=1e-2)


@mark.frontier
def test_ef_points(init_efficient_frontier):
    assert init_efficient_frontier.ef_points['Mean return'].iloc[-1] == approx(0.20007879286573038, rel=1e-2)


@mark.rebalance
@mark.frontier
@mark.usefixtures('_init_efficient_frontier_reb')
class TestEfficientFrontierReb:
    # TODO: Add tests
    def test_ef_points_reb(self):
        assert self.ef.ef_points['GLD.US'].iloc[-1] == approx(1.0, rel=1e-2)





