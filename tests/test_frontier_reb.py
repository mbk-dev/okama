import pytest
from pytest import approx
from pytest import mark

import numpy as np
from numpy.testing import assert_allclose

import okama as ok


@mark.rebalance
@mark.frontier
def test_init_efficient_frontier_reb():
    with pytest.raises(Exception, match=r'The number of symbols cannot be less than two'):
        ok.EfficientFrontierReb(symbols=['MCFTR.INDX'])


@mark.rebalance
@mark.frontier
def test_gmv_annual_weights(init_efficient_frontier_reb):
    assert_allclose(init_efficient_frontier_reb.gmv_annual_weights, np.array([0.765787, 0.234213]), rtol=1e-2, atol=1e-2)


@mark.rebalance
@mark.frontier
def test_gmv_annual_values(init_efficient_frontier_reb):
    assert init_efficient_frontier_reb.gmv_annual_values[0] == approx(0.09660054, rel=1e-2)


@mark.rebalance
@mark.frontier
def test_max_return(init_efficient_frontier_reb):
    assert init_efficient_frontier_reb.max_return['CAGR'] == approx(0.14904342, rel=1e-2)


@mark.rebalance
@mark.frontier
def test_ef_points_reb(init_efficient_frontier_reb):
    assert init_efficient_frontier_reb.ef_points['CAGR'].iloc[1] == approx(0.14268245, rel=1e-2)
