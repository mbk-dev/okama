import pytest
from pytest import approx
from pytest import mark

import numpy as np
from numpy.testing import assert_allclose

import pandas as pd

import okama as ok


@mark.rebalance
@mark.frontier
def test_init_efficient_frontier_reb():
    with pytest.raises(ValueError, match=r'The number of symbols cannot be less than two'):
        ok.EfficientFrontierReb(assets=['MCFTR.INDX'])


def test_repr(init_efficient_frontier_reb):
    value = pd.Series(dict(
        symbols="[SPY.US, GLD.US]",
        currency="RUB",
        first_date="2019-01",
        last_date="2020-02",
        period_length="1 years, 2 months",
        rebalancing_period="year",
        inflation="RUB.INFL",
    ))
    assert repr(init_efficient_frontier_reb) == repr(value)


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
    assert init_efficient_frontier_reb.global_max_return_portfolio['CAGR'] == approx(0.14904342, rel=1e-2)


@mark.rebalance
@mark.frontier
def test_ef_points_reb(init_efficient_frontier_reb):
    assert init_efficient_frontier_reb.ef_points['CAGR'].iloc[1] == approx(0.14268245, rel=1e-2)
