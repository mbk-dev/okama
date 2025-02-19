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
    with pytest.raises(ValueError, match=r"The number of symbols cannot be less than two"):
        ok.EfficientFrontierReb(assets=["MCFTR.INDX"])


def test_repr(init_efficient_frontier_reb):
    value = pd.Series(
        dict(
            symbols="[SPY.US, GLD.US]",
            currency="USD",
            first_date="2019-01",
            last_date="2020-02",
            period_length="1 years, 2 months",
            rebalancing_period="year",
            inflation="USD.INFL",
        )
    )
    assert repr(init_efficient_frontier_reb) == repr(value)


@mark.rebalance
@mark.frontier
def test_gmv_annual_weights(init_efficient_frontier_reb):
    assert_allclose(
        init_efficient_frontier_reb.gmv_annual_weights,
        np.array([0.384194, 0.615806]),
        rtol=1e-2,
        atol=1e-2,
    )


@mark.rebalance
@mark.frontier
def test_gmv_annual_values(init_efficient_frontier_reb):
    assert init_efficient_frontier_reb.gmv_annual_values[0] == approx(0.1189, rel=1e-1)


@mark.rebalance
@mark.frontier
def test_max_return(init_efficient_frontier_reb):
    assert init_efficient_frontier_reb.global_max_return_portfolio["CAGR"] == approx(0.1889, abs=1e-2)


@mark.rebalance
@mark.frontier
def test_ef_points_reb(init_efficient_frontier_reb):
    assert init_efficient_frontier_reb.ef_points["CAGR"].iloc[1] == approx(0.1889, abs=1e-2)


@mark.rebalance
@mark.frontier
def convex_right_frontier():

    ls_m = ["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"]
    curr_rub = "RUB"

    x = ok.EfficientFrontierReb(
        assets=ls_m,
        first_date="2005-01",
        last_date="2020-11",
        ccy=curr_rub,
        rebalancing_period="year", 
        n_points=5,
        verbose=True,
    )

    result = x._max_cagr_asset_right_to_max_cagr

    expected_result = {
        "max_asset_cagr": 0.17520700138002665,
        "ticker_with_largest_cagr": "PGJ.US",  
        "list_position": 2 
    }

    assert result == expected_result

@mark.rebalance
@mark.frontier
def nonconvex_right_frontier():

    ls_m = ["SPY.US", "GLD.US", "VB.US", "RGBITR.INDX", "MCFTR.INDX"]
    curr_rub = "RUB"

    x = ok.EfficientFrontierReb(
        assets=ls_m,
        first_date="2004-12",
        last_date="2020-12",
        ccy=curr_rub,
        rebalancing_period="year", 
        n_points=5,
        verbose=True,
    )

    result = x._max_cagr_asset_right_to_max_cagr

    expected_result = {
        "max_asset_cagr": 0.15691138904751512,
        "ticker_with_largest_cagr": "MCFTR.INDX",  
        "list_position": 4
    }

    assert result == expected_result
    
    
@mark.rebalance
@mark.frontier
def test_maximize_risk_with_convex_right_frontier():

    ls_m = ["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"]
    curr_rub = "RUB"

    x = EfficientFrontierReb(
        assets=ls_m,
        first_date="2005-01",
        last_date="2020-11",
        ccy=curr_rub,
        rebalancing_period="year", 
        n_points=5,
        verbose=True,
    )

    result = x._maximize_risk(0.17520700138002665)
    
    expected_result = (0, 0, 1, 0, 0)

    assert result == expected_result

@mark.rebalance
@mark.frontier
def test_maximize_risk_with_nonconvex_right_frontier():

    ls_m = ["SPY.US", "GLD.US", "VB.US", "RGBITR.INDX", "MCFTR.INDX"]
    curr_rub = "RUB"

    x = EfficientFrontierReb(
        assets=ls_m,
        first_date="2004-12",
        last_date="2020-12",
        ccy=curr_rub,
        rebalancing_period="year", 
        n_points=5,
        verbose=True,
    )

    result = x._maximize_risk(0.15691138904751512)
    
    expected_result = (0, 0, 0, 0, 1)

    assert result == expected_result
  






