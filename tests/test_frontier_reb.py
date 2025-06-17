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
            symbols=["SPY.US", "GLD.US"],
            currency="USD",
            first_date="2019-01",
            last_date="2020-02",
            period_length="1 years, 2 months",
            rebalancing_period="year",
            rebalancing_abs_deviation=None,
            rebalancing_rel_deviation=None,
            bounds=((0, 1), (0, 1)),
            inflation="USD.INFL"
        )
    )
    assert repr(init_efficient_frontier_reb) == repr(value)


@mark.rebalance
@mark.frontier
def test_bounds_frontier(init_bounds_frontier):
    assert init_bounds_frontier.bounds == ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0, 1), (0, 1))


@mark.rebalance
@mark.frontier
def test_bounds_setter_valid_input(init_frontier_with_bounds):
    frontier = init_frontier_with_bounds
    expected_bounds = ((0, 0.4), (0, 1), (0, 1))
    assert frontier.bounds == expected_bounds


@mark.rebalance
@mark.frontier
def test_bounds_setter_empty_input(init_frontier_without_bounds):
    frontier = init_frontier_without_bounds
    frontier.bounds = None
    assert frontier.bounds == ((0.0, 1.0),) * len(frontier._assets)  


def test_bounds_setter_ef_points_reset(init_frontier_with_bounds):
    frontier = init_frontier_with_bounds

    frontier._ef_points = pd.DataFrame({
        'GLD.US': [0.25],
        'PGJ.US': [0.25],
        'VB.US': [0.25]
    })

    frontier.bounds = ((0, 1), (0, 1), (0, 1))

    assert frontier._ef_points.empty


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


test_params = {
    "with_bounds": {
        "target_cagr_1": 0.060973018282528796,
        "expected_risk_1": 0.14242,
        "target_cagr_2": 0.1035764996098511,
        "expected_risk_2": 0.1325340380516895,
    },
    "without_bounds": {
        "target_cagr_1": 0.060973018282528796,
        "expected_risk_1": 0.14242,
        "target_cagr_2": 0.1035764996098511,
        "expected_risk_2": 0.13428,
    }
}


@mark.rebalance
@mark.frontier
def test_minimize_risk_with_bounds(init_frontier_with_bounds):
    params = test_params["with_bounds"]
    
    result = init_frontier_with_bounds.minimize_risk(params["target_cagr_1"])
    assert np.isclose(result["Risk"], params["expected_risk_1"], atol=1e-2)
    
    result = init_frontier_with_bounds.minimize_risk(params["target_cagr_2"])
    assert np.isclose(result["Risk"], params["expected_risk_2"], atol=1e-2)


@mark.rebalance
@mark.frontier
def test_minimize_risk_without_bounds(init_frontier_without_bounds):
    params = test_params["without_bounds"]

    result = init_frontier_without_bounds.minimize_risk(params["target_cagr_1"])
    assert np.isclose(result["Risk"], params["expected_risk_1"], rtol=1e-2)
    
    result = init_frontier_without_bounds.minimize_risk(params["target_cagr_2"])
    assert np.isclose(result["Risk"], params["expected_risk_2"], rtol=1e-2)


@mark.rebalance
@mark.frontier
def test_minimize_risk_raises_error_when_no_solution(init_frontier_with_bounds):
    target_cagr = 0.5 
    
    with pytest.raises(RecursionError) as exc_info:
        init_frontier_with_bounds.minimize_risk(target_cagr)
    
    assert str(exc_info.value) == f"No solution found for target CAGR value: {target_cagr}."


@mark.rebalance
@mark.frontier
def test_min_ratio_asset_when_none(init_frontier_with_none):
    x = init_frontier_with_none
    result = x._min_ratio_asset

    assert result is None


@mark.rebalance
@mark.frontier
def test_min_ratio_asset_when_not_none(init_frontier_with_not_none):
    x = init_frontier_with_not_none
    result = x._min_ratio_asset
    # TODO: add case when left_assets.any() is False

    expected_result = {
        "min_asset_cagr": approx(0.1959425614987127, abs=1e-2),
        "ticker_with_smallest_ratio": "SPY.US",
        "list_position": 0
    }

    assert result == expected_result


@mark.rebalance
@mark.frontier
def test_convex_right_frontier(init_convex_frontier):
    x = init_convex_frontier
    result = x._max_ratio_asset_right_to_max_cagr

    expected_result = {
        "max_asset_cagr": approx(0.17520700138002665, abs=1e-2),
        "ticker_with_largest_cagr": "PGJ.US",
        "list_position": 2
    }

    assert result == expected_result


@mark.rebalance
@mark.frontier
def test_nonconvex_right_frontier(init_nonconvex_frontier):
    x = init_nonconvex_frontier
    result = x._max_ratio_asset_right_to_max_cagr

    expected_result = {
        "max_asset_cagr": approx(0.15691138904751512, abs=1e-2),
        "ticker_with_largest_cagr": "MCFTR.INDX",
        "list_position": 4
    }

    assert result == expected_result


@mark.rebalance
@mark.frontier
def test_maximize_risk_with_convex_right_frontier(init_convex_frontier):
    x = init_convex_frontier
    result = x._maximize_risk(0.17520700138002665)
    
    result_risk = result['Risk']
    expected_risk = approx(0.30419612104254684, abs=1e-2)

    assert result_risk == expected_risk


@mark.rebalance
@mark.frontier
def test_maximize_risk_with_nonconvex_right_frontier(init_nonconvex_frontier):
    x = init_nonconvex_frontier
    result = x._maximize_risk(0.15691138904751512)
    
    result_risk = result['Risk']
    expected_risk = approx(0.28761107914313766, abs=1e-2)

    assert result_risk == expected_risk

# TODO: add test for `get_monte_carlo`
# TODO: add test for `plot_pair_ef`