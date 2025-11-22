import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import okama as ok


@pytest.fixture()
def ef_reb_ab(synthetic_env):
    """EfficientFrontierReb with two mocked assets A.US and B.US in USD.

    Uses synthetic_env to patch asset loading and currency, so no API is called.
    """
    return ok.EfficientFrontierReb(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10, rebalancing_strategy=ok.Rebalance(period="year")
    )


@pytest.fixture()
def ef_reb_three(synthetic_env):
    """EfficientFrontierReb with three mocked assets IDX.US, A.US and B.US."""
    return ok.EfficientFrontierReb(
        ["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False, n_points=12, rebalancing_strategy=ok.Rebalance(period="year")
    )


def test_init_efficient_frontier_reb_failing():
    # Error is raised before data loading, no API call occurs
    with pytest.raises(ValueError, match=r"The number of symbols cannot be less than two"):
        ok.EfficientFrontierReb(assets=["A.US"], ccy="USD")


def test_repr_contains_key_fields(ef_reb_ab):
    r = repr(ef_reb_ab)
    # Basic sanity: important fields are present
    assert "symbols" in r
    assert "currency" in r and "USD" in r
    assert "bounds" in r
    assert "rebalancing_period" in r or "rebalancing_period" in r
    assert "n_points" in r


def test_bounds_setter_reset_points(ef_reb_ab):
    ef = ef_reb_ab
    # Pre-fill cache
    _ = ef.ef_points
    assert not ef._ef_points.empty
    # Change bounds and ensure cache is cleared
    ef.bounds = tuple((0.0, 1.0) for _ in ef.symbols)
    assert ef._ef_points.empty


def test_gmv_annual_weights_basic_properties(ef_reb_ab):
    w = ef_reb_ab.gmv_annual_weights
    # weights length equals number of assets
    assert len(w) == len(ef_reb_ab.symbols)
    # weights within bounds and sum to 1
    lo_hi = ef_reb_ab.bounds
    assert_allclose(np.sum(w), 1.0, atol=1e-8)
    assert np.all(w >= np.array([lo for lo, _ in lo_hi]))
    assert np.all(w <= np.array([hi for _, hi in lo_hi]))


def test_gmv_annual_values_types(ef_reb_ab):
    risk, cagr = ef_reb_ab.gmv_annual_values
    assert isinstance(risk, float) and isinstance(cagr, float)
    assert risk >= 0.0


def test_global_max_return_portfolio_basic(ef_reb_ab):
    res = ef_reb_ab.global_max_return_portfolio
    # Keys exist
    assert set(res.keys()) >= {"Weights", "Risk", "Risk_monthly", "CAGR"}
    w = np.asarray(res["Weights"])
    assert_allclose(np.sum(w), 1.0, atol=1e-8)
    # Max return portfolio CAGR should be at least GMV CAGR
    _, gmv_cagr = ef_reb_ab.gmv_annual_values
    assert res["CAGR"] >= gmv_cagr - 1e-8


def test_minimize_risk_reaches_target_cagr(ef_reb_ab):
    # Pick a target in the middle of the left range
    r = ef_reb_ab._target_cagr_range_left
    target = float((r[0] + r[-1]) / 2)
    result = ef_reb_ab.minimize_risk(target)
    # Achieved CAGR is near the target
    assert result["CAGR"] == pytest.approx(target, rel=1e-2, abs=1e-3)
    # Weights sum to 1
    assert_allclose(np.sum(np.asarray(result["Weights"])), 1.0, atol=1e-8)


def test_ef_points_shape_and_monotonicity(ef_reb_ab):
    pts = ef_reb_ab.ef_points
    # At least n_points rows (can be larger if right part exists)
    assert len(pts) >= ef_reb_ab.n_points
    # CAGR should be non-decreasing along the left-to-right grid
    assert np.all(np.diff(pts["CAGR"]) >= -1e-12)


def test_get_monte_carlo_returns_dataframe(ef_reb_ab):
    np.random.seed(0)
    rp = ef_reb_ab.get_monte_carlo(10)
    assert list(rp.columns)[:2] == ["Risk", "CAGR"]
    assert len(rp) == 10


def test_plot_pair_ef_returns_axes(ef_reb_three):
    ax = ef_reb_three.plot_pair_ef(tickers="tickers")
    assert hasattr(ax, "lines") and len(ax.lines) >= 1

# 2


@pytest.fixture()
def ef_reb_ab(synthetic_env):
    """EfficientFrontierReb with two mocked assets A.US and B.US in USD.

    Uses synthetic_env to patch asset loading and currency, so no API is called.
    """
    return ok.EfficientFrontierReb(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10, rebalancing_strategy=ok.Rebalance(period="year")
    )


def test_gmv_monthly_weights_basic(ef_reb_ab):
    w = ef_reb_ab.gmv_monthly_weights
    # length equals number of assets
    assert len(w) == len(ef_reb_ab.symbols)
    # within bounds and sum to 1
    lo_hi = ef_reb_ab.bounds
    assert np.isclose(np.sum(w), 1.0, atol=1e-8)
    assert np.all(w >= np.array([lo for lo, _ in lo_hi]))
    assert np.all(w <= np.array([hi for _, hi in lo_hi]))


def test_get_gmv_monthly_returns_types_and_sign(ef_reb_ab):
    risk_m, mean_m = ef_reb_ab._get_gmv_monthly()
    # both are floats and non-negative risk
    assert isinstance(risk_m, float)
    assert isinstance(mean_m, float)
    assert risk_m >= 0.0


def test_rebalancing_strategy_setter_resets_cache(ef_reb_ab):
    # warm up cache
    _ = ef_reb_ab.ef_points
    assert not ef_reb_ab._ef_points.empty
    # change strategy and ensure cache is cleared
    ef_reb_ab.rebalancing_strategy = ok.Rebalance(period="none")
    assert ef_reb_ab.rebalancing_strategy.period == "none"
    assert ef_reb_ab._ef_points.empty


def test_n_points_setter_validation_and_cache(ef_reb_ab):
    # validation
    with pytest.raises(ValueError, match=r"n_points should be an integer"):
        ef_reb_ab.n_points = 10.0  # type: ignore
    with pytest.raises(ValueError, match=r"n_points should be greater than zero"):
        ef_reb_ab.n_points = 0

    # cache reset and effect on ef_points grid size
    _ = ef_reb_ab.ef_points
    assert not ef_reb_ab._ef_points.empty
    ef_reb_ab.n_points = 7
    assert ef_reb_ab._ef_points.empty
    pts = ef_reb_ab.ef_points
    assert len(pts) >= 7


def test_ticker_names_affects_result_keys(ef_reb_ab):
    # by default ticker_names=True -> per-asset keys are symbols
    target = float((ef_reb_ab._target_cagr_range_left[0] + ef_reb_ab._target_cagr_range_left[-1]) / 2)
    res_default = ef_reb_ab.minimize_risk(target)
    assert any(s in res_default for s in ef_reb_ab.symbols)

    # when set to False -> use asset names instead of tickers
    ef_reb_ab.ticker_names = False
    res_named = ef_reb_ab.minimize_risk(target)
    # names are stored in ef_reb_ab.names values
    asset_names = list(ef_reb_ab.names.values())
    assert any(n in res_named for n in asset_names)
    # and the ticker keys should not all be present simultaneously
    assert not all(s in res_named for s in ef_reb_ab.symbols)


def test_get_cagr_with_equal_weights_in_bounds(ef_reb_ab):
    n = len(ef_reb_ab.symbols)
    w = np.repeat(1.0 / n, n)
    cagr = ef_reb_ab._get_cagr(w)
    # should lie between min and max asset CAGR for synthetic series
    asset_cagrs = ok.common.helpers.helpers.Frame.get_cagr(ef_reb_ab.assets_ror)
    assert asset_cagrs.min() - 1e-6 <= cagr <= asset_cagrs.max() + 1e-6


def test_target_cagr_range_right_optional_and_sorted(ef_reb_ab):
    rr = ef_reb_ab._target_cagr_range_right
    if rr is None:
        assert rr is None
    else:
        assert isinstance(rr, np.ndarray)
        assert len(rr) >= 1
        # should decrease from global max to the asset cagr on the right
        assert np.all(np.diff(rr) <= 1e-12)


def test_target_risk_range_monotonic_and_start_is_gmv(ef_reb_ab):
    tr = ef_reb_ab.target_risk_range
    assert isinstance(tr, np.ndarray)
    assert len(tr) == ef_reb_ab.n_points
    # non-decreasing
    assert np.all(np.diff(tr) >= -1e-12)
    # starts from GMV annual risk (allow small tolerance)
    gmv_risk, _ = ef_reb_ab.gmv_annual_values
    assert tr[0] == pytest.approx(gmv_risk, rel=1e-3, abs=1e-3)


def test_max_annual_risk_asset_structure(ef_reb_ab):
    info = ef_reb_ab._max_annual_risk_asset
    assert set(info.keys()) == {"max_annual_risk", "ticker_with_largest_risk", "list_position"}
    assert info["ticker_with_largest_risk"] in ef_reb_ab.symbols
    assert 0 <= info["list_position"] < len(ef_reb_ab.symbols)
    assert ef_reb_ab.symbols[info["list_position"]] == info["ticker_with_largest_risk"]

