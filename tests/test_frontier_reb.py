import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import okama as ok


@pytest.fixture()
def ef_reb_ab(synthetic_env):
    """
    EfficientFrontierReb with two mocked assets A.US and B.US in USD.

    Uses synthetic_env to patch asset loading and currency, so no API is called.
    """
    return ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10, rebalancing_strategy=ok.Rebalance(period="year")
    )


@pytest.fixture()
def ef_reb_three(synthetic_env):
    """
    EfficientFrontierReb with three mocked assets IDX.US, A.US and B.US.
    """
    return ok.EfficientFrontier(
        ["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False, n_points=12, rebalancing_strategy=ok.Rebalance(period="year")
    )


def test_init_efficient_frontier_reb_failing():
    # Error is raised before data loading, no API call occurs
    with pytest.raises(ValueError, match=r"The number of symbols cannot be less than two"):
        ok.EfficientFrontier(assets=["A.US"], ccy="USD")


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


def test_verbose_property_setter(ef_reb_ab):
    """Test verbose property getter and setter."""
    # Default value from fixture
    assert isinstance(ef_reb_ab.verbose, bool)
    # Set new value and verify cache is cleared
    _ = ef_reb_ab.ef_points
    assert not ef_reb_ab._ef_points.empty
    ef_reb_ab.verbose = True
    assert ef_reb_ab.verbose is True
    assert ef_reb_ab._ef_points.empty
    # Test validation
    with pytest.raises(ValueError, match=r"verbose should be True or False"):
        ef_reb_ab.verbose = "true"  # type: ignore


def test_full_frontier_parameter(synthetic_env):
    """Test that full_frontier parameter is properly set."""
    ef_full = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10, 
        rebalancing_strategy=ok.Rebalance(period="year"), full_frontier=True
    )
    assert ef_full.full_frontier is True
    
    ef_partial = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10, 
        rebalancing_strategy=ok.Rebalance(period="year"), full_frontier=False
    )
    assert ef_partial.full_frontier is False


def test_max_cagr_asset_structure(ef_reb_ab):
    """Test _max_cagr_asset property returns correct structure."""
    info = ef_reb_ab._max_cagr_asset
    assert set(info.keys()) == {"max_asset_cagr", "ticker_with_largest_cagr", "list_position"}
    assert info["ticker_with_largest_cagr"] in ef_reb_ab.symbols
    assert 0 <= info["list_position"] < len(ef_reb_ab.symbols)
    assert ef_reb_ab.symbols[info["list_position"]] == info["ticker_with_largest_cagr"]
    assert isinstance(info["max_asset_cagr"], (float, np.floating))


def test_min_ratio_asset_structure(ef_reb_ab):
    """Test _min_ratio_asset property returns correct structure."""
    info = ef_reb_ab._min_ratio_asset
    assert set(info.keys()) == {"min_asset_cagr", "ticker_with_smallest_ratio", "list_position"}
    assert info["ticker_with_smallest_ratio"] in ef_reb_ab.symbols
    assert 0 <= info["list_position"] < len(ef_reb_ab.symbols)
    assert ef_reb_ab.symbols[info["list_position"]] == info["ticker_with_smallest_ratio"]


def test_max_ratio_asset_right_to_max_cagr(ef_reb_ab):
    """Test _max_ratio_asset_right_to_max_cagr property returns correct structure or None."""
    info = ef_reb_ab._max_ratio_asset_right_to_max_cagr
    if info is not None:
        assert set(info.keys()) == {"max_asset_cagr", "ticker_with_largest_cagr", "list_position"}
        assert info["ticker_with_largest_cagr"] in ef_reb_ab.symbols
        assert 0 <= info["list_position"] < len(ef_reb_ab.symbols)


def test_target_cagr_range_left_properties(ef_reb_ab):
    """Test _target_cagr_range_left returns proper array."""
    r = ef_reb_ab._target_cagr_range_left
    assert isinstance(r, np.ndarray)
    assert len(r) == ef_reb_ab.n_points
    # Should be non-decreasing
    assert np.all(np.diff(r) >= -1e-12)


def test_bounds_validation(synthetic_env):
    """Test bounds validation raises error for incorrect number of bounds."""
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year")
    )
    # Try to set bounds with wrong length
    with pytest.raises(ValueError, match=r"The number of symbols .* and the length of bounds .* should be equal"):
        ef.bounds = ((0.0, 0.5),)  # Only one bound for two assets


def test_rebalancing_strategy_validation(synthetic_env):
    """Test that rebalancing_strategy setter validates input type."""
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year")
    )
    with pytest.raises(ValueError, match=r"rebalancing_strategy must be of type Rebalance"):
        ef.rebalancing_strategy = "year"  # type: ignore


def test_ticker_names_validation(ef_reb_ab):
    """Test ticker_names property validation."""
    assert isinstance(ef_reb_ab.ticker_names, bool)
    with pytest.raises(ValueError, match=r"tickers should be True or False"):
        ef_reb_ab.ticker_names = "yes"  # type: ignore


def test_get_monte_carlo_with_bounds(synthetic_env):
    """Test get_monte_carlo respects bounds constraints."""
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0.3, 0.7), (0.3, 0.7))
    )
    np.random.seed(42)
    mc = ef.get_monte_carlo(n=5)
    assert len(mc) == 5
    assert "Risk" in mc.columns
    assert "CAGR" in mc.columns


def test_ef_points_caching(ef_reb_ab):
    """Test that ef_points results are cached properly."""
    # First call computes
    pts1 = ef_reb_ab.ef_points
    # Second call returns cached result
    pts2 = ef_reb_ab.ef_points
    assert pts1 is pts2  # Same object reference
    pd.testing.assert_frame_equal(pts1, pts2)


def test_minimize_risk_with_extreme_target(ef_reb_ab):
    """Test minimize_risk with target at boundaries."""
    # Test with target near GMV CAGR
    _, gmv_cagr = ef_reb_ab.gmv_annual_values
    result = ef_reb_ab.minimize_risk(gmv_cagr)
    assert "CAGR" in result
    assert "Risk" in result
    assert "Weights" in result
    assert result["CAGR"] == pytest.approx(gmv_cagr, rel=1e-2, abs=1e-3)


def test_get_most_diversified_portfolio_has_expected_fields(ef_reb_ab):
    """Test get_most_diversified_portfolio returns dict with required fields."""
    dic = ef_reb_ab.get_most_diversified_portfolio()
    # Check that all required fields are present
    assert set(dic.keys()) >= {"Risk", "CAGR", "Diversification ratio"}
    # Check that asset weights are present
    assert any(symbol in dic for symbol in ef_reb_ab.symbols)
    # Check types
    assert isinstance(dic["Risk"], (float, np.floating))
    assert isinstance(dic["CAGR"], (float, np.floating))
    assert isinstance(dic["Diversification ratio"], (float, np.floating))


def test_get_most_diversified_portfolio_with_target_return(ef_reb_ab):
    """Test get_most_diversified_portfolio with specified target_return."""
    # Get a target CAGR in the middle of the range
    r = ef_reb_ab._target_cagr_range_left
    target = float((r[0] + r[-1]) / 2)
    dic = ef_reb_ab.get_most_diversified_portfolio(target_return=target)
    # Check that CAGR is close to target
    assert dic["CAGR"] == pytest.approx(target, rel=1e-2, abs=1e-3)
    # Check that weights sum to 1
    weights = [dic[s] for s in ef_reb_ab.symbols]
    assert_allclose(np.sum(weights), 1.0, atol=1e-8)


def test_get_most_diversified_portfolio_weights_sum_to_one(ef_reb_ab):
    """Test that weights in MDP sum to 1."""
    dic = ef_reb_ab.get_most_diversified_portfolio()
    weights = [dic[s] for s in ef_reb_ab.symbols]
    assert_allclose(np.sum(weights), 1.0, atol=1e-8)


def test_get_most_diversified_portfolio_with_bounds(synthetic_env):
    """Test get_most_diversified_portfolio respects bounds."""
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0.3, 0.7), (0.3, 0.7))
    )
    dic = ef.get_most_diversified_portfolio()
    # Check bounds are respected
    for i, symbol in enumerate(ef.symbols):
        lo, hi = ef.bounds[i]
        assert lo <= dic[symbol] <= hi


def test_mdp_points_basic_properties(ef_reb_three):
    """Test mdp_points returns DataFrame with correct structure."""
    mdp = ef_reb_three.mdp_points
    # Expected number of points
    assert len(mdp) == ef_reb_three.n_points
    # Columns include required metrics
    assert {"Risk", "CAGR", "Diversification ratio"}.issubset(set(mdp.columns))
    # Weights columns are the asset symbols
    weight_cols = [c for c in mdp.columns if c in ef_reb_three.symbols]
    assert set(weight_cols) == set(ef_reb_three.symbols)
    # Each row weights sum to 1 (within numerical tolerance)
    s = mdp[weight_cols].sum(axis=1)
    assert np.allclose(s.values, 1.0, atol=1e-8)


def test_mdp_points_caching(ef_reb_ab):
    """Test that mdp_points results are cached properly."""
    # First call computes
    pts1 = ef_reb_ab.mdp_points
    # Second call returns cached result
    pts2 = ef_reb_ab.mdp_points
    assert pts1 is pts2  # Same object reference
    pd.testing.assert_frame_equal(pts1, pts2)


def test_mdp_points_cleared_on_bounds_change(ef_reb_ab):
    """Test that mdp_points cache is cleared when bounds change."""
    # Pre-fill cache
    _ = ef_reb_ab.mdp_points
    assert not ef_reb_ab._mdp_points.empty
    # Change bounds and ensure cache is cleared
    ef_reb_ab.bounds = ((0.2, 0.8), (0.2, 0.8))
    assert ef_reb_ab._mdp_points.empty


def test_mdp_points_cleared_on_n_points_change(ef_reb_ab):
    """Test that mdp_points cache is cleared when n_points changes."""
    # Pre-fill cache
    _ = ef_reb_ab.mdp_points
    assert not ef_reb_ab._mdp_points.empty
    # Change n_points and ensure cache is cleared
    ef_reb_ab.n_points = 15
    assert ef_reb_ab._mdp_points.empty


def test_mdp_points_cagr_monotonicity(ef_reb_ab):
    """Test that CAGR values in mdp_points are non-decreasing."""
    mdp = ef_reb_ab.mdp_points
    # CAGR should be non-decreasing
    assert np.all(np.diff(mdp["CAGR"]) >= -1e-12)


def test_get_tangency_portfolio_has_expected_fields(ef_reb_ab):
    """Test get_tangency_portfolio returns dict with required fields."""
    result = ef_reb_ab.get_tangency_portfolio(rf_return=0.02)
    # Check that all required fields are present
    assert set(result.keys()) == {"Weights", "Rate_of_return", "Risk"}
    # Check types
    assert isinstance(result["Weights"], np.ndarray)
    assert isinstance(result["Rate_of_return"], (float, np.floating))
    assert isinstance(result["Risk"], (float, np.floating))
    # Check weights length
    assert len(result["Weights"]) == len(ef_reb_ab.symbols)


def test_get_tangency_portfolio_weights_sum_to_one(ef_reb_ab):
    """Test that weights in tangency portfolio sum to 1."""
    result = ef_reb_ab.get_tangency_portfolio(rf_return=0.0)
    assert_allclose(np.sum(result["Weights"]), 1.0, atol=1e-8)


def test_get_tangency_portfolio_with_rate_of_return_cagr(ef_reb_ab):
    """Test get_tangency_portfolio with rate_of_return='cagr'."""
    result = ef_reb_ab.get_tangency_portfolio(rf_return=0.01, rate_of_return="cagr")
    assert "Weights" in result
    assert "Rate_of_return" in result
    assert "Risk" in result
    # Weights sum to 1
    assert_allclose(np.sum(result["Weights"]), 1.0, atol=1e-8)


def test_get_tangency_portfolio_with_rate_of_return_mean(ef_reb_ab):
    """Test get_tangency_portfolio with rate_of_return='mean_return'."""
    result = ef_reb_ab.get_tangency_portfolio(rf_return=0.01, rate_of_return="mean_return")
    assert "Weights" in result
    assert "Rate_of_return" in result
    assert "Risk" in result
    # Weights sum to 1
    assert_allclose(np.sum(result["Weights"]), 1.0, atol=1e-8)


def test_get_tangency_portfolio_respects_bounds(synthetic_env):
    """Test get_tangency_portfolio respects bounds."""
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0.2, 0.8), (0.2, 0.8))
    )
    result = ef.get_tangency_portfolio(rf_return=0.0)
    # Check bounds are respected
    for i, weight in enumerate(result["Weights"]):
        lo, hi = ef.bounds[i]
        assert lo - 1e-8 <= weight <= hi + 1e-8


def test_get_tangency_portfolio_invalid_rate_of_return(ef_reb_ab):
    """Test get_tangency_portfolio raises error for invalid rate_of_return."""
    with pytest.raises(ValueError, match="rate_of_return must be"):
        ef_reb_ab.get_tangency_portfolio(rf_return=0.0, rate_of_return="invalid")


def test_get_tangency_portfolio_positive_sharpe_ratio(ef_reb_ab):
    """Test that tangency portfolio has positive Sharpe ratio when rf_return is reasonable."""
    rf = 0.02
    result = ef_reb_ab.get_tangency_portfolio(rf_return=rf, rate_of_return="cagr")
    # Sharpe ratio should be positive (rate of return > rf_return for optimal portfolio)
    sharpe = (result["Rate_of_return"] - rf) / result["Risk"]
    assert sharpe > 0 or np.isclose(sharpe, 0, atol=1e-6)


def test_plot_cml_returns_axes(ef_reb_ab):
    """Test plot_cml returns matplotlib axes object."""
    ax = ef_reb_ab.plot_cml(rf_return=0.02)
    assert hasattr(ax, "lines")
    assert hasattr(ax, "collections")  # scatter plot creates collections


def test_plot_cml_with_zero_rf_return(ef_reb_ab):
    """Test plot_cml with zero risk-free return."""
    ax = ef_reb_ab.plot_cml(rf_return=0.0)
    assert ax is not None
    # Check that plot has multiple elements (EF line, MSR point, CML line)
    assert len(ax.lines) >= 2


def test_plot_cml_with_custom_figsize(ef_reb_ab):
    """Test plot_cml with custom figure size."""
    ax = ef_reb_ab.plot_cml(rf_return=0.01, figsize=(10, 8))
    fig = ax.get_figure()
    # Check that figure size is approximately as requested
    assert fig.get_figwidth() == pytest.approx(10, abs=0.1)
    assert fig.get_figheight() == pytest.approx(8, abs=0.1)


def test_plot_cml_has_expected_elements(ef_reb_ab):
    """Test plot_cml creates expected plot elements."""
    ax = ef_reb_ab.plot_cml(rf_return=0.02)
    # Should have lines (EF curve + CML line)
    assert len(ax.lines) >= 2
    # Should have scatter points (MSR point)
    assert len(ax.collections) >= 1
    # Should have annotations (MSR label)
    assert len(ax.texts) >= 1

