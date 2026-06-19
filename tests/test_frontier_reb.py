import numpy as np
import pandas as pd
import pytest
from joblib import parallel_config
from numpy.testing import assert_allclose

import okama as ok
from okama.common.helpers import helpers
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset


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
        ["IDX.US", "A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=12,
        rebalancing_strategy=ok.Rebalance(period="year"),
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


def test_minimize_risk_raises_runtimeerror_on_failure(ef_reb_ab):
    """Failed SLSQP optimisation must raise plain RuntimeError, not RecursionError
    (no recursion is involved; RecursionError misrepresents the failure mode)."""
    with pytest.raises(RuntimeError) as excinfo:
        ef_reb_ab.minimize_risk(target_value=10.0)  # 1000% CAGR is infeasible
    assert type(excinfo.value) is RuntimeError, f"Expected exact RuntimeError, got {type(excinfo.value).__name__}"


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
    # Weight columns present and sum to 1
    assert "A.US" in rp.columns
    assert "B.US" in rp.columns
    for _, row in rp.iterrows():
        assert_allclose(row["A.US"] + row["B.US"], 1.0, atol=1e-12)


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
    ts = ef_reb_ab._get_portfolio_ror_ts(ef_reb_ab.gmv_monthly_weights)
    expected_geometric_mean = float((ts.add(1.0).prod()) ** (1 / ts.shape[0]) - 1.0)
    # both are floats and non-negative risk
    assert isinstance(risk_m, float)
    assert isinstance(mean_m, float)
    assert risk_m >= 0.0
    assert mean_m == pytest.approx(expected_geometric_mean)


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
        ["A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        full_frontier=True,
    )
    assert ef_full.full_frontier is True

    ef_partial = ok.EfficientFrontier(
        ["A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        full_frontier=False,
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
    # n_points samples, plus asset CAGRs lying inside the range
    assert len(r) >= ef_reb_ab.n_points
    # Should be non-decreasing
    assert np.all(np.diff(r) >= -1e-12)


def test_bounds_validation(synthetic_env):
    """Test bounds validation raises error for incorrect number of bounds."""
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10, rebalancing_strategy=ok.Rebalance(period="year")
    )
    # Try to set bounds with wrong length
    with pytest.raises(ValueError, match=r"The number of symbols .* and the length of bounds .* should be equal"):
        ef.bounds = ((0.0, 0.5),)  # Only one bound for two assets


def test_rebalancing_strategy_validation(synthetic_env):
    """Test that rebalancing_strategy setter validates input type."""
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"], ccy="USD", inflation=False, n_points=10, rebalancing_strategy=ok.Rebalance(period="year")
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
        ["A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0.3, 0.7), (0.3, 0.7)),
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
        ["A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0.3, 0.7), (0.3, 0.7)),
    )
    dic = ef.get_most_diversified_portfolio()
    # Check bounds are respected
    for i, symbol in enumerate(ef.symbols):
        lo, hi = ef.bounds[i]
        assert lo <= dic[symbol] <= hi


def test_mdp_points_basic_properties(ef_reb_three):
    """Test mdp_points returns DataFrame with correct structure."""
    mdp = ef_reb_three.mdp_points
    # At least n_points rows (the target grid also samples asset CAGRs inside the range)
    assert len(mdp) >= ef_reb_three.n_points
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
        ["A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0.2, 0.8), (0.2, 0.8)),
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


# --- get_grid_portfolios tests (multi-period) ---


def test_get_grid_portfolios_returns_dataframe(ef_reb_ab):
    result = ef_reb_ab.get_grid_portfolios(step=0.50)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns)[:2] == ["Risk", "CAGR"]
    assert len(result) == 3  # 2 assets, step 0.50 → 3 combos
    # Weight columns present and sum to 1
    assert "A.US" in result.columns
    assert "B.US" in result.columns
    for _, row in result.iterrows():
        assert_allclose(row["A.US"] + row["B.US"], 1.0, atol=1e-12)


def test_get_grid_portfolios_row_count_three_assets(ef_reb_three):
    result = ef_reb_three.get_grid_portfolios(step=0.50)
    assert len(result) == 6  # 3 assets, step 0.50 → 6 combos


def test_get_grid_portfolios_with_bounds(synthetic_env):
    ef = ok.EfficientFrontier(
        ["A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=10,
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0.25, 0.75), (0.25, 0.75)),
    )
    result = ef.get_grid_portfolios(step=0.25)
    assert len(result) == 3  # bounds constrain to 3 combos
    assert "Risk" in result.columns
    assert "CAGR" in result.columns


def test_get_grid_portfolios_risk_and_cagr_are_floats(ef_reb_ab):
    result = ef_reb_ab.get_grid_portfolios(step=0.50)
    assert result["Risk"].dtype == np.float64 or np.issubdtype(result["Risk"].dtype, np.floating)
    assert result["CAGR"].dtype == np.float64 or np.issubdtype(result["CAGR"].dtype, np.floating)


def test_get_grid_portfolios_does_not_grow_ror_cache(ef_reb_three):
    """Grid enumeration yields only unique weight vectors, so it must not
    populate the optimization ror cache (which would otherwise grow O(points))."""
    ef_reb_three.get_grid_portfolios(step=0.50)
    assert len(ef_reb_three._ror_cache) == 0


def test_get_monte_carlo_does_not_grow_ror_cache(ef_reb_three):
    """Monte-Carlo enumeration yields unique weight vectors, so it must not
    populate the optimization ror cache."""
    ef_reb_three.get_monte_carlo(n=20)
    assert len(ef_reb_three._ror_cache) == 0


def test_get_grid_portfolios_respects_max_points(ef_reb_three):
    """get_grid_portfolios forwards max_points to the grid generator so an
    oversized request fails fast (3 assets, step 0.50 = 6 points > 2)."""
    with pytest.raises(ValueError, match="max_points"):
        ef_reb_three.get_grid_portfolios(step=0.50, max_points=2)


# --- minimum-variance corner: min-risk asset is not the min-CAGR asset ---


@pytest.fixture()
def ef_min_variance_corner(mocker):
    """EfficientFrontier where the lowest-risk asset is NOT the lowest-CAGR asset.

    DEP.US  - very low volatility, moderate CAGR  (true minimum-variance vertex)
    GLD.US  - the lowest CAGR                      (sets the left end of the CAGR range)
    STK.US  - high CAGR
    VOL.US  - extreme volatility, high CAGR

    DEP's CAGR sits in the interior of the target-CAGR range, and a high-risk
    DEP/VOL blend reproduces the same CAGR as pure DEP. A single equal-weights start
    lets SLSQP settle in that high-risk basin instead of the true minimum (pure DEP).
    """
    idx = pd.period_range("2010-01", periods=180, freq="M")
    n = len(idx)
    even = np.arange(n) % 2 == 0
    rng = np.random.default_rng(12)
    dep = pd.Series(0.009 + 0.001 * np.where(even, 1.0, -1.0), index=idx, name="DEP.US")
    gld = pd.Series(rng.normal(0.010, 0.07, n), index=idx, name="GLD.US")
    stk = pd.Series(rng.normal(0.016, 0.05, n), index=idx, name="STK.US")
    vol = pd.Series(np.where(even, 0.22, -0.15), index=idx, name="VOL.US")
    fake = {
        "DEP.US": FakeAsset("DEP.US", dep, currency="USD", name="Deposit"),
        "GLD.US": FakeAsset("GLD.US", gld, currency="USD", name="Gold-like"),
        "STK.US": FakeAsset("STK.US", stk, currency="USD", name="Stock"),
        "VOL.US": FakeAsset("VOL.US", vol, currency="USD", name="Volatile"),
    }

    def _get(symbols, first_date=None, last_date=None):
        out = {}
        for s in symbols:
            key = s.symbol if hasattr(s, "symbol") else s
            out[key] = fake[key]
        return out

    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", side_effect=_get)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=FakeCurrencyAsset)
    return ok.EfficientFrontier(
        ["STK.US", "GLD.US", "DEP.US", "VOL.US"],
        ccy="USD",
        inflation=False,
        n_points=20,
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


def test_minimize_risk_reaches_minimum_variance_asset(ef_min_variance_corner):
    """minimize_risk at the min-variance asset's CAGR must return that asset's own risk.

    The lowest-risk single asset is itself a feasible portfolio at its own CAGR, so it is
    the global risk minimum for that CAGR target. A single equal-weights start makes SLSQP
    settle in a high-risk basin and miss it; the result must not exceed the asset's risk.
    """
    ef = ef_min_variance_corner
    cagr = helpers.Frame.get_cagr(ef.assets_ror)
    risk = helpers.Float.annualize_risk(ef.assets_ror.std(), ef.assets_ror.mean())
    min_risk_asset = risk.idxmin()
    # precondition: the min-risk asset is not also the min-CAGR asset (otherwise no corner)
    assert cagr.idxmin() != min_risk_asset

    result = ef.minimize_risk(float(cagr[min_risk_asset]))

    assert result["Risk"] == pytest.approx(float(risk[min_risk_asset]), abs=1e-3)


def test_ef_points_pass_through_minimum_variance_asset(ef_min_variance_corner):
    """The lowest-risk single asset must lie on the frontier, not just outside it.

    The target-CAGR sampling must include the minimum-variance asset's CAGR so the drawn
    frontier passes through that asset instead of cutting the corner near it.
    """
    ef = ef_min_variance_corner
    cagr = helpers.Frame.get_cagr(ef.assets_ror)
    risk = helpers.Float.annualize_risk(ef.assets_ror.std(), ef.assets_ror.mean())
    mva = risk.idxmin()
    pts = ef.ef_points
    distance = np.sqrt((pts["Risk"] - float(risk[mva])) ** 2 + (pts["CAGR"] - float(cagr[mva])) ** 2).min()
    assert distance == pytest.approx(0.0, abs=1e-3)


# --- right-part corner: target equals the right asset's own CAGR (issue #84) ---


@pytest.fixture()
def ef_right_corner(mocker):
    """EfficientFrontier with a 'right asset' beyond the global max-CAGR portfolio (issue #84).

    HIVOL.US - high volatility, the lowest CAGR
    MODV.US  - moderate volatility; the highest single-asset CAGR, but a rebalanced mix
               (rebalancing bonus) reaches a higher CAGR at lower risk, so MODV lies to
               the right of the global max-CAGR portfolio and bounds the right part
    MIDC.US  - CAGR strictly inside the left target range, not the minimum-variance asset

    The right part of the frontier must end exactly at the MODV corner: the terminal
    target of `_target_cagr_range_right` is MODV's own CAGR, and the 100% MODV portfolio
    is the maximum-risk portfolio for that CAGR. SLSQP started exactly at that vertex of
    the bounds fails spuriously, and the fallback start converges to an interior local
    maximum with lower risk, drawing a dominated hook (mbk-dev/okama#84).
    """
    idx = pd.period_range("2005-01", periods=240, freq="M")
    rng = np.random.default_rng(4)

    def make_ror(loc, scale):
        # Standardize the draw so the realized mean and std match loc/scale exactly.
        z = rng.normal(0, 1, len(idx))
        z = (z - z.mean()) / z.std(ddof=1)
        return loc + scale * z

    hivol = pd.Series(make_ror(0.00692, 0.09), index=idx, name="HIVOL.US")
    modv = pd.Series(make_ror(0.00807, 0.05), index=idx, name="MODV.US")
    midc = pd.Series(make_ror(0.00632, 0.058), index=idx, name="MIDC.US")
    fake = {
        "HIVOL.US": FakeAsset("HIVOL.US", hivol, currency="USD", name="High vol"),
        "MODV.US": FakeAsset("MODV.US", modv, currency="USD", name="Moderate vol"),
        "MIDC.US": FakeAsset("MIDC.US", midc, currency="USD", name="Middle CAGR"),
    }

    def _get(symbols, first_date=None, last_date=None):
        out = {}
        for s in symbols:
            key = s.symbol if hasattr(s, "symbol") else s
            out[key] = fake[key]
        return out

    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", side_effect=_get)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=FakeCurrencyAsset)
    return ok.EfficientFrontier(
        ["HIVOL.US", "MODV.US", "MIDC.US"],
        ccy="USD",
        inflation=False,
        n_points=16,
        full_frontier=True,
        rebalancing_strategy=ok.Rebalance(period="month"),
    )


def test_maximize_risk_reaches_right_asset_corner(ef_right_corner):
    """_maximize_risk at the right asset's own CAGR must return that single-asset portfolio.

    The 100% right-asset portfolio is feasible at its own CAGR and is the maximum-risk
    solution there, so the right part of the frontier must end at this corner. Accepting
    a lower-risk interior solution draws a dominated hook next to the asset point.
    """
    ef = ef_right_corner
    right = ef._max_ratio_asset_right_to_max_cagr
    assert right is not None  # precondition: the right part of the frontier exists
    ticker = right["ticker_with_largest_cagr"]
    cagr = helpers.Frame.get_cagr(ef.assets_ror)
    risk = helpers.Float.annualize_risk(ef.assets_ror.std(), ef.assets_ror.mean())

    result = ef._maximize_risk(float(cagr[ticker]))

    assert result["Risk"] == pytest.approx(float(risk[ticker]), abs=1e-3)
    assert result[ticker] == pytest.approx(1.0, abs=1e-3)


def test_target_cagr_range_left_includes_interior_asset_cagrs(ef_right_corner):
    """Each asset whose CAGR lies inside the left target range must be sampled exactly.

    Otherwise the frontier polyline passes near single-asset points instead of through
    them; when min-risk jumps at an asset's CAGR (issue #84) the chord misses the asset.
    """
    ef = ef_right_corner
    cagr = helpers.Frame.get_cagr(ef.assets_ror)
    r = ef._target_cagr_range_left
    interior = cagr[(cagr > r[0]) & (cagr < r[-1])]
    assert len(interior) > 0  # precondition: at least one asset CAGR strictly inside the range
    for value in interior:
        assert np.isclose(r, value, rtol=0, atol=1e-12).any()


def test_target_cagr_range_right_keeps_terminal_for_small_n_points(ef_right_corner):
    """The right range must keep its terminal value (the right asset's CAGR) for any n_points.

    When the right CAGR span is much narrower than the left one, the point-count formula
    produced a single-point range that became empty after dropping the first point,
    silently removing the whole right part of the frontier together with its corner.
    """
    ef = ef_right_corner
    right = ef._max_ratio_asset_right_to_max_cagr
    assert right is not None  # precondition: the right part of the frontier exists
    ef.n_points = 10
    rr = ef._target_cagr_range_right
    assert rr is not None and len(rr) >= 1
    assert np.isclose(rr[-1], right["max_asset_cagr"], rtol=0, atol=1e-15)


def test_ef_points_right_rows_not_duplicated_under_threading_backend(ef_right_corner):
    """The right-part worker must only return its row, never append it itself (issue #86).

    With a thread-based joblib backend the worker shares the records list with the caller,
    so a worker-side append lands in the shared list and the `+=` collection of the
    Parallel results adds every right-part row a second time.
    """
    ef = ef_right_corner
    ef.n_points = 4  # keep the optimizer runs few; the right part is what matters
    assert ef._target_cagr_range_right is not None  # precondition: the right part exists
    with parallel_config(backend="threading"):
        pts = ef.ef_points
    assert not pts[["Risk", "CAGR"]].duplicated().any()


# --- pairwise EF gap: mix barely beats the best asset (issue #87) ---


@pytest.fixture()
def ef_pair_small_bonus(mocker):
    """Two-asset EfficientFrontier where the yearly-rebalanced mix barely beats the best asset.

    HIVOL.US - higher volatility, lower CAGR
    MODV.US  - the best single asset; the global max-CAGR portfolio is a HIVOL/MODV mix
               whose CAGR exceeds MODV's by less than 1% *relative* (a small rebalancing
               bonus), while its risk is several percent lower than MODV's.

    This is the issue #87 geometry (MCFTR.INDX/GC.COMM pair): the 1% CAGR tolerance in
    `_max_ratio_asset_right_to_max_cagr` treats the mix as "being" the asset, the right
    part of the frontier is skipped, and the frontier line stops at the mix instead of
    descending to the MODV corner.
    """
    idx = pd.period_range("2005-01", periods=240, freq="M")
    rng = np.random.default_rng(4)

    def make_ror(loc, scale):
        # Standardize the draw so the realized mean and std match loc/scale exactly.
        z = rng.normal(0, 1, len(idx))
        z = (z - z.mean()) / z.std(ddof=1)
        return loc + scale * z

    hivol = pd.Series(make_ror(0.005, 0.06), index=idx, name="HIVOL.US")
    modv = pd.Series(make_ror(0.00807, 0.05), index=idx, name="MODV.US")
    fake = {
        "HIVOL.US": FakeAsset("HIVOL.US", hivol, currency="USD", name="High vol"),
        "MODV.US": FakeAsset("MODV.US", modv, currency="USD", name="Moderate vol"),
    }

    def _get(symbols, first_date=None, last_date=None):
        out = {}
        for s in symbols:
            key = s.symbol if hasattr(s, "symbol") else s
            out[key] = fake[key]
        return out

    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", side_effect=_get)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=FakeCurrencyAsset)
    return ok.EfficientFrontier(
        ["HIVOL.US", "MODV.US"],
        ccy="USD",
        inflation=False,
        n_points=20,
        full_frontier=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


def test_right_asset_detected_when_mix_barely_beats_best_asset(ef_pair_small_bonus):
    """An asset with a big risk gap to the global max point must bound the right part.

    The asset 'is' the global max portfolio only when both its CAGR and its risk match
    the global max point. A sub-1% CAGR edge of the best mix with a multi-percent risk
    gap must not suppress the right part of the frontier (issue #87).
    """
    ef = ef_pair_small_bonus
    cagr = helpers.Frame.get_cagr(ef.assets_ror)
    risk = helpers.Float.annualize_risk(ef.assets_ror.std(), ef.assets_ror.mean())
    gm = ef.global_max_return_portfolio
    best = cagr.idxmax()
    # preconditions: the global max is a mix with a small CAGR edge and a big risk gap
    assert 0 < 1 - float(cagr[best]) / gm["CAGR"] < 0.01
    assert float(risk[best]) > gm["Risk"] * 1.01

    right = ef._max_ratio_asset_right_to_max_cagr

    assert right is not None
    assert right["ticker_with_largest_cagr"] == best


def test_pair_ef_points_reach_best_asset_corner_for_small_bonus(ef_pair_small_bonus):
    """The two-asset frontier must terminate at the best asset point, not at the mix.

    A 100% single-asset portfolio is always a member of the two-asset opportunity set,
    so the frontier line has no reason to stop short of the asset dot (issue #87).
    """
    ef = ef_pair_small_bonus
    cagr = helpers.Frame.get_cagr(ef.assets_ror)
    risk = helpers.Float.annualize_risk(ef.assets_ror.std(), ef.assets_ror.mean())
    best = cagr.idxmax()

    last = ef.ef_points.iloc[-1]

    assert last["Risk"] == pytest.approx(float(risk[best]), abs=1e-3)
    assert last["CAGR"] == pytest.approx(float(cagr[best]), abs=1e-4)


def test_plot_pair_ef_uses_parent_rebalancing_strategy(synthetic_env, mocker):
    """plot_pair_ef must compute pair frontiers with the parent's rebalancing strategy.

    Pair EfficientFrontier objects were created without rebalancing_strategy, silently
    falling back to the default yearly rebalancing whatever the parent uses.
    """
    ef = ok.EfficientFrontier(
        ["IDX.US", "A.US", "B.US"],
        ccy="USD",
        inflation=False,
        n_points=12,
        rebalancing_strategy=ok.Rebalance(period="month"),
    )
    spy = mocker.spy(ok.EfficientFrontier, "__init__")

    ef.plot_pair_ef(tickers="tickers")

    assert len(spy.call_args_list) == 3  # one per asset pair
    for call in spy.call_args_list:
        strategy = call.kwargs.get("rebalancing_strategy")
        assert strategy is not None and strategy.period == "month"
