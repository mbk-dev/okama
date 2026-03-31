import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import okama as ok


@pytest.fixture()
def ef_ab(synthetic_env):
    """EfficientFrontier with two mocked assets A.US and B.US in USD.

    Uses synthetic_env to patch asset loading and currency, so no API is called.
    """
    return ok.EfficientFrontierSingle(["A.US", "B.US"], ccy="USD", inflation=False, n_points=10)


@pytest.fixture()
def ef_three(synthetic_env):
    """EfficientFrontier with three mocked assets IDX.US, A.US and B.US."""
    return ok.EfficientFrontierSingle(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False, n_points=20)


def test_init_efficient_frontier_failing():
    # Does not hit API because the error is raised before base class initialization
    with pytest.raises(ValueError, match=r"The number of symbols cannot be less than two"):
        ok.EfficientFrontierSingle(assets=["A.US"])  # any one symbol


def test_bounds_setter_failing(ef_ab):
    with pytest.raises(
        ValueError,
        match=r"The number of symbols \(2\) and the length of bounds \(3\) should be equal.",
    ):
        ef_ab.bounds = ((0, 1.0), (0.5, 1.0), (0, 0.5))


def test_repr_contains_key_fields(ef_ab):
    r = repr(ef_ab)
    # Basic sanity: important fields are present
    assert "symbols" in r
    assert "currency" in r and "USD" in r
    assert "bounds" in r
    assert "n_points" in r


def test_gmv_weights_basic_properties(ef_ab):
    w = ef_ab.gmv_monthly_weights
    # weights length equals number of assets
    assert len(w) == 2
    # weights are within bounds and sum to 1
    lo_hi = ef_ab.bounds
    assert_allclose(np.sum(w), 1.0, atol=1e-8)
    assert np.all(w >= np.array([lo for lo, _ in lo_hi]))
    assert np.all(w <= np.array([hi for _, hi in lo_hi]))


def test_gmv_annual_weights_basic_properties(ef_ab):
    w = ef_ab.gmv_annual_weights
    # weights length equals number of assets
    assert len(w) == len(ef_ab.symbols)
    # weights within bounds and sum to 1
    lo_hi = ef_ab.bounds
    assert_allclose(np.sum(w), 1.0, atol=1e-8)
    assert np.all(w >= np.array([lo for lo, _ in lo_hi]))
    assert np.all(w <= np.array([hi for _, hi in lo_hi]))


def test_gmv_monthly_and_annualized_consistency(ef_ab):
    risk_m, ret_m = ef_ab.gmv_monthly
    risk_a, ret_a = ef_ab.gmv_annualized
    assert isinstance(risk_m, float) and isinstance(ret_m, float)
    assert isinstance(risk_a, float) and isinstance(ret_a, float)
    # Annualization relationships hold by definition of helpers
    assert ret_a == ok.common.helpers.helpers.Float.annualize_return(ret_m)


def test_optimize_return_monotonicity(ef_ab):
    m = ef_ab.optimize_return(option="max")
    n = ef_ab.optimize_return(option="min")
    # Keys exist
    assert {"Mean_return_monthly", "Risk_monthly"}.issubset(set(m.keys()))
    # Max has mean return not smaller than min
    assert m["Mean_return_monthly"] >= n["Mean_return_monthly"]


def test_minimize_risk_reaches_target(ef_ab):
    rrange = ef_ab.mean_return_range
    lo, hi = rrange[0], rrange[-1]
    target = (lo + hi) / 2
    weights = ef_ab.minimize_risk(target_return=target, monthly_return=True)
    # Achieved monthly mean near the target (method uses tolerance for constraints)
    achieved = ok.common.helpers.helpers.Frame.get_portfolio_mean_return(
        np.array([weights[s] for s in ef_ab.symbols]), ef_ab.assets_ror
    )
    assert achieved == pytest.approx(target, rel=1e-2, abs=1e-3)


def test_mean_return_range_is_valid(ef_ab):
    rrange = ef_ab.mean_return_range
    lo, hi = rrange[0], rrange[-1]
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo <= hi


def test_ef_points_shape_and_monotonicity(ef_ab):
    pts = ef_ab.ef_points
    # Expected number of points
    assert len(pts) == ef_ab.n_points
    # Mean return should be non-decreasing along the frontier grid
    assert np.all(np.diff(pts["Mean return"]) >= -1e-12)


@pytest.mark.parametrize("rate_of_return", ["mean_return", "cagr"])  # reuse both modes
def test_get_tangency_portfolio_basic(rate_of_return, ef_ab):
    res = ef_ab.get_tangency_portfolio(rf_return=0.01, rate_of_return=rate_of_return)
    assert set(res.keys()) >= {"Weights", "Risk"}
    w = np.asarray(res["Weights"])
    assert_allclose(np.sum(w), 1.0, atol=1e-8)
    # within bounds
    lo_hi = ef_ab.bounds
    assert np.all(w >= np.array([lo for lo, _ in lo_hi]))
    assert np.all(w <= np.array([hi for _, hi in lo_hi]))


def test_get_most_diversified_portfolio_has_expected_fields(ef_ab):
    dic = ef_ab.get_most_diversified_portfolio()
    assert set(dic.keys()) >= {"Risk", "Mean return", "CAGR", "Diversification ratio"}


def test_get_monte_carlo_returns_dataframe(ef_ab):
    np.random.seed(0)
    rp = ef_ab.get_monte_carlo(10, kind="mean")
    assert list(rp.columns)[:2] == ["Risk", "Return"]
    assert len(rp) == 10


def test_plot_functions_return_lines(ef_three):
    # Capital Market Line plot
    ax = ef_three.plot_cml(rf_return=0.0, y_axe="mean_return")
    assert hasattr(ax, "lines") and len(ax.lines) >= 1

    # Transition map
    ax2 = ef_three.plot_transition_map(x_axe="risk")
    assert hasattr(ax2, "lines") and len(ax2.lines) >= 1

    # Pair EF
    ax3 = ef_three.plot_pair_ef(tickers="tickers")
    assert hasattr(ax3, "lines") and len(ax3.lines) >= 1


def test_mdp_points_basic_properties(ef_three):
    """MDP points should return a DataFrame with n_points rows and valid weights."""
    mdp = ef_three.mdp_points
    # Expected number of points
    assert len(mdp) == ef_three.n_points
    # Columns include metrics
    assert {"Risk", "Mean return", "CAGR"}.issubset(set(mdp.columns))
    # Weights columns are the asset symbols
    weight_cols = [c for c in mdp.columns if c in ef_three.symbols]
    assert set(weight_cols) == set(ef_three.symbols)
    # Each row weights sum to 1 (within numerical tolerance)
    s = mdp[weight_cols].sum(axis=1)
    assert np.allclose(s.values, 1.0, atol=1e-8)


def test_get_assets_tickers_modes(synthetic_env):
    """get_assets_tickers should return symbols when ticker_names=True and names when False."""
    ef_symbols = ok.EfficientFrontierSingle(
        ["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False, n_points=10, ticker_names=True
    )
    ef_names = ok.EfficientFrontierSingle(
        ["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False, n_points=10, ticker_names=False
    )

    # Tickers mode
    assert ef_symbols.get_assets_tickers() == ["IDX.US", "A.US", "B.US"]

    # Names mode (from synthetic_env fake assets)
    assert ef_names.get_assets_tickers() == ["Index", "Asset A", "Asset B"]


def test_plot_pair_ef_raises_with_less_than_three_assets(ef_ab):
    """plot_pair_ef should raise if number of assets < 3."""
    with pytest.raises(ValueError, match=r"The number of symbols cannot be less than 3"):
        ef_ab.plot_pair_ef()


def test_mean_return_range_when_full_frontier_false(ef_ab):
    """When full_frontier=False, the min is GMV monthly return, max is from optimize_return(max)."""
    # Recreate EF with full_frontier=False
    ef = ok.EfficientFrontierSingle(["A.US", "B.US"], ccy="USD", inflation=False, n_points=12, full_frontier=False)
    rrange = ef.mean_return_range
    # First equals GMV monthly return
    gmv_ret = ef.gmv_monthly[1]
    assert rrange[0] == pytest.approx(gmv_ret, rel=1e-8, abs=1e-12)
    # Last equals max mean return from optimize_return
    max_ret = ef.optimize_return(option="max")["Mean_return_monthly"]
    assert rrange[-1] == pytest.approx(max_ret, rel=1e-8, abs=1e-12)
    # Monotonic non-decreasing
    assert np.all(np.diff(rrange) >= -1e-12)
