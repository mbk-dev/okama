import pytest
from pandas._testing import assert_series_equal
from pytest import approx
from pytest import mark

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

import okama as ok
from tests import conftest


@mark.frontier
def test_init_efficient_frontier_failing():
    with pytest.raises(ValueError, match=r"The number of symbols cannot be less than two"):
        ok.EfficientFrontier(assets=["MCFTR.INDX"])


@mark.frontier
def test_bounds_setter_failing(init_efficient_frontier):
    with pytest.raises(
        ValueError,
        match=r"The number of symbols \(2\) and the length of bounds \(3\) should be equal.",
    ):
        init_efficient_frontier.bounds = ((0, 1.0), (0.5, 1.0), (0, 0.5))


def test_repr(init_efficient_frontier):
    value = pd.Series(
        dict(
            symbols="[SPY.US, SBMX.MOEX]",
            currency="RUB",
            first_date="2018-11",
            last_date="2020-02",
            period_length="1 years, 4 months",
            bounds="((0.0, 1.0), (0.0, 1.0))",
            inflation="RUB.INFL",
            n_points="2",
        )
    )
    assert repr(init_efficient_frontier) == repr(value)


@mark.frontier
def test_gmv(init_efficient_frontier):
    assert_allclose(
        init_efficient_frontier.gmv_weights,
        np.array([0.689278, 0.310722]),
        rtol=1e-2,
        atol=1e-2,
    )


@mark.frontier
def test_gmv_monthly(init_efficient_frontier):
    assert init_efficient_frontier.gmv_monthly[0] == approx(0.02541, rel=1e-2)


@mark.frontier
def test_gmv_annualized(init_efficient_frontier):
    assert init_efficient_frontier.gmv_annualized[0] == approx(0.0992, rel=1e-2)


@mark.frontier
def test_optimize_return(init_efficient_frontier):
    assert init_efficient_frontier.optimize_return(option="max")["Mean_return_monthly"] == approx(0.015324, rel=1e-2)
    assert init_efficient_frontier.optimize_return(option="min")["Mean_return_monthly"] == approx(0.008803, rel=1e-2)


@mark.frontier
def test_minimize_risk(init_efficient_frontier):
    assert init_efficient_frontier.minimize_risk(target_return=0.015324, monthly_return=True)["SBMX.MOEX"] == approx(
        1, rel=1e-2
    )
    assert init_efficient_frontier.minimize_risk(target_return=0.139241, monthly_return=False)["SBMX.MOEX"] == approx(
        0.32498, rel=1e-2
    )


@mark.frontier
def test_minimize_risk_bounds(init_efficient_frontier_bounds):
    assert init_efficient_frontier_bounds.minimize_risk(target_return=0.015324, monthly_return=True)[
        "SBMX.MOEX"
    ] == approx(1, rel=1e-2)
    assert init_efficient_frontier_bounds.minimize_risk(target_return=0.1548, monthly_return=False)[
        "SBMX.MOEX"
    ] == approx(0.50030, rel=1e-2)


@mark.frontier
def test_mean_return_range(init_efficient_frontier):
    assert_allclose(
        init_efficient_frontier.mean_return_range,
        np.array([0.008803, 0.015325]),
        rtol=1e-2,
    )


@mark.frontier
def test_mean_return_range_bounds(init_efficient_frontier_bounds):
    assert_allclose(
        init_efficient_frontier_bounds.mean_return_range,
        np.array([0.012064, 0.015325]),
        rtol=1e-2,
    )


@mark.frontier
def test_ef_points(init_efficient_frontier):
    assert init_efficient_frontier.ef_points["Mean return"].iloc[-1] == approx(0.20007879286573038, rel=1e-2)


test_tangency_data = [
    (False, [0.409596, 0.590404], 0.1627),  # cagr = False
    (True, [0.424916, 0.575084], 0.15617),  # cagr = True
]


@pytest.mark.parametrize(
    "cagr, expected_weights, expected_return", test_tangency_data, ids=["MSR Arithmetic mean", "MSR geometric mean"]
)
@mark.frontier
def test_get_tangency_portfolio(init_efficient_frontier, cagr, expected_weights, expected_return):
    rf_rate = 0.05
    dic = init_efficient_frontier.get_tangency_portfolio(cagr=cagr, rf_return=rf_rate)
    assert_allclose(dic["Weights"], expected_weights, atol=1e-2)
    assert dic["Rate_of_return"] == approx(expected_return, rel=1e-2)


@mark.frontier
def test_get_most_diversified_portfolio_global(init_efficient_frontier):
    dic = init_efficient_frontier.get_most_diversified_portfolio()
    dic_expected = {
        "SPY.US": 0.5811911230126665,
        "SBMX.MOEX": 0.41880887698733327,
        "Mean return": 0.147345849732854,
        "CAGR": 0.14303531962607208,
        "Risk": 0.10193153231742957,
        "Diversification ratio": 1.3203141067677615,
    }
    df = pd.Series(dic)
    df_expected = pd.Series(dic_expected)
    assert_series_equal(df, df_expected, rtol=1e-03)

test_monte_carlo = [
    ("mean", 0.10831, 0.1571, 0.5278),  # kind = 'mean'
    ("cagr", 0.10831, 0.1520, 0.5278),  # kind = 'cagr'
]


@pytest.mark.parametrize(
    "kind, risk, ror, weight", test_monte_carlo, ids=["Monte Carlo - Arithmetic mean", "Monte Carlo - Geometric mean"]
)
@mark.frontier
def test_get_monte_carlo(init_efficient_frontier, kind, risk, ror, weight):
    np.random.seed(0)
    rp = init_efficient_frontier.get_monte_carlo(10, kind=kind)
    rr = "Return" if kind == "mean" else "CAGR"
    assert rp.loc[9, "Risk"] == approx(risk, abs=1e-3)
    assert rp.loc[9, rr] == approx(ror, abs=1e-3)
    assert rp.loc[9, "SBMX.MOEX"] == approx(weight, abs=1e-3)


@mark.frontier
def test_get_most_diversified_portfolio(init_efficient_frontier):
    dic = init_efficient_frontier.get_most_diversified_portfolio(target_return=0.13, monthly_return=False)
    dic_expected = {
        "SPY.US": 0.7772116006495302,
        "SBMX.MOEX": 0.22278839935046985,
        "Mean return": 0.13000000306630777,
        "CAGR": 0.1258096704274465,
        "Risk": 0.09988773745239889,
        "Diversification ratio": 1.2385358051048263,
    }
    df = pd.Series(dic)
    df_expected = pd.Series(dic_expected)
    assert_series_equal(df, df_expected, rtol=1e-03)


@mark.frontier
def test_mdp_points(init_efficient_frontier_three_assets):
    assert init_efficient_frontier_three_assets.mdp_points["Mean return"].iloc[10] == approx(0.12039, rel=1e-2)
    assert init_efficient_frontier_three_assets.mdp_points["Diversification ratio"].iloc[10] == approx(1.6050, rel=1e-1)


@mark.frontier
def test_plot_cml(init_efficient_frontier):
    rf_rate = 0.02
    axes_data = np.array(init_efficient_frontier.plot_cml(rf_return=rf_rate).lines[1].get_data())
    expected = np.array([[0, 0.11053], [0.02, 0.1578]])
    assert_allclose(axes_data, expected, atol=1e-2)


@mark.frontier
def test_plot_transition_map(init_efficient_frontier_three_assets):
    axes_data = np.array(init_efficient_frontier_three_assets.plot_transition_map(cagr=False).lines[0].get_data())
    values = np.genfromtxt(conftest.data_folder / "test_transition_map.csv", delimiter=",")
    assert axes_data.shape == values.shape
    assert axes_data[0, 0] == approx(values[0, 0], abs=1e-1)


@mark.frontier
def test_plot_pair_ef(init_efficient_frontier_three_assets):
    axes_data = init_efficient_frontier_three_assets.plot_pair_ef(tickers="names").lines[0].get_data()
    values = np.genfromtxt(conftest.data_folder / "test_plot_pair_ef.csv", delimiter=",")
    assert_allclose(axes_data, values, rtol=1e-1, atol=1e-1)
