import pytest
from pytest import approx
from pytest import mark

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

import okama as ok
from .conftest import data_folder


@mark.frontier
def test_init_efficient_frontier_failing():
    with pytest.raises(ValueError, match=r'The number of symbols cannot be less than two'):
        ok.EfficientFrontier(assets=['MCFTR.INDX'])


@mark.frontier
def test_bounds_setter_failing(init_efficient_frontier):
    with pytest.raises(ValueError, match=r'The number of symbols \(2\) and the length of bounds \(3\) should be equal.'):
        init_efficient_frontier.bounds = ((0, 1.), (0.5, 1.), (0, 0.5))


def test_repr(init_efficient_frontier):
    value = pd.Series(dict(
        symbols="[SPY.US, SBMX.MOEX]",
        currency="RUB",
        first_date="2018-11",
        last_date="2020-02",
        period_length="1 years, 4 months",
        bounds="((0.0, 1.0), (0.0, 1.0))",
        inflation="RUB.INFL",
        n_points="2",
    ))
    assert repr(init_efficient_frontier) == repr(value)


@mark.frontier
def test_gmv(init_efficient_frontier):
    assert_allclose(init_efficient_frontier.gmv_weights, np.array([0.67501259, 0.32498741]), rtol=1e-2, atol=1e-2)


@mark.frontier
def test_gmv_monthly(init_efficient_frontier):
    assert init_efficient_frontier.gmv_monthly[0] == approx(0.026076618401825784, rel=1e-2)


@mark.frontier
def test_gmv_annualized(init_efficient_frontier):
    assert init_efficient_frontier.gmv_annualized[0] == approx(0.10198459385117883, rel=1e-2)


@mark.frontier
def test_optimize_return(init_efficient_frontier):
    assert init_efficient_frontier.optimize_return(option='max')['Mean_return_monthly'] == approx(0.015324, rel=1e-2)
    assert init_efficient_frontier.optimize_return(option='min')['Mean_return_monthly'] == approx(0.008803, rel=1e-2)


@mark.frontier
def test_minimize_risk(init_efficient_frontier):
    assert init_efficient_frontier.minimize_risk(target_return=0.015324, monthly_return=True)['SBMX.MOEX'] == approx(1, rel=1e-2)
    assert init_efficient_frontier.minimize_risk(target_return=0.139241, monthly_return=False)['SBMX.MOEX'] == approx(0.32498, rel=1e-2)


@mark.frontier
def test_minimize_risk_bounds(init_efficient_frontier_bounds):
    assert init_efficient_frontier_bounds.minimize_risk(target_return=0.015324, monthly_return=True)['SBMX.MOEX'] == approx(1, rel=1e-2)
    assert init_efficient_frontier_bounds.minimize_risk(target_return=0.1548, monthly_return=False)['SBMX.MOEX'] == approx(0.50030, rel=1e-2)


@mark.frontier
def test_mean_return_range(init_efficient_frontier):
    assert_allclose(init_efficient_frontier.mean_return_range, np.array([0.008803, 0.015325]), rtol=1e-2)


@mark.frontier
def test_mean_return_range_bounds(init_efficient_frontier_bounds):
    assert_allclose(init_efficient_frontier_bounds.mean_return_range, np.array([0.012064, 0.015325]), rtol=1e-2)


@mark.frontier
def test_ef_points(init_efficient_frontier):
    assert init_efficient_frontier.ef_points['Mean return'].iloc[-1] == approx(0.20007879286573038, rel=1e-2)


@mark.frontier
def test_get_tangency_portfolio(init_efficient_frontier):
    rf_rate = 0.05
    dic = init_efficient_frontier.get_tangency_portfolio(rf_return=rf_rate)
    expected = [0.388589, 0.611411]
    assert_allclose(dic["Weights"], expected, atol=1e-2)
    assert dic['Mean_return'] == approx(0.1647, rel=1e-2)


@mark.frontier
def test_plot_cml(init_efficient_frontier):
    rf_rate = 0.02
    axes_data = np.array(init_efficient_frontier.plot_cml(rf_return=rf_rate).lines[1].get_data())
    expected = np.array([[0, 0.11053], [0.02, 0.1578]])
    assert_allclose(axes_data, expected, atol=1e-2)

@mark.frontier
def test_plot_transition_map(init_efficient_frontier_three_assets):
    axes_data = np.array(init_efficient_frontier_three_assets.plot_transition_map(cagr=False).lines[0].get_data())
    values = np.genfromtxt(data_folder / 'test_transition_map.csv', delimiter=',')
    assert axes_data.shape == values.shape
    assert axes_data[0, 0] == approx(values[0, 0], abs=1e-1)


@mark.frontier
def test_plot_pair_ef(init_efficient_frontier_three_assets):
    axes_data = init_efficient_frontier_three_assets.plot_pair_ef(tickers='names').lines[0].get_data()
    values = np.genfromtxt(data_folder / 'test_plot_pair_ef.csv', delimiter=',')
    assert_allclose(axes_data, values, rtol=1e-1, atol=1e-1)
