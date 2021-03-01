import numpy as np
from pytest import mark
from numpy.testing import assert_allclose

from .conftest import data_folder

@mark.xfail
@mark.plots
def test_transition_map(init_plots):
    axes_data = init_plots.plot_transition_map(cagr=False, full_frontier=False).lines[0].get_data()
    values = np.genfromtxt(data_folder / 'test_transition_map.csv', delimiter=',')
    assert_allclose(axes_data, values, rtol=1e-1, atol=1e-1)


@mark.plots
def test_plot_assets(init_plots):
    axes_data = init_plots.plot_assets(tickers='names').collections[0].get_offsets().data
    values = np.genfromtxt(data_folder / 'test_plot_assets.csv', delimiter=',')
    assert_allclose(axes_data, values, rtol=1e-1, atol=1e-1)


@mark.plots
def test_plot_pair_ef(init_plots):
    axes_data = init_plots.plot_pair_ef(tickers='names').lines[0].get_data()
    values = np.genfromtxt(data_folder / 'test_plot_pair_ef.csv', delimiter=',')
    assert_allclose(axes_data, values, rtol=1e-1, atol=1e-1)
