import matplotlib

# Ensure non-interactive backend for headless test runs
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import okama as ok
from okama.settings import DEFAULT_DISCOUNT_RATE


@pytest.fixture()
def pf_single_monthly(synthetic_env):
    """Single-asset Portfolio with monthly rebalancing and no inflation (mocked data)."""
    return ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))


def _configure_dcf_for_plot(pf):
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -500
    ind.indexation = DEFAULT_DISCOUNT_RATE
    pf.dcf.cashflow_parameters = ind
    pf.dcf.mc.period = 1
    pf.dcf.mc.number = 3
    return pf.dcf


def test_plot_forecast_monte_carlo_returns_axes_backtest(pf_single_monthly):
    dcf = _configure_dcf_for_plot(pf_single_monthly)
    ax = dcf.plot_forecast_monte_carlo(backtest=True, figsize=(4, 3))
    assert ax is not None
    assert ax.figure is not None
    assert len(ax.lines) >= 1
    plt.close("all")


def test_plot_forecast_monte_carlo_returns_axes_no_backtest(pf_single_monthly):
    dcf = _configure_dcf_for_plot(pf_single_monthly)
    ax = dcf.plot_forecast_monte_carlo(backtest=False, figsize=(4, 3))
    assert ax is not None
    assert ax.figure is not None
    assert len(ax.lines) == dcf.mc.number
    plt.close("all")
