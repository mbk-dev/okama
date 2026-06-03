import pandas as pd
import pytest

import okama as ok
from okama.common.helpers import helpers


@pytest.fixture()
def pf_mc_irr():
    """Fresh single-asset portfolio per test (offline assets via package conftest)."""
    return ok.Portfolio(
        ["MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf_mcirr.PF",
    )


def test_monte_carlo_irr_equals_cagr_distribution_without_cashflows(pf_mc_irr):
    # With no intermediate cash flows, each path's MWRR equals that path's CAGR.
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=50, seed=42)
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "none"
    pf_mc_irr.dcf.cashflow_parameters = ind

    irr_dist = pf_mc_irr.dcf.monte_carlo_irr()
    # Same cached draw -> deterministic oracle.
    expected = helpers.Frame.get_cagr(pf_mc_irr.dcf.mc.monte_carlo_returns_ts)
    pd.testing.assert_series_equal(irr_dist, expected, check_names=False, atol=1e-9, rtol=0)


def test_monte_carlo_irr_raises_when_cashflow_parameters_none(pf_mc_irr):
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=10, seed=1)
    pf_mc_irr.dcf.cashflow_parameters = None
    with pytest.raises(AttributeError, match=r"'cashflow_parameters' is not defined\."):
        pf_mc_irr.dcf.monte_carlo_irr()


def test_monte_carlo_irr_shape_and_name(pf_mc_irr):
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=25, seed=7)
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -300
    ind.indexation = 0.0
    pf_mc_irr.dcf.cashflow_parameters = ind

    s = pf_mc_irr.dcf.monte_carlo_irr()
    assert isinstance(s, pd.Series)
    assert len(s) == 25
    assert s.name == "monte_carlo_irr"
