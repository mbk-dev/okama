import numpy as np  # noqa: I001
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


def _reference_irr_column(flows_column):
    """Independent per-path reference: solve the NPV with scipy.brentq."""
    from scipy import optimize

    n = flows_column.shape[0]
    t = np.arange(n, dtype=float)
    lower = -1.0 + max(1e-9, 10.0 ** (-300.0 / max(n - 1, 1)))

    def npv(rate):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float((flows_column * (1.0 + rate) ** (-t)).sum())

    try:
        monthly = optimize.brentq(npv, lower, 1e6, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")
    return (1.0 + monthly) ** 12 - 1.0


def test_monte_carlo_irr_matches_brentq_reference(pf_mc_irr):
    # Build the per-path reference from the (shared-draw) public MC methods and an
    # independent brentq solve; it must agree with the vectorized monte_carlo_irr.
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=4, mc_number=30, seed=11)
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -400
    ind.indexation = 0.0
    pf_mc_irr.dcf.cashflow_parameters = ind

    irr_dist = pf_mc_irr.dcf.monte_carlo_irr()

    wealth = pf_mc_irr.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=False)
    cash_flow = pf_mc_irr.dcf.monte_carlo_cash_flow(discounting="fv", remove_if_wealth_index_negative=True)
    terminal = wealth.iloc[-1]
    initial_investment = ind.initial_investment
    n_months, n_paths = cash_flow.shape

    for j in range(n_paths):
        v = np.empty(n_months + 1, dtype=float)
        v[0] = -initial_investment
        v[1:] = -cash_flow.iloc[:, j].to_numpy()
        v[-1] += terminal.iloc[j]
        ref = _reference_irr_column(v)
        got = irr_dist.iloc[j]
        if np.isnan(ref) or np.isnan(got):
            if not (np.isnan(ref) and np.isnan(got)):
                # Detailed diagnostic
                t_diag = np.arange(len(v), dtype=float)
                lower = -1.0 + max(1e-9, 10.0 ** (-300.0 / max(len(v) - 1, 1)))

                def npv_test(rate, v_=v, t_=t_diag):
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        return float((v_ * (1.0 + rate) ** (-t_)).sum())

                msg = f"Path {j} mismatch: ref={ref}, got={got}\\n"
                msg += f"  cash flow: v[0]={v[0]}, v[-1]={v[-1]}, sum={v.sum()}\\n"
                msg += f"  NPV(lower={lower})={npv_test(lower)}, NPV(0)={npv_test(0.0)}, NPV(1e6)={npv_test(1e6)}"
                pytest.fail(msg)
        else:
            assert got == pytest.approx(ref, abs=1e-8)


def test_monte_carlo_irr_reproducible_with_seed(pf_mc_irr):
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -400
    ind.indexation = 0.0
    pf_mc_irr.dcf.cashflow_parameters = ind

    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=4, mc_number=30, seed=123)
    first = pf_mc_irr.dcf.monte_carlo_irr()
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=4, mc_number=30, seed=123)
    second = pf_mc_irr.dcf.monte_carlo_irr()
    pd.testing.assert_series_equal(first, second)
