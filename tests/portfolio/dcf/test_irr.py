import numpy as np
import pytest

import okama as ok  # noqa: F401  (used by integration tests added in later tasks)
from okama.portfolios import dcf_calculations
from okama.settings import _MONTHS_PER_YEAR, DEFAULT_DISCOUNT_RATE  # noqa: F401


def test_irr_core_single_in_single_out_matches_closed_form():
    # -1000 at t0, +1200 at t12 on a monthly grid -> annual IRR is exactly 20%.
    cf = np.zeros(13)
    cf[0] = -1000.0
    cf[12] = 1200.0
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=_MONTHS_PER_YEAR)
    assert result[0] == pytest.approx(0.2, abs=1e-9)


def test_irr_core_textbook_per_period_rate():
    # -100, +60, +60 with periods_per_year=1 -> per-period IRR from 100 x^2 - 60 x - 60 = 0.
    cf = np.array([-100.0, 60.0, 60.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    x = (60.0 + np.sqrt(3600.0 + 24000.0)) / 200.0
    assert result[0] == pytest.approx(x - 1.0, abs=1e-9)


def test_irr_core_no_sign_change_returns_nan():
    cf = np.array([-100.0, -50.0, -30.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert np.isnan(result[0])


def test_irr_core_vectorized_columns_are_independent():
    # Two columns solved at once; periods_per_year=1 so annual == per-period rate.
    cf = np.array(
        [
            [-1000.0, -1000.0],
            [0.0, 0.0],
            [1200.0, 900.0],
        ]
    )
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert result[0] == pytest.approx(np.sqrt(1.2) - 1.0, abs=1e-9)
    assert result[1] == pytest.approx(np.sqrt(0.9) - 1.0, abs=1e-9)


def test_irr_core_depleted_partial_recovery_is_negative():
    # Invested 1000, recovered only 300 over the period, terminal 0 -> finite, negative IRR.
    cf = np.array([-1000.0, 100.0, 100.0, 100.0, 0.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert np.isfinite(result[0])
    assert result[0] < 0.0


def test_irr_core_uses_brentq_fallback_when_newton_does_not_converge():
    # A deliberately bad seed + a single Newton step cannot converge, so the column
    # is routed to the brentq fallback, which must still find the textbook root.
    cf = np.array([-100.0, 60.0, 60.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1, guess=1e6, max_iter=1)
    x = (60.0 + np.sqrt(3600.0 + 24000.0)) / 200.0
    assert result[0] == pytest.approx(x - 1.0, abs=1e-9)


@pytest.fixture()
def pf_no_inflation(synthetic_env):
    """Two-asset portfolio, monthly rebalancing, no inflation (mocked data)."""
    return ok.Portfolio(
        ["A.US", "B.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month")
    )


def test_irr_equals_cagr_without_intermediate_cashflows(pf_no_inflation):
    # MWRR with a single inflow and single outflow is identically the TWR/CAGR.
    ind = ok.IndexationStrategy(pf_no_inflation)
    ind.initial_investment = 10_000
    ind.frequency = "none"  # no regular cash flow -> only initial_investment and terminal value
    ind.indexation = DEFAULT_DISCOUNT_RATE
    pf_no_inflation.dcf.cashflow_parameters = ind

    expected_cagr = pf_no_inflation.get_cagr().iloc[-1].loc[pf_no_inflation.symbol]
    assert pf_no_inflation.dcf.irr() == pytest.approx(expected_cagr, abs=1e-9)


def test_irr_raises_when_cashflow_parameters_none(pf_no_inflation):
    pf_no_inflation.dcf.cashflow_parameters = None
    with pytest.raises(AttributeError, match=r"'cashflow_parameters' is not defined\."):
        pf_no_inflation.dcf.irr()


def _reference_irr(dcf_obj):
    """Independent, slow reference: rebuild the vector and solve with scipy.brentq."""
    from scipy import optimize

    cash_flow = dcf_obj.cash_flow_ts("fv", remove_if_wealth_index_negative=True)
    terminal = dcf_obj.wealth_index("fv", include_negative_values=False)[dcf_obj.parent.symbol].iloc[-1]
    initial_investment = dcf_obj.cashflow_parameters.initial_investment
    n_months = dcf_obj.parent.ror.shape[0]

    v = np.empty(n_months + 1, dtype=float)
    v[0] = -initial_investment
    v[1:] = -cash_flow.reindex(dcf_obj.parent.ror.index).fillna(0.0).to_numpy()
    v[-1] += terminal

    t = np.arange(n_months + 1, dtype=float)

    def npv(rate):
        return float((v * (1.0 + rate) ** (-t)).sum())

    monthly = optimize.brentq(npv, -1.0 + 1e-9, 1e6, xtol=1e-12, maxiter=200)
    return (1.0 + monthly) ** _MONTHS_PER_YEAR - 1.0


def test_irr_matches_brentq_reference_indexation(pf_no_inflation):
    ind = ok.IndexationStrategy(pf_no_inflation)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -500
    ind.indexation = DEFAULT_DISCOUNT_RATE
    pf_no_inflation.dcf.cashflow_parameters = ind
    assert pf_no_inflation.dcf.irr() == pytest.approx(_reference_irr(pf_no_inflation.dcf), abs=1e-8)


def test_irr_matches_brentq_reference_percentage(pf_no_inflation):
    pc = ok.PercentageStrategy(pf_no_inflation)
    pc.initial_investment = 50_000
    pc.frequency = "half-year"
    pc.percentage = -0.05
    pf_no_inflation.dcf.cashflow_parameters = pc
    assert pf_no_inflation.dcf.irr() == pytest.approx(_reference_irr(pf_no_inflation.dcf), abs=1e-8)
