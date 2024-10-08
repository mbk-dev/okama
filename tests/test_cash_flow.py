from pytest import approx

from okama.settings import DEFAULT_DISCOUNT_RATE


def test_dcf_discount_rate(portfolio_dcf, portfolio_dcf_no_inflation, portfolio_dcf_discount_rate):
    assert portfolio_dcf.discount_rate == approx(0.05548, rel=1e-2)  # average inflation
    assert portfolio_dcf_no_inflation.discount_rate == DEFAULT_DISCOUNT_RATE  # no inflation
    assert portfolio_dcf_discount_rate.discount_rate == approx(0.08, abs=1e-3)  # defined discount_rate


def test_initial_investment_pv(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.initial_investment_pv == approx(7633.93, rel=1e-2)

