from pytest import approx
import pandas as pd

from okama.settings import DEFAULT_DISCOUNT_RATE


def test_dcf_discount_rate(portfolio_dcf, portfolio_dcf_no_inflation, portfolio_dcf_discount_rate):
    assert portfolio_dcf.discount_rate == approx(0.05548, rel=1e-2)  # average inflation
    assert portfolio_dcf_no_inflation.discount_rate == DEFAULT_DISCOUNT_RATE  # no inflation
    assert portfolio_dcf_discount_rate.discount_rate == approx(0.08, abs=1e-3)  # defined discount_rate


def test_wealth_index(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.wealth_index.iloc[0, 0] == 10_000
    assert portfolio_dcf_indexation.wealth_index.iloc[-1, 0] == approx(11047.67, rel=1e-2)


def test_wealth_index_with_assets(portfolio_dcf_percentage):
    df = portfolio_dcf_percentage.wealth_index_with_assets
    assert df.shape == (62, 4)
    # discounted initial investments
    assert portfolio_dcf_percentage.wealth_index_with_assets.iloc[0, 0] == approx(76339.3156, rel=1e-2)
    # FV
    assert portfolio_dcf_percentage.wealth_index_with_assets.iloc[-1, 0] == approx(232477.6576, rel=1e-2)

def test_survival_period_hist(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.survival_period_hist(threshold=0) == approx(5.1, rel=1e-2)

def test_survival_date_hist(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.survival_date_hist(threshold=0) == pd.Timestamp('2020-01-31 00:00:00')

def test_initial_investment_pv(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.initial_investment_pv == approx(7633.93, rel=1e-2)

def test_initial_investment_fv(portfolio_dcf_percentage):
    assert portfolio_dcf_percentage.initial_investment_fv == approx(171594.5442, rel=1e-2)

