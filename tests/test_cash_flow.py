import numpy as np
import pandas as pd
import pytest
from pytest import approx
from pytest import mark
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_series_equal, assert_frame_equal

import okama as ok
from okama.common.error import LongRollingWindowLengthError, RollingWindowLengthBelowOneYearError

from tests import conftest

# DCF Methods
def test_dcf_discount_rate(
    portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation, portfolio_cashflows_NO_inflation_NO_discount_rate
):
    assert portfolio_cashflows_inflation.discount_rate == approx(0.0554, abs=1e-3)  # average inflation
    assert portfolio_cashflows_NO_inflation.discount_rate == approx(0.09, abs=1e-3)  # defined discount rate
    assert portfolio_cashflows_NO_inflation_NO_discount_rate.discount_rate == approx(0.05, abs=1e-3)  # default rate


def test_dcf_wealth_index(portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation):
    assert portfolio_cashflows_inflation.dcf.wealth_index.iloc[-1, 0] == approx(179950.30, rel=1e-2)
    assert portfolio_cashflows_inflation.dcf.wealth_index.iloc[-1, 1] == approx(100050.78, rel=1e-2)
    assert portfolio_cashflows_NO_inflation.dcf.wealth_index.iloc[-1, 0] == approx(152642.54, rel=1e-2)


def test_dcf_survival_date(portfolio_cashflows_inflation):
    assert portfolio_cashflows_inflation.dcf.survival_date_hist == pd.to_datetime("2020-01-31")


def test_dcf_cashflow_pv(portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation_NO_discount_rate):
    assert portfolio_cashflows_inflation.dcf.cashflow_pv == approx(-76.33, rel=1e-2)
    assert portfolio_cashflows_NO_inflation_NO_discount_rate.dcf.cashflow_pv == approx(-78.35, rel=1e-2)


def test_dcf_initial_amount_pv(portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation_NO_discount_rate):
    assert portfolio_cashflows_inflation.dcf.initial_amount_pv == approx(76339.31, rel=1e-2)
    assert portfolio_cashflows_NO_inflation_NO_discount_rate.dcf.initial_amount_pv == approx(78352.61, rel=1e-2)


def test_dcf_survival_period(portfolio_cashflows_inflation):
    assert portfolio_cashflows_inflation.dcf.survival_period_hist == approx(5.1, rel=1e-2)


@mark.parametrize(
    "distribution, expected",
    [("norm", 93899.64), ("lognorm", 92155.15), ("t", 93123.36)],
)
def test_dcf_monte_carlo_wealth(portfolio_cashflows_inflation_large_cf, distribution, expected):
    result = portfolio_cashflows_inflation_large_cf.dcf.monte_carlo_wealth(
        first_value=100_000, distr=distribution, years=1, n=100
    )
    assert result.iloc[-1].mean() == approx(expected, rel=1e-1)


@mark.parametrize(
    "distribution, expected",
    [("norm", 6.2), ("lognorm", 6.2), ("t", 5.9)],
)
def test_dcf_monte_carlo_survival_period(portfolio_cashflows_inflation_large_cf, distribution, expected):
    result = portfolio_cashflows_inflation_large_cf.dcf.monte_carlo_survival_period(distr=distribution, years=25, n=100)
    assert result.mean() == approx(expected, rel=1e-1)


def test_find_the_largest_withdrawals_size():
    assert False
