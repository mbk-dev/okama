import pytest
from pytest import approx
from pytest import mark

import pandas as pd

import okama as ok


@mark.rebalance
def test_validate_period_failing():
    with pytest.raises(ValueError):
        ok.Rebalance(period="not existing")

@mark.rebalance
def test_validate_abs_deviation_big_failing():
    with pytest.raises(ValueError, match=r"Absolute deviation must be less or equal to 1."):
        ok.Rebalance(abs_deviation=1.5)

@mark.rebalance
def test_validate_abs_deviation_big_failing():
    with pytest.raises(ValueError, match=r"Absolute deviation must be positive."):
        ok.Rebalance(abs_deviation=-100)

@mark.rebalance
def test_validate_rel_deviation_failing():
    with pytest.raises(ValueError, match=r"Relative deviation must be positive."):
        ok.Rebalance(rel_deviation=-100)


def test_repr(init_rebalance_no_rebalancing):
    value = pd.Series(
        dict(
            period="none",
            abs_deviation=None,
            rel_deviation=None,
        )
    )
    assert repr(init_rebalance_no_rebalancing) == repr(value)


def test_wealth_ts_no_rebalancing(init_rebalance_no_rebalancing, portfolio_not_rebalanced):
    ror = portfolio_not_rebalanced.assets_ror
    target_weights = portfolio_not_rebalanced.weights
    ws = init_rebalance_no_rebalancing.wealth_ts(target_weights, ror, calculate_assets_wealth_indexes=False)
    assert ws.portfolio_wealth_index.iloc[-1] == approx(2501.89, rel=1e-2)
    assert ws.assets_wealth_indexes.empty
    assert ws.events.empty

    ws = init_rebalance_no_rebalancing.wealth_ts(target_weights, ror, calculate_assets_wealth_indexes=True)
    assert ws.assets_wealth_indexes.iloc[-1, 0] == approx(1039.87, rel=1e-2)


@mark.parametrize(
    "abs_d, rel_d, exp1, exp2, exp3, exp4",
    [
        (0.01, None, 2497.98, 1248.80, "abs", "2019-12"),
        (None, 0.01, 2499.72, 1249.67, "rel", "2019-12"),
        (0.01, 0.01, 2499.72, 1249.67, "abs", "2019-12"),
    ],
    ids=["Only abs deviation", "Only rel deviation", "Abs & Rel deviations"],
)
def test_wealth_ts_rebalancing_conditional(portfolio_not_rebalanced, abs_d, rel_d, exp1, exp2, exp3, exp4):
    rb = ok.Rebalance(period="none", abs_deviation=abs_d, rel_deviation=rel_d)
    ror = portfolio_not_rebalanced.assets_ror
    target_weights = portfolio_not_rebalanced.weights
    ws = rb.wealth_ts(target_weights, ror, calculate_assets_wealth_indexes=True)
    assert ws.portfolio_wealth_index.iloc[-1] == approx(exp1, rel=1e-4)
    assert ws.assets_wealth_indexes.iloc[-1, 0] == approx(exp2, rel=1e-4)
    if not ws.events.empty:
        assert ws.events.iloc[-1] == exp3
        assert ws.events.index[-1] == pd.Period(exp4, freq="M")

@mark.parametrize(
    "period, abs_d, rel_d, exp1, exp2, exp3, exp4",
    [
        ("month", 0.01, None, 2497.98, 1248.80, "abs", "2019-12"),
        ("quarter", None, 0.01, 2492.13, 1245.88, "rel", "2019-12"),
        ("half-year", 0.01, 0.01, 2496.57, 1248.10, "abs", "2019-12"),
        ("year", None, None, 2490.84, 1245.23, "calendar", "2019-12"),
    ],
    ids=["month", "quarter", "half-year", "year"],
)
def test_wealth_ts_rebalancing_calendar(portfolio_not_rebalanced, period, abs_d, rel_d, exp1, exp2, exp3, exp4):
    rb = ok.Rebalance(period=period, abs_deviation=abs_d, rel_deviation=rel_d)
    ror = portfolio_not_rebalanced.assets_ror
    target_weights = portfolio_not_rebalanced.weights
    ws = rb.wealth_ts(target_weights, ror, calculate_assets_wealth_indexes=True)
    assert ws.portfolio_wealth_index.iloc[-1] == approx(exp1, rel=1e-4)
    assert ws.assets_wealth_indexes.iloc[-1, 0] == approx(exp2, rel=1e-4)
    if not ws.events.empty:
        assert ws.events.iloc[-1] == exp3
        assert ws.events.index[-1] == pd.Period(exp4, freq="M")

@mark.parametrize(
    "period, abs_d, rel_d, exp1, exp2",
    [
        ("none", None, None, 0.4156, 0.5843),
        ("month", 0.01, None, 0.499, 0.5000),
        ("quarter", None, 0.01, 0.4999, 0.5000),
        ("half-year", 0.01, 0.01, 0.4999, 0.5000),
        ("year", None, None, 0.4999, 0.5000),
    ],
    ids=["none", "month", "quarter", "half-year", "year"],
)
def test_assets_weights_ts(portfolio_not_rebalanced, period, abs_d, rel_d, exp1, exp2):
    rb = ok.Rebalance(period=period, abs_deviation=abs_d, rel_deviation=rel_d)
    ror = portfolio_not_rebalanced.assets_ror
    target_weights = portfolio_not_rebalanced.weights
    weights_ts = rb.assets_weights_ts(target_weights=target_weights, ror=ror)
    assert weights_ts.shape[1] == len(target_weights)
    assert weights_ts.iloc[-1, 0] == approx(exp1, abs=1e-2)
    assert weights_ts.iloc[-1, 1] == approx(exp2, abs=1e-2)

@mark.parametrize(
    "period, abs_d, rel_d, exp",
    [
        ("none", None, None, 2.5018),
        ("month", 0.01, None, 2.4979),
        ("quarter", None, 0.01, 2.4921),
        ("half-year", 0.01, 0.01, 2.4965),
        ("year", None, None, 2.4908),
    ],
    ids=["none", "month", "quarter", "half-year", "year"],
)
def test_return_ror_ts(portfolio_not_rebalanced, period, abs_d, rel_d, exp):
    rb = ok.Rebalance(period=period, abs_deviation=abs_d, rel_deviation=rel_d)
    ror = portfolio_not_rebalanced.assets_ror
    target_weights = portfolio_not_rebalanced.weights
    ror_ts = rb.return_ror_ts(target_weights=target_weights, ror=ror)
    assert (ror_ts + 1).prod() == approx(exp, abs=1e-4)