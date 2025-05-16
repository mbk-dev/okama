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
        (None, None, 2501.89, 1039.87, "rel", "2018-09"),
    ],
    ids=["Only abs deviation", "Only rel deviation", "None & None deviations"],
)
def test_wealth_ts_rebalancing_conditional(abs_d, rel_d, exp1, exp2, exp3, exp4):
    portfolio_not_rebalanced = ok.Portfolio(
        ['RGBITR.INDX', 'MCFTR.INDX'],
        first_date="2015-01",
        last_date="2020-01",
        ccy='RUB', inflation=True)
    rb = ok.Rebalance(period="none", abs_deviation=abs_d, rel_deviation=rel_d)
    ror = portfolio_not_rebalanced.assets_ror
    target_weights = portfolio_not_rebalanced.weights
    ws = rb.wealth_ts(target_weights, ror, calculate_assets_wealth_indexes=True)
    assert ws.portfolio_wealth_index.iloc[-1] == approx(exp1, rel=1e-4)
    assert ws.assets_wealth_indexes.iloc[-1, 0] == approx(exp2, rel=1e-4)
    if not ws.events.empty:
        assert ws.events.iloc[-1] == exp3
        assert ws.events.index[-1] == pd.Period(exp4, freq="M")