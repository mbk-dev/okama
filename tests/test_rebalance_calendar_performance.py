"""Complete calendar Result comparisons against original scalar period arithmetic."""

import numpy as np
import pandas as pd
import pytest

import okama as ok
from tests.helpers.rebalance_reference import ScalarCalendarRebalance
from tests.test_rebalance_performance import legacy_warnings, returns_frame


def assert_results_equal(actual, expected) -> None:
    pd.testing.assert_series_equal(
        actual.portfolio_wealth_index, expected.portfolio_wealth_index, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(actual.assets_wealth_indexes, expected.assets_wealth_indexes, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(actual.events, expected.events)


def test_calendar_dataframe_construction_does_not_grow_per_period(mocker) -> None:
    short, long = returns_frame(12), returns_frame(36)
    strategy = ok.Rebalance(period="month")
    constructors = mocker.spy(pd.DataFrame, "__init__")
    with legacy_warnings():
        strategy.wealth_ts([0.4, 0.3, 0.3], short, True)
        short_calls = constructors.call_count
        constructors.reset_mock()
        result = strategy.wealth_ts([0.4, 0.3, 0.3], long, True)
    assert len(result.portfolio_wealth_index) == 37
    assert constructors.call_count <= short_calls + 8


@pytest.mark.parametrize("period", ["month", "quarter", "half-year", "year"])
@pytest.mark.parametrize(
    "absolute, relative", [(None, None), (0.05, None), (None, 0.1), (0.05, 0.1), (0.0, None), (None, 0.0)]
)
@pytest.mark.parametrize("with_assets", [False, True])
def test_calendar_full_result_matches_original(
    period: str, absolute: float | None, relative: float | None, with_assets: bool
) -> None:
    ror = returns_frame(36)
    weights = [0.4, 0.3, 0.3]
    parameters = {"period": period, "abs_deviation": absolute, "rel_deviation": relative}
    with legacy_warnings():
        expected = ScalarCalendarRebalance(**parameters).wealth_ts(weights, ror, with_assets)
    actual = ok.Rebalance(**parameters).wealth_ts(weights, ror, with_assets)
    assert_results_equal(actual, expected)


@pytest.mark.parametrize("period, event_month", [("quarter", "2020-03"), ("half-year", "2020-06"), ("year", "2020-12")])
def test_partial_first_period_checks_threshold_only_at_calendar_boundary(period: str, event_month: str) -> None:
    index = pd.period_range("2020-03", periods=15, freq="M")
    ror = pd.DataFrame({"A": [0.5] + [0.0] * 14, "B": [0.0] * 15}, index=index)
    result = ok.Rebalance(period=period, abs_deviation=0.05, rel_deviation=0.05).wealth_ts([0.5, 0.5], ror, True)
    assert list(result.events.index) == [pd.Period(event_month, "M")]
    assert list(result.events) == ["abs"]


def test_zero_calendar_threshold_keeps_unconditional_calendar_events() -> None:
    ror = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=pd.period_range("2020-01", periods=3, freq="M")
    )
    result = ok.Rebalance(period="month", abs_deviation=0.0).wealth_ts([0.5, 0.5], ror)
    pd.testing.assert_series_equal(result.events, pd.Series(["calendar", "calendar"], index=ror.index[:2]))


@pytest.mark.parametrize("case", ["nan", "gap", "float32", "float32_weights", "zero_balance", "zero_target"])
def test_calendar_edge_inputs_preserve_original_behavior(case: str) -> None:
    ror = returns_frame(36, 2)
    weights = [0.5, 0.5]
    if case == "nan":
        ror.iloc[1, 0] = np.nan
    elif case == "gap":
        ror = ror.drop(ror.index[4])
    elif case == "float32":
        ror = ror.astype("float32")
    elif case == "float32_weights":
        weights = np.array([0.1, 0.9], dtype="float32")
    elif case == "zero_balance":
        ror.iloc[0, :] = -1.0
    else:
        weights = [1.0, 0.0]
    parameters = {"period": "year", "abs_deviation": 0.05, "rel_deviation": 0.1}
    with legacy_warnings():
        expected = ScalarCalendarRebalance(**parameters).wealth_ts(weights, ror, True)
        actual = ok.Rebalance(**parameters).wealth_ts(weights, ror, True)
    assert_results_equal(actual, expected)


def test_missing_entire_calendar_period_keeps_existing_error() -> None:
    ror = returns_frame(6, 2).drop(pd.Period("2020-05", "M"))
    with legacy_warnings(), pytest.raises(IndexError):
        ScalarCalendarRebalance(period="month").wealth_ts([0.5, 0.5], ror)
    with legacy_warnings(), pytest.raises(IndexError):
        ok.Rebalance(period="month").wealth_ts([0.5, 0.5], ror)
