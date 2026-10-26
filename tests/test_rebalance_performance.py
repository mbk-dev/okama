"""Full financial Result equivalence for the conditional Rebalance fast path."""

import warnings
from contextlib import contextmanager
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

import okama as ok


@contextmanager
def legacy_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
        yield


@pytest.fixture(scope="module")
def baseline_class() -> type:
    from tests.helpers.rebalance_reference import ScalarConditionalRebalance

    return ScalarConditionalRebalance


def returns_frame(months: int = 36, assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(127)
    frame = pd.DataFrame(
        rng.normal(0.005, 0.08, (months, assets)),
        index=pd.period_range("2020-03", periods=months, freq="M", name="Date"),
        columns=[f"A{i}" for i in range(assets)],
    )
    frame.columns.name = "Symbols"
    return frame


@pytest.mark.parametrize(
    "absolute, relative, expected",
    [(0.05, None, (True, True)), (None, 0.08, (True, False)), (0.2, 0.5, (False, False))],
)
def test_threshold_check_accepts_array_values(
    absolute: float | None, relative: float | None, expected: tuple[bool, bool]
) -> None:
    rebalance = ok.Rebalance(period="none", abs_deviation=absolute, rel_deviation=relative)
    try:
        result = rebalance._check_if_rebalancing_required(np.array([600.0, 400.0]), 1000.0, np.array([0.5, 0.5]))
    except AttributeError:
        pytest.fail("Threshold checks need ndarray support without constructing pandas Series")
    assert result == expected
    assert all(type(value) is bool for value in result)


def test_conditional_container_work_does_not_grow_per_month(mocker) -> None:
    short, long = returns_frame(12), returns_frame(36)
    strategy = ok.Rebalance(period="none", abs_deviation=0.05)
    constructors = mocker.spy(pd.Series, "__init__")
    with legacy_warnings():
        strategy.wealth_ts([0.4, 0.3, 0.3], short)
        short_calls = constructors.call_count
        constructors.reset_mock()
        result = strategy.wealth_ts([0.4, 0.3, 0.3], long)
    assert len(result.portfolio_wealth_index) == 37
    assert constructors.call_count <= short_calls + 8


@pytest.mark.parametrize(
    "absolute, relative", [(0.05, None), (None, 0.1), (0.02, 0.04), (0.0, None), (None, 0.0), (0.9, 2.0)]
)
@pytest.mark.parametrize("with_assets", [False, True])
def test_full_conditional_result_matches_frozen_source(
    baseline_class, absolute: float | None, relative: float | None, with_assets: bool
) -> None:
    ror = returns_frame()
    weights = [0.4, 0.3, 0.3]
    parameters = {"period": "none", "abs_deviation": absolute, "rel_deviation": relative}
    with legacy_warnings():
        expected = baseline_class(**parameters).wealth_ts(weights, ror, with_assets)
    actual = ok.Rebalance(**parameters).wealth_ts(weights, ror, with_assets)
    pd.testing.assert_series_equal(
        actual.portfolio_wealth_index, expected.portfolio_wealth_index, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(actual.assets_wealth_indexes, expected.assets_wealth_indexes, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(actual.events, expected.events)


def test_event_dates_and_absolute_priority_are_preserved() -> None:
    index = pd.period_range("2020-01", periods=3, freq="M")
    ror = pd.DataFrame({"A": [0.5, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=index)
    result = ok.Rebalance(period="none", abs_deviation=0.05, rel_deviation=0.05).wealth_ts([0.5, 0.5], ror, True)
    pd.testing.assert_series_equal(result.events, pd.Series(["abs"], index=index[:1]))
    pd.testing.assert_frame_equal(
        result.assets_wealth_indexes,
        pd.DataFrame(
            {"A": [500.0, 750.0, 625.0, 625.0], "B": [500.0, 500.0, 625.0, 625.0]},
            index=pd.period_range("2019-12", periods=4, freq="M"),
        ),
    )


@pytest.mark.parametrize("absolute, relative", [(0.25, None), (None, 0.5)])
def test_threshold_boundary_requires_strict_exceedance(absolute: float | None, relative: float | None) -> None:
    ror = pd.DataFrame({"A": [2.0, 0.0], "B": [0.0, 0.0]}, index=pd.period_range("2020-01", periods=2, freq="M"))
    result = ok.Rebalance(period="none", abs_deviation=absolute, rel_deviation=relative).wealth_ts([0.5, 0.5], ror)
    assert result.events.empty


@pytest.mark.parametrize(
    "case", ["missing_values", "zero_balance", "zero_target", "gaps", "integer_weights", "float32_weights"]
)
def test_conditional_edge_cases_match_frozen_source(baseline_class, case: str) -> None:
    ror = returns_frame(18, 2)
    weights = [0.5, 0.5]
    if case == "missing_values":
        ror.iloc[1, 0] = np.nan
        ror.iloc[4, :] = np.nan
    elif case == "zero_balance":
        ror.iloc[0, :] = -1.0
    elif case == "zero_target":
        weights = [1.0, 0.0]
    elif case == "gaps":
        ror = ror.drop(ror.index[[1, 4, 5]])
    elif case == "float32_weights":
        weights = np.array([0.1, 0.9], dtype=np.float32)
    else:
        weights = [1, 0]
    parameters = {"period": "none", "abs_deviation": 0.05, "rel_deviation": 0.1}
    with legacy_warnings():
        expected = baseline_class(**parameters).wealth_ts(weights, ror, True)
        actual = ok.Rebalance(**parameters).wealth_ts(weights, ror, True)
    pd.testing.assert_series_equal(
        actual.portfolio_wealth_index, expected.portfolio_wealth_index, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(actual.assets_wealth_indexes, expected.assets_wealth_indexes, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(actual.events, expected.events)


@pytest.mark.parametrize("case", ["float32", "indexed_weights", "duplicate_dates", "unsorted_dates"])
def test_nonstandard_inputs_preserve_scalar_fallback(baseline_class, case: str) -> None:
    ror = returns_frame(12, 2)
    weights = [0.5, 0.5]
    if case == "float32":
        ror = ror.astype("float32")
    elif case == "indexed_weights":
        weights = pd.Series(weights)
    elif case == "duplicate_dates":
        ror.index = pd.PeriodIndex([ror.index[0], *ror.index[:-1]], name="Date")
    else:
        ror = ror.iloc[::-1]
    parameters = {"period": "none", "abs_deviation": 0.02, "rel_deviation": 0.1}
    with legacy_warnings():
        expected = baseline_class(**parameters).wealth_ts(weights, ror, True)
        actual = ok.Rebalance(**parameters).wealth_ts(weights, ror, True)
    pd.testing.assert_series_equal(
        actual.portfolio_wealth_index, expected.portfolio_wealth_index, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(actual.assets_wealth_indexes, expected.assets_wealth_indexes, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(actual.events, expected.events)


def test_custom_subclass_keeps_pandas_threshold_hook() -> None:
    class CustomRebalance(ok.Rebalance):
        def _check_if_rebalancing_required(self, values: pd.Series, total: float, weights: list) -> tuple[bool, bool]:
            assert isinstance(values, pd.Series)
            return True, True

    ror = returns_frame(4, 2)
    with legacy_warnings():
        result = CustomRebalance(period="none", abs_deviation=0.1).wealth_ts([0.5, 0.5], ror, True)
    assert len(result.events) == len(ror) - 1
    assert (result.events == "abs").all()


def test_conditional_rebalance_keeps_input_data_and_readonly_weights_unchanged() -> None:
    ror = returns_frame(12, 3)
    original = ror.copy(deep=True)
    weights = np.array([0.4, 0.3, 0.3])
    weights.setflags(write=False)
    result = ok.Rebalance(period="none", abs_deviation=0.05).wealth_ts(weights, ror, True)
    assert len(result.assets_wealth_indexes) == len(ror) + 1
    pd.testing.assert_frame_equal(ror, original)
    np.testing.assert_array_equal(weights, [0.4, 0.3, 0.3])
