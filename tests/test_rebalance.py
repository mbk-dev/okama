import warnings

import pandas as pd
import pandas.testing as pdt
import pytest

import okama as ok


def _synthetic_returns(months: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [0.01] * months, "B": [0.005] * months},
        index=pd.period_range("2020-01", periods=months, freq="M"),
    )


def _assert_event_series_types(events: pd.Series) -> None:
    assert isinstance(events.index, pd.PeriodIndex)
    assert events.index.freqstr == "M"
    assert pd.api.types.is_string_dtype(events.dtype)


def test_validate_period_failing():
    with pytest.raises(ValueError):
        ok.Rebalance(period="not existing")


def test_validate_abs_deviation_big_failing():
    with pytest.raises(ValueError, match=r"Absolute deviation must be less or equal to 1."):
        ok.Rebalance(abs_deviation=1.5)


def test_validate_abs_deviation_small_failing():
    with pytest.raises(ValueError, match=r"Absolute deviation must be positive."):
        ok.Rebalance(abs_deviation=-100)


def test_validate_rel_deviation_failing():
    with pytest.raises(ValueError, match=r"Relative deviation must be positive."):
        ok.Rebalance(rel_deviation=-100)


def test_repr(init_rebalance_no_rebalancing):
    value = pd.Series(
        {
            "period": "none",
            "abs_deviation": None,
            "rel_deviation": None,
        }
    )
    assert repr(init_rebalance_no_rebalancing) == repr(value)


@pytest.mark.parametrize("weights", [[0.6, 0.4], pd.Series([0.6, 0.4], index=["A", "B"])])
def test_calendar_rebalance_events_use_string_values_and_period_index_without_pandas4warning(
    weights: list[float] | pd.Series,
) -> None:
    returns = _synthetic_returns(24)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ok.Rebalance(period="year").wealth_ts(weights, returns)
    assert caught == []
    _assert_event_series_types(result.events)
    expected = pd.Series(["calendar"], index=pd.PeriodIndex(["2020-12"], freq="M"), dtype="str")
    pdt.assert_series_equal(result.events, expected)


@pytest.mark.parametrize("weights", [[0.6, 0.4], pd.Series([0.6, 0.4], index=["A", "B"])])
def test_conditional_rebalance_events_use_string_values_and_period_index_without_pandas4warning(
    weights: list[float] | pd.Series,
) -> None:
    returns = _synthetic_returns(24)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ok.Rebalance(period="none", abs_deviation=0.01).wealth_ts(weights, returns)
    assert caught == []
    _assert_event_series_types(result.events)
    expected = pd.Series(
        ["abs", "abs"],
        index=pd.PeriodIndex(["2020-09", "2021-06"], freq="M"),
        dtype="str",
    )
    pdt.assert_series_equal(result.events, expected)


@pytest.mark.parametrize(
    "strategy, months",
    [
        (ok.Rebalance(period="none"), 24),
        (ok.Rebalance(period="year"), 1),
        (ok.Rebalance(period="none", abs_deviation=0.5), 6),
    ],
)
def test_empty_rebalance_events_use_string_dtype_and_period_index_without_pandas4warning(
    strategy: ok.Rebalance, months: int
) -> None:
    returns = pd.DataFrame(
        {"A": [0.01] * months, "B": [0.01] * months},
        index=pd.period_range("2020-01", periods=months, freq="M"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = strategy.wealth_ts([0.5, 0.5], returns)
    assert caught == []
    assert result.events.empty
    _assert_event_series_types(result.events)


def test_rebalance_by_condition_events_with_mock(mocker):
    mocker.patch.object(ok.Rebalance, "_check_if_rebalancing_required", return_value=(True, True))

    idx = pd.period_range("2020-01", "2020-03", freq="M")
    ror = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.0]}, index=idx)
    target_weights = [0.5, 0.5]

    r = ok.Rebalance(period="none", abs_deviation=0.01)
    res = r.wealth_ts(target_weights=target_weights, ror=ror, calculate_assets_wealth_indexes=True)

    assert not res.events.empty
    assert (res.events == "abs").any()


def test_check_if_rebalancing_required_series_abs():
    # Series: trigger by absolute deviation
    r = ok.Rebalance(period="none", abs_deviation=0.05)
    assets = pd.Series([600.0, 400.0], index=["A", "B"])  # weights: 0.6 and 0.4
    portfolio_total = 1000.0
    target = [0.5, 0.5]

    cond, cond_abs = r._check_if_rebalancing_required(assets, portfolio_total, target)
    assert cond is True
    assert cond_abs is True


def test_check_if_rebalancing_required_series_rel():
    # Series: trigger only by relative deviation
    r = ok.Rebalance(period="none", rel_deviation=0.08)
    assets = pd.Series([550.0, 450.0], index=["A", "B"])  # weights: 0.55 and 0.45
    portfolio_total = 1000.0
    target = [0.5, 0.5]

    cond, cond_abs = r._check_if_rebalancing_required(assets, portfolio_total, target)
    assert cond is True
    assert cond_abs is False


def test_check_if_rebalancing_required_series_no_trigger():
    # Series: no trigger for both thresholds
    r = ok.Rebalance(period="none", abs_deviation=0.2, rel_deviation=0.5)
    assets = pd.Series([520.0, 480.0], index=["A", "B"])  # weights: 0.52 and 0.48
    portfolio_total = 1000.0
    target = [0.5, 0.5]

    cond, cond_abs = r._check_if_rebalancing_required(assets, portfolio_total, target)
    assert cond is False
    assert cond_abs is False


def test_check_if_rebalancing_required_dataframe_last_row_rel():
    # DataFrame + Series: only the last row is considered (iloc[-1])
    r = ok.Rebalance(period="none", rel_deviation=0.05)
    idx = pd.period_range("2020-01", "2020-02", freq="M")
    # First row without deviation, second row violates the relative threshold
    awi = pd.DataFrame(
        [[500.0, 500.0], [560.0, 440.0]], index=idx, columns=["A", "B"]
    )  # end-of-period weights: 0.5/0.5, then 0.56/0.44
    pwi = pd.Series([1000.0, 1000.0], index=idx)
    target = [0.5, 0.5]

    cond, cond_abs = r._check_if_rebalancing_required(awi, pwi, target)
    assert cond is True
    assert cond_abs is False


def test_assets_weights_ts_no_rebalancing_manual():
    # No rebalancing at all: assets weights should be equal to manual calculation
    idx = pd.period_range("2020-01", "2020-03", freq="M")
    ror = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.0]}, index=idx)
    target_weights = [0.6, 0.4]

    # no calendar rebalancing and no conditions -> pure buy & hold
    r = ok.Rebalance(period="none", abs_deviation=None, rel_deviation=None)
    weights_ts = r.assets_weights_ts(target_weights=target_weights, ror=ror)

    # Manual calculation
    initial_inv = 1000.0
    first_date = ror.index[0]
    first_wealth_index_date = first_date - 1
    tw = pd.Series(target_weights, index=ror.columns)
    initial_allocation = tw * initial_inv
    assets_wealth_indexes_manual = initial_allocation * (1 + ror).cumprod()
    portfolio_wealth_index_manual = assets_wealth_indexes_manual.sum(axis=1)

    # Insert the initial point
    assets_wealth_indexes_manual.loc[first_wealth_index_date] = initial_allocation
    assets_wealth_indexes_manual = assets_wealth_indexes_manual.sort_index()
    portfolio_wealth_index_manual.loc[first_wealth_index_date] = initial_inv
    portfolio_wealth_index_manual = portfolio_wealth_index_manual.sort_index()

    weights_manual = assets_wealth_indexes_manual.divide(portfolio_wealth_index_manual, axis=0)

    # Structure checks
    assert list(weights_ts.columns) == list(weights_manual.columns)
    pdt.assert_index_equal(weights_ts.index, weights_manual.index)

    # Values check (allow tiny numeric noise)
    pdt.assert_frame_equal(weights_ts, weights_manual, atol=1e-12, rtol=1e-12)


def test_return_ror_ts_no_rebalancing_manual():
    # Verify portfolio return series equals manual pct_change of wealth index in no-rebalancing case
    idx = pd.period_range("2020-01", "2020-03", freq="M")
    ror_df = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.0]}, index=idx)
    target_weights = [0.6, 0.4]

    r = ok.Rebalance(period="none")
    ror_series = r.return_ror_ts(target_weights=target_weights, ror=ror_df)

    # Manual wealth index including the pre-first-period point
    initial_inv = 1000.0
    tw = pd.Series(target_weights, index=ror_df.columns)
    assets_wi_manual = (tw * initial_inv) * (1 + ror_df).cumprod()
    portfolio_wi_manual = assets_wi_manual.sum(axis=1)
    # Prepend the initial investment point so pct_change covers the first ROR date
    first_wealth_index_date = ror_df.index[0] - 1
    portfolio_wi_manual.loc[first_wealth_index_date] = initial_inv
    portfolio_wi_manual = portfolio_wi_manual.sort_index()

    manual_ror = portfolio_wi_manual.pct_change().dropna()

    pdt.assert_index_equal(ror_series.index, manual_ror.index)
    pdt.assert_series_equal(ror_series, manual_ror, atol=1e-12, rtol=1e-12)
