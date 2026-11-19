"""Bounded calendar preparation and original multi-period optimizer behavior."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from scipy.optimize import OptimizeResult

import okama as ok
from okama.common.error import ShortPeriodLengthError
from okama.common.helpers import helpers
from okama.frontier.errors import FrontierSolverError, InsufficientFrontierHistoryError
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset
from tests.helpers.multi_optimizer_reference import ScalarMultiOptimizer
from tests.helpers.rebalance_reference import ScalarCalendarRebalance
from tests.test_rebalance_performance import returns_frame


@pytest.fixture()
def frontiers(mocker) -> tuple[ok.EfficientFrontier, ScalarMultiOptimizer]:
    ror = returns_frame(48, 2)
    assets = [FakeAsset(name, ror[name]) for name in ror]
    mocker.patch("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset)
    mocker.patch("okama.settings.resolve_n_jobs", return_value=1)
    kwargs = {"ccy": "USD", "inflation": False, "n_points": 5}
    return (
        ok.EfficientFrontier(assets, rebalancing_strategy=ok.Rebalance(), **kwargs),
        ScalarMultiOptimizer(assets, rebalancing_strategy=ok.Rebalance(), **kwargs),
    )


def test_optimizer_groups_returns_once_instead_of_per_trial(frontiers, mocker) -> None:
    current, _ = frontiers
    target = helpers.Frame.get_cagr(current.assets_ror).mean()
    groupby = mocker.spy(pd.DataFrame, "groupby")
    point = current.minimize_risk(target)
    assert point["Weights"].shape == (2,)
    assert groupby.call_count <= 2


def test_frontier_reports_insufficient_finite_common_months(
    frontiers: tuple[ok.EfficientFrontier, ScalarMultiOptimizer], mocker: MockerFixture
) -> None:
    current, _ = frontiers
    history = current.assets_ror.iloc[:12].copy()
    history.iloc[0, 0] = np.nan
    current._assets_ror = history
    optimize = mocker.patch(
        "okama.frontier.multi_period.minimize",
        side_effect=AssertionError("Optimization started before the common history check"),
    )

    with pytest.raises(InsufficientFrontierHistoryError) as failure:
        _ = current.ef_points

    assert failure.value.common_months == 11
    assert failure.value.common_first_month == str(history.index[1])
    assert failure.value.common_last_month == str(history.index[-1])
    optimize.assert_not_called()


def test_frontier_reports_missing_overlap_with_currency_history(mocker: MockerFixture) -> None:
    ror = returns_frame(24, 2)
    assets = [FakeAsset(ror.columns[0], ror.iloc[:, 0], currency="EUR"), FakeAsset(ror.columns[1], ror.iloc[:, 1])]

    def currency(symbol: str, **kwargs: object) -> FakeCurrencyAsset:
        if symbol == "EURUSD.FX":
            return FakeCurrencyAsset(symbol, first_date="1990-01", last_date="1991-01")
        return FakeCurrencyAsset(symbol)

    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=currency)

    with pytest.raises(ShortPeriodLengthError, match="No monthly returns in USD"):
        ok.EfficientFrontier(assets, ccy="USD", inflation=False)


@pytest.mark.parametrize("method", ["minimize_risk", "_maximize_risk"])
@pytest.mark.parametrize("solver_success", [False, True])
def test_frontier_rejects_unverified_solver_target(
    frontiers: tuple[ok.EfficientFrontier, ScalarMultiOptimizer],
    mocker: MockerFixture,
    method: str,
    solver_success: bool,
) -> None:
    frontier, _ = frontiers
    weights = np.array([0.5, 0.5])
    target = frontier._get_cagr(weights) + 0.001
    global_max = frontier.global_max_return_portfolio
    mocker.patch.object(
        ok.EfficientFrontier, "global_max_return_portfolio", new_callable=mocker.PropertyMock, return_value=global_max
    )

    def solve(objective: Callable[[np.ndarray], float], *args: object, **kwargs: object) -> OptimizeResult:
        return OptimizeResult(success=solver_success, x=weights, fun=objective(weights), nit=1)

    mocker.patch("okama.frontier.multi_period.minimize", side_effect=solve)

    with pytest.raises(FrontierSolverError, match="No solution found for target CAGR"):
        getattr(frontier, method)(target)


@pytest.mark.parametrize("period", ["none", "month", "quarter", "half-year", "year"])
@pytest.mark.parametrize("weights", [[0.4, 0.6], [0.4 + 1.49e-8, 0.6], [1.0, 0.0], [-0.1, 1.1]])
def test_prepared_trial_returns_match_original_arithmetic(frontiers, period: str, weights: list[float]) -> None:
    current, _ = frontiers
    current.rebalancing_strategy = ok.Rebalance(period=period)
    expected = ScalarCalendarRebalance(period=period).return_ror_ts_ef(np.array(weights), current.assets_ror)
    actual = current._get_portfolio_ror_ts(np.array(weights))
    pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("period", ["none", "month", "quarter", "half-year", "year"])
def test_full_frontier_matches_original_weights_and_solver_diagnostics(frontiers, period: str) -> None:
    current, original = frontiers
    for frontier in frontiers:
        frontier.rebalancing_strategy = ok.Rebalance(period=period)
    pd.testing.assert_frame_equal(current.ef_points, original.ef_points, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(current.ef_points["Weights"], original.ef_points["Weights"], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("change", ["returns", "period"])
def test_trial_cache_refreshes_after_mutable_inputs_change(frontiers, change: str) -> None:
    current, _ = frontiers
    weights = np.array([0.4, 0.6])
    before = current._get_portfolio_ror_ts(weights).copy()
    if change == "returns":
        current.assets_ror.iloc[3, 0] += 0.07
    else:
        current.rebalancing_strategy.period = "quarter"
    expected = ScalarCalendarRebalance(period=current.rebalancing_strategy.period).return_ror_ts_ef(
        weights, current.assets_ror
    )
    actual = current._get_portfolio_ror_ts(weights)
    assert not before.equals(expected)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("assets", [3, 16])
@pytest.mark.parametrize("period", ["none", "month", "quarter", "half-year", "year"])
def test_single_weight_reductions_match_pandas_for_wide_frames(assets: int, period: str) -> None:
    frame = returns_frame(121, assets)
    rebalance = ok.Rebalance(period=period)
    prepared = rebalance._prepare_ef_data(frame)
    weights = np.arange(1.0, assets + 1)
    weights /= weights.sum()
    weights[0] += 1.49e-8
    expected = ScalarCalendarRebalance(period=period).return_ror_ts_ef(weights, frame)
    np.testing.assert_array_equal(prepared.return_ror_single(weights, frame.iloc[0] @ weights), expected.to_numpy())


@pytest.mark.parametrize("case", ["nan", "total_loss", "gap", "float32", "nullable", "extreme", "duplicate"])
def test_unsupported_history_preserves_original_return_series(frontiers, mocker, case: str) -> None:
    current, _ = frontiers
    if case == "nan":
        current.assets_ror.iloc[4, 0] = np.nan
    elif case == "total_loss":
        current.assets_ror.iloc[4, :] = -1.0
    elif case == "gap":
        current._assets_ror = current.assets_ror.drop(current.assets_ror.index[4])
    elif case in ("float32", "nullable"):
        current._assets_ror = current.assets_ror.astype("Float64" if case == "nullable" else "float32")
    elif case == "duplicate":
        current._assets_ror.index = current.assets_ror.index[:-1].append(current.assets_ror.index[-2:-1])
    else:
        current.assets_ror.iloc[:12, :] = 1e100
    weights = np.array([0.4, 0.6])
    scalar = mocker.spy(ok.Rebalance, "return_ror_ts_ef")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expected = ScalarCalendarRebalance(period="year").return_ror_ts_ef(weights, current.assets_ror)
        actual = current._get_portfolio_ror_ts(weights)
    assert scalar.call_count == 1
    pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("change", ["returns", "period", "bounds", "points", "full_frontier", "labels"])
def test_cached_frontier_refreshes_at_external_calculation_boundary(frontiers, change: str) -> None:
    current, original = frontiers
    _ = current.ef_points
    for frontier in frontiers:
        if change == "returns":
            frontier.assets_ror.iloc[2, 0] += 0.1
        elif change == "period":
            frontier.rebalancing_strategy.period = "quarter"
        elif change == "bounds":
            frontier.bounds = ((0.0, 1.1), (0.0, 1.0))
        elif change == "points":
            frontier.n_points = 3
        elif change == "full_frontier":
            frontier.full_frontier = False
        else:
            frontier.labels = "name"
    pd.testing.assert_frame_equal(current.ef_points, original.ef_points, rtol=1e-12, atol=1e-12)


def test_complete_frontier_prepares_calendar_only_once(frontiers, mocker) -> None:
    current, _ = frontiers
    prepare = mocker.spy(ok.Rebalance, "_prepare_ef_data")
    first = current.ef_points
    assert prepare.call_count == 1
    assert current.ef_points is first
    assert prepare.call_count == 1
    assert current._optimizer_depth == 0


def test_infeasible_solver_preserves_failure_and_releases_calculation_scope(frontiers) -> None:
    for frontier in frontiers:
        frontier.bounds = ((0.7, 1.0), (0.7, 1.0))
        with pytest.raises(RuntimeError, match="No solution found for target CAGR value"):
            frontier.minimize_risk(0.05)
        assert frontier._optimizer_depth == 0


def test_original_rounded_weight_cache_is_retained(frontiers) -> None:
    current, _ = frontiers
    first = current._get_portfolio_ror_ts(np.array([0.4, 0.6]))
    second = current._get_portfolio_ror_ts(np.array([0.4 + 1e-12, 0.6]))
    assert first is second


def test_parallel_point_workers_keep_full_frontier_equivalence(frontiers, mocker) -> None:
    current, original = frontiers
    mocker.patch("okama.settings.resolve_n_jobs", return_value=2)
    pd.testing.assert_frame_equal(current.ef_points, original.ef_points, rtol=1e-12, atol=1e-12)


def test_subclass_keeps_scalar_trial_evaluation(frontiers, mocker) -> None:
    class CustomFrontier(ok.EfficientFrontier):
        pass

    current, _ = frontiers
    current.__class__ = CustomFrontier
    scalar = mocker.spy(ok.Rebalance, "return_ror_ts_ef")
    weights = np.array([0.4, 0.6])
    expected = ScalarCalendarRebalance(period="year").return_ror_ts_ef(weights, current.assets_ror)
    pd.testing.assert_series_equal(current._get_portfolio_ror_ts(weights), expected, check_exact=True)
    assert scalar.call_count == 1


@pytest.mark.parametrize("period", ["none", "month", "quarter", "half-year", "year"])
@pytest.mark.parametrize("assets", [2, 3])
def test_fragmented_history_preserves_scalar_arithmetic_after_layout_change(
    frontiers, mocker, period: str, assets: int
) -> None:
    rng = np.random.default_rng(58)
    frame = pd.DataFrame(
        rng.normal(0.005, 0.03, (121, assets)),
        index=pd.period_range("2000-02", periods=121, freq="M"),
        columns=[f"A{i}" for i in range(assets)],
    )
    for column in frame:
        frame[column] = frame[column].copy()
    current = ok.EfficientFrontier(
        [FakeAsset(name, frame[name]) for name in frame], rebalancing_strategy=ok.Rebalance(period=period)
    )
    current._assets_ror = frame.copy(deep=True)
    weights = np.array([0.2 + 1.49e-8, 0.8] if assets == 2 else [0.2 + 1.49e-8, 0.3, 0.5])
    current._get_portfolio_ror_ts(weights)
    scalar = mocker.spy(ok.Rebalance, "return_ror_ts_ef")
    current._assets_ror = frame
    expected = ScalarCalendarRebalance(period=period).return_ror_ts_ef(weights, frame)
    actual = current._get_portfolio_ror_ts(weights)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    assert scalar.call_count == (0 if assets == 2 else 1)
