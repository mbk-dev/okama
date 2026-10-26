"""Remaining optimizer equivalence and invariant-statistics work bounds."""

import numpy as np
import pytest

import okama as ok
from tests.helpers.optimizer_reference import ScalarOptimizer


@pytest.fixture()
def frontiers(synthetic_env) -> tuple[ok.EfficientFrontierSingle, ScalarOptimizer]:
    kwargs = {"ccy": "USD", "inflation": False, "n_points": 8}
    return (
        ok.EfficientFrontierSingle(["IDX.US", "A.US", "B.US"], **kwargs),
        ScalarOptimizer(["IDX.US", "A.US", "B.US"], **kwargs),
    )


def call_optimizer(frontier, method: str):
    if method == "minimize_risk":
        target = float(frontier.assets_ror.mean().mean())
        return frontier.minimize_risk(target, monthly_return=True)
    if method.startswith("tangency"):
        return frontier.get_tangency_portfolio(rf_return=0.01, rate_of_return=method.split("/", 1)[1])
    return frontier.optimize_return(option=method.split("/", 1)[1])


@pytest.mark.parametrize(
    "method", ["minimize_risk", "tangency/cagr", "tangency/mean_return", "return/max", "return/min"]
)
def test_optimizer_statistics_do_not_repeat_per_evaluation(frontiers, mocker, method: str) -> None:
    frontier, _ = frontiers
    target = float(frontier.assets_ror.mean().mean())
    mean = mocker.spy(frontier.assets_ror, "mean")
    cov = mocker.spy(frontier.assets_ror, "cov")
    if method == "minimize_risk":
        point = frontier.minimize_risk(target, monthly_return=True)
    else:
        point = call_optimizer(frontier, method)
    assert point
    assert cov.call_count == (0 if method == "tangency/cagr" else 1)
    assert mean.call_count <= (2 if method == "minimize_risk" else 1)


@pytest.mark.parametrize(
    "method", ["minimize_risk", "tangency/cagr", "tangency/mean_return", "return/max", "return/min"]
)
@pytest.mark.parametrize("case", ["normal", "nan", "bounded", "name", "local_name"])
def test_prepared_optimizers_match_original_methods(frontiers, method: str, case: str) -> None:
    current, original = frontiers
    for frontier in frontiers:
        if case == "nan":
            frontier.assets_ror.iloc[3, 0] = np.nan
            frontier.assets_ror.iloc[5, 1] = np.nan
        elif case == "bounded":
            frontier.bounds = ((0.05, 0.7), (0.05, 0.8), (0.0, 0.8))
        elif case in ("name", "local_name"):
            frontier.labels = case
    expected, actual = call_optimizer(original, method), call_optimizer(current, method)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("monthly", [False, True])
@pytest.mark.parametrize("tolerance", [1e-8, 1e-10])
def test_minimize_risk_preserves_target_conversion_and_tolerance(frontiers, monthly: bool, tolerance: float) -> None:
    current, original = frontiers
    target = float(current.assets_ror.mean().mean())
    target = target if monthly else (1 + target) ** 12 - 1
    expected = original.minimize_risk(target, monthly_return=monthly, tolerance=tolerance)
    actual = current.minimize_risk(target, monthly_return=monthly, tolerance=tolerance)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["minimize_risk", "tangency/cagr", "tangency/mean_return", "return/max"])
def test_optimizer_preparation_is_fresh_after_input_and_bounds_changes(frontiers, method: str) -> None:
    current, original = frontiers
    first = call_optimizer(current, method)
    for frontier in frontiers:
        frontier.assets_ror.iloc[:, 0] *= 2
        frontier.bounds = ((0.1, 0.6), (0.1, 0.7), (0.0, 0.8))
    actual, expected = call_optimizer(current, method), call_optimizer(original, method)
    assert first.keys() == actual.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["minimize_risk", "tangency/cagr", "tangency/mean_return", "return/max"])
def test_optimizer_failure_message_remains_compatible(frontiers, method: str) -> None:
    for frontier in frontiers:
        frontier.bounds = ((0.7, 1.0),) * 3
        with pytest.raises(RuntimeError, match="No solutions were found"):
            call_optimizer(frontier, method)


@pytest.mark.parametrize("option", ["invalid", "MAX", "Min"])
def test_return_optimizer_option_validation_is_unchanged(frontiers, option: str) -> None:
    for frontier in frontiers:
        with pytest.raises(ValueError, match="option should"):
            frontier.optimize_return(option=option)


@pytest.mark.parametrize("option", ["CAGR", "MEAN_RETURN"])
def test_tangency_option_normalization_is_unchanged(frontiers, option: str) -> None:
    current, original = frontiers
    actual = current.get_tangency_portfolio(rate_of_return=option)
    expected = original.get_tangency_portfolio(rate_of_return=option)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("full", [False, True])
def test_complete_frontier_matches_original_optimizers(frontiers, full: bool) -> None:
    import pandas as pd

    current, original = frontiers
    current.full_frontier = original.full_frontier = full
    pd.testing.assert_frame_equal(current.ef_points, original.ef_points, rtol=1e-12, atol=1e-12)
