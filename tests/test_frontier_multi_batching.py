"""Public multi-period sampling equivalence and bounded preparation work."""

import numpy as np
import pandas as pd
import pytest

import okama as ok
from okama.common.helpers import helpers
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset
from tests.helpers.rebalance_reference import ScalarCalendarRebalance
from tests.test_rebalance_performance import legacy_warnings, returns_frame


@pytest.fixture()
def frontier(mocker) -> ok.EfficientFrontier:
    ror = returns_frame(36, 3)
    assets = [FakeAsset(name, ror[name], name=f"Name {name}") for name in ror]
    mocker.patch("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset)
    return ok.EfficientFrontier(assets, ccy="USD", inflation=False)


def scalar_points(frontier: ok.EfficientFrontier, weights: np.ndarray) -> pd.DataFrame:
    rebalance = ScalarCalendarRebalance(period=frontier.rebalancing_strategy.period)
    rows = []
    for row_weights in weights:
        ror = rebalance.return_ror_ts_ef(row_weights, frontier.assets_ror)
        point = dict(zip(frontier._asset_labels(frontier._labels_mode), row_weights, strict=True))
        point["Risk"] = helpers.Float.annualize_risk(ror.std(), ror.mean())
        point["CAGR"] = helpers.Frame.get_cagr(ror)
        rows.append(point)
    return helpers.Frame.change_columns_order(pd.DataFrame.from_records(rows), ["Risk", "CAGR"])


@pytest.mark.parametrize("method", ["get_monte_carlo", "get_grid_portfolios"])
def test_return_history_is_grouped_once_per_sampling_request(frontier, mocker, method: str) -> None:
    groups = mocker.spy(pd.DataFrame, "groupby")
    result = getattr(frontier, method)(**({"n": 12} if method == "get_monte_carlo" else {"step": 0.2}))
    assert len(result) > 1
    assert groups.call_count <= 2


@pytest.mark.parametrize("period", ["none", "month", "quarter", "half-year", "year"])
@pytest.mark.parametrize("method", ["get_monte_carlo", "get_grid_portfolios"])
@pytest.mark.parametrize("labels", ["ticker", "name", "local_name"])
def test_public_sampling_matches_original_scalar_periods(frontier, period: str, method: str, labels: str) -> None:
    frontier.rebalancing_strategy = ok.Rebalance(period=period)
    frontier.labels = labels
    frontier.bounds = ((0.1, 0.5), (0.2, 0.6), (0.0, 0.7))
    np.random.seed(127)
    weights = np.stack(
        helpers.Float.get_random_weights(17, 3, frontier.bounds)
        if method == "get_monte_carlo"
        else helpers.Float.get_grid_weights(3, step=0.1, bounds=frontier.bounds)
    )
    np.random.seed(127)
    with legacy_warnings():
        expected = scalar_points(frontier, weights)
    actual = getattr(frontier, method)(**({"n": 17} if method == "get_monte_carlo" else {"step": 0.1}))
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    assert len(frontier._ror_cache) == 0


@pytest.mark.parametrize("case", ["short", "nan", "constant", "total_loss", "gap"])
def test_exceptional_inputs_keep_original_risk_and_cagr(frontier, case: str) -> None:
    if case == "short":
        frontier._assets_ror = frontier.assets_ror.iloc[:6].copy()
    elif case == "nan":
        frontier._assets_ror.iloc[2, 0] = np.nan
    elif case == "constant":
        frontier._assets_ror.iloc[:, :] = 0.01
    elif case == "total_loss":
        frontier._assets_ror.iloc[0, :] = -1.0
    else:
        frontier._assets_ror = frontier.assets_ror.drop(frontier.assets_ror.index[3])
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.5, bounds=frontier.bounds))
    with legacy_warnings(), np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expected = scalar_points(frontier, weights)
        actual = frontier.get_grid_portfolios(step=0.5)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    if case == "short":
        assert actual["CAGR"].dtype == object
        assert all(value is None for value in actual["CAGR"])


def test_duplicate_and_metric_asset_names_keep_original_columns(frontier) -> None:
    frontier.names = dict(zip(frontier.symbols, ["Same", "Same", "CAGR"], strict=True))
    frontier.labels = "name"
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.5, bounds=frontier.bounds))
    with legacy_warnings():
        expected = scalar_points(frontier, weights)
    pd.testing.assert_frame_equal(frontier.get_grid_portfolios(step=0.5), expected, rtol=1e-12, atol=1e-12)


def test_negative_grid_weights_keep_scalar_results(frontier) -> None:
    frontier.bounds = ((-0.5, 1.5), (0.0, 1.0), (-0.5, 1.5))
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.5, bounds=frontier.bounds))
    with legacy_warnings(), np.errstate(invalid="ignore"):
        expected = scalar_points(frontier, weights)
        actual = frontier.get_grid_portfolios(step=0.5)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


def test_empty_monte_carlo_keeps_existing_validation(frontier) -> None:
    with pytest.raises(KeyError):
        frontier.get_monte_carlo(n=0)


@pytest.mark.parametrize("months", [6, 36])
def test_batch_sizes_preserve_output_and_partial_last_chunk(frontier, monkeypatch, months: int) -> None:
    frontier._assets_ror = frontier.assets_ror.iloc[:months].copy()
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.1, bounds=frontier.bounds))
    with legacy_warnings():
        expected = scalar_points(frontier, weights)
    for size in (1, 7, 1024):
        monkeypatch.setattr(frontier, "_BATCH_SIZE", size)
        actual = frontier.get_grid_portfolios(step=0.1)
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


def test_period_matrix_element_budget_limits_weight_chunks(frontier, monkeypatch, mocker) -> None:
    monkeypatch.setattr(frontier, "_BATCH_SIZE", 1000)
    monkeypatch.setattr(frontier, "_BATCH_ELEMENTS", len(frontier.assets_ror) * 2)
    stacks = mocker.spy(np, "stack")
    actual = frontier.get_monte_carlo(n=9)
    assert len(actual) == 9
    assert stacks.call_count >= 5
    assert all(len(call.args[0]) <= 2 for call in stacks.call_args_list)


def test_history_exceeding_element_budget_uses_scalar_fallback(frontier, monkeypatch) -> None:
    monkeypatch.setattr(frontier, "_BATCH_ELEMENTS", 1)
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.5, bounds=frontier.bounds))
    with legacy_warnings():
        expected = scalar_points(frontier, weights)
        actual = frontier.get_grid_portfolios(step=0.5)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("period", ["month", "year"])
@pytest.mark.parametrize("volatility", [1e-6, 1e-4])
def test_low_risk_portfolios_keep_scalar_annualization(frontier, period: str, volatility: float) -> None:
    values = np.random.default_rng(127).normal(0.01, volatility, frontier.assets_ror.shape)
    frontier._assets_ror.iloc[:, :] = values
    frontier.rebalancing_strategy = ok.Rebalance(period=period)
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.2, bounds=frontier.bounds))
    with legacy_warnings(), np.errstate(invalid="ignore"):
        expected = scalar_points(frontier, weights)
        actual = frontier.get_grid_portfolios(step=0.2)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


def test_low_risk_corner_uses_scalar_fallback_without_disabling_other_ports(frontier, mocker) -> None:
    frontier._assets_ror.iloc[:, 0] = 0.001
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.25, bounds=frontier.bounds))
    with legacy_warnings(), np.errstate(invalid="ignore"):
        expected = scalar_points(frontier, weights)
        scalar = mocker.spy(ok.Rebalance, "return_ror_ts_ef")
        actual = frontier.get_grid_portfolios(step=0.25)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    assert 0 < scalar.call_count < len(weights)


@pytest.mark.parametrize("period", ["month", "year"])
def test_high_mean_low_variance_keeps_scalar_risk_precision(frontier, period: str) -> None:
    frontier._assets_ror.iloc[:, :] = np.random.default_rng(127).normal(2.0, 0.002, frontier.assets_ror.shape)
    frontier.rebalancing_strategy = ok.Rebalance(period=period)
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.5, bounds=frontier.bounds))
    with legacy_warnings():
        expected = scalar_points(frontier, weights)
        actual = frontier.get_grid_portfolios(step=0.5)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
