"""Repeat rendering reuses only numerical pair tables with fresh financial inputs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import okama as ok
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset
from tests.test_rebalance_performance import returns_frame


@pytest.fixture()
def frontier(mocker) -> ok.EfficientFrontier:
    frame = returns_frame(48, 3)
    assets = [FakeAsset(name, frame[name]) for name in frame]
    mocker.patch("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset)
    mocker.patch("okama.settings.resolve_n_jobs", return_value=1)
    result = ok.EfficientFrontier(assets, n_points=5, rebalancing_strategy=ok.Rebalance())
    yield result
    plt.close("all")


def test_replot_changes_rendering_without_reoptimizing_pairs(frontier, mocker) -> None:
    optimize = mocker.spy(ok.EfficientFrontier, "_get_ef_points")
    first = frontier.plot_pair_ef(figsize=(5, 4))
    assert optimize.call_count == 3
    frontier.labels = "name"
    second = frontier.plot_pair_ef(tickers="names", figsize=(7, 5))
    assert optimize.call_count == 3
    assert first is not second
    assert second.figure.get_size_inches().tolist() == [7, 5]
    for a, b in zip(first.lines, second.lines, strict=True):
        np.testing.assert_array_equal(a.get_xydata(), b.get_xydata())


def test_mutable_source_asset_refreshes_only_affected_pairs(frontier, mocker) -> None:
    optimize = mocker.spy(ok.EfficientFrontier, "_get_ef_points")
    before = frontier._get_pair_frontiers()
    assert optimize.call_count == 3
    first_asset = next(iter(frontier.asset_obj_dict.values()))
    first_asset.ror.iloc[2] += 0.03
    after = frontier._get_pair_frontiers()
    assert optimize.call_count == 5
    assert not before[0].equals(after[0])
    pd.testing.assert_frame_equal(before[2], after[2], check_exact=True)


@pytest.mark.parametrize("change", ["period", "points", "bounds", "dates"])
def test_financial_configuration_invalidates_pair_tables(frontier, mocker, change: str) -> None:
    optimize = mocker.spy(ok.EfficientFrontier, "_get_ef_points")
    frontier._get_pair_frontiers()
    if change == "period":
        frontier.rebalancing_strategy.period = "quarter"
    elif change == "points":
        frontier.n_points = 4
    elif change == "bounds":
        frontier.bounds = ((0.0, 1.1),) * 3
    else:
        frontier.first_date = frontier.first_date + pd.DateOffset(months=1)
    frontier._get_pair_frontiers()
    assert optimize.call_count == 6


def test_cached_pair_tables_are_not_exposed_to_caller_mutation(frontier, mocker) -> None:
    optimize = mocker.spy(ok.EfficientFrontier, "_get_ef_points")
    first = frontier._get_pair_frontiers()
    expected = first[0].copy(deep=True)
    first[0].iloc[0, 0] = -999
    second = frontier._get_pair_frontiers()
    assert optimize.call_count == 3
    pd.testing.assert_frame_equal(second[0], expected, check_exact=True)


def test_custom_parent_preserves_base_pair_construction_without_persistent_cache(frontier, mocker) -> None:
    class CustomFrontier(ok.EfficientFrontier):
        pass

    frontier.__class__ = CustomFrontier
    optimize = mocker.spy(ok.EfficientFrontier, "_get_ef_points")
    first = frontier._get_pair_frontiers()
    second = frontier._get_pair_frontiers()
    assert optimize.call_count == 6
    for actual, expected in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
