import socket

import pytest
import okama as ok
from pathlib import Path
import numpy as np
import pandas as pd

# Helper classes for Asset/currency mocks
from tests.asset_list.conftest import _FakeAsset, _FakeCurrencyAsset

data_folder = Path(__file__).parent / "data"

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fake_socket(*args, **kwargs):
        raise AssertionError("Network calls are disabled during tests!")

    monkeypatch.setattr(socket, "socket", fake_socket)

# Efficient Frontier Single Period
@pytest.fixture(scope="module")
def init_efficient_frontier_values1():
    return dict(
        assets=["MCFTR.INDX", "RGBITR.INDX"],
        ccy="RUB",
        first_date="2018-11",
        last_date="2020-02",
        inflation=True,
        n_points=2,
    )


@pytest.fixture(scope="module")
def init_efficient_frontier_values2():
    return dict(
        assets=["SPY.US", "AGG.US", "GLD.US"],
        ccy="USD",
        first_date="2010-01",
        last_date="2020-01",
        inflation=True,
        n_points=20,
        full_frontier=True,
    )


@pytest.fixture(scope="module")
def init_efficient_frontier(init_efficient_frontier_values1):
    return ok.EfficientFrontier(**init_efficient_frontier_values1)


@pytest.fixture(scope="module")
def init_efficient_frontier_bounds(init_efficient_frontier_values1):
    bounds = ((0.0, 0.5), (0.0, 1.0))
    return ok.EfficientFrontier(**init_efficient_frontier_values1, bounds=bounds)


@pytest.fixture(scope="module")
def init_efficient_frontier_three_assets(init_efficient_frontier_values2):
    return ok.EfficientFrontier(**init_efficient_frontier_values2)


# Efficient Frontier Multi-Period
@pytest.fixture(scope="module")
def init_efficient_frontier_reb():
    return ok.EfficientFrontierReb(
        assets=["SPY.US", "GLD.US"],
        ccy="USD",
        first_date="2019-01",
        last_date="2020-02",
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0, 1), (0, 1)),
        inflation=True,
        n_points=2,
        full_frontier=False,
    )


@pytest.fixture(scope="module")
def bounds_frontier_params():
    return dict(
        assets=["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"],
        ccy="USD",
        first_date="2019-01",
        last_date="2020-02",
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0, 1), (0, 1)),
        inflation=True,
    )


@pytest.fixture(scope="function")
def without_bounds_params():
    return dict(
        assets=["VOO.US", "GLD.US", "SCHA.US"],
        ccy="USD",
        first_date="2004-10",
        last_date="2020-10",
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


@pytest.fixture(scope="function")
def with_bounds_params():
    return dict(
        assets=["VOO.US", "GLD.US", "SCHA.US"],
        ccy="USD",
        first_date="2004-10",
        last_date="2020-10",
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0, 0.4), (0, 1), (0, 1)),
    )


@pytest.fixture(scope="module")
def _min_ratio_asset_when_none_params():
    return dict(
        assets=["SPY.US", "GLD.US"],
        ccy="USD",
        last_date="2020-10",
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


@pytest.fixture(scope="module")
def _min_ratio_asset_when_not_none_params():
    return dict(
        assets=["SPY.US", "MCFTR.INDX"],
        ccy="RUB",
        last_date="2025-03",
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


@pytest.fixture(scope="module")
def convex_frontier_params():
    return dict(
        assets=["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-11",
        rebalancing_strategy=ok.Rebalance(period="year"),
        n_points=5,
        verbose=True,
    )


@pytest.fixture(scope="module")
def nonconvex_frontier_params():
    return dict(
        assets=["SPY.US", "GLD.US", "VB.US", "RGBITR.INDX", "MCFTR.INDX"],
        ccy="RUB",
        first_date="2004-12",
        last_date="2020-12",
        rebalancing_strategy=ok.Rebalance(period="year"),
        n_points=5,
        verbose=True,
    )


@pytest.fixture(scope="module")
def init_bounds_frontier(bounds_frontier_params):
    return ok.EfficientFrontierReb(**bounds_frontier_params)


@pytest.fixture(scope="function")
def init_frontier_without_bounds(without_bounds_params):
    return ok.EfficientFrontierReb(**without_bounds_params)


@pytest.fixture(scope="function")
def init_frontier_with_bounds(with_bounds_params):
    return ok.EfficientFrontierReb(**with_bounds_params)


@pytest.fixture(scope="module")
def init_frontier_with_none(_min_ratio_asset_when_none_params):
    return ok.EfficientFrontierReb(**_min_ratio_asset_when_none_params)


@pytest.fixture(scope="module")
def init_frontier_with_not_none(_min_ratio_asset_when_not_none_params):
    return ok.EfficientFrontierReb(**_min_ratio_asset_when_not_none_params)


@pytest.fixture(scope="module")
def init_convex_frontier(convex_frontier_params):
    return ok.EfficientFrontierReb(**convex_frontier_params)


@pytest.fixture(scope="module")
def init_nonconvex_frontier(nonconvex_frontier_params):
    return ok.EfficientFrontierReb(**nonconvex_frontier_params)


# Rebalance
@pytest.fixture(scope="module")
def init_rebalance_no_rebalancing():
    return ok.Rebalance(
        period="none",
        abs_deviation=None,
        rel_deviation=None,
    )


# Global synthetic_env fixture available to all tests
@pytest.fixture
def synthetic_env(mocker):
    """Three assets over 24 months with deterministic correlation (global fixture).

    Makes the synthetic_env fixture available to all tests under tests/.
    Patches ListMaker._get_asset_obj_dict and the currency Asset to remove external dependencies.
    """
    rng = np.random.default_rng(12345)
    idx = pd.period_range("2020-01", periods=24, freq="M")

    a1 = pd.Series(rng.normal(0.01 / 12, 0.05, size=len(idx)), index=idx, name="IDX.US")
    a2 = pd.Series(rng.normal(0.008 / 12, 0.04, size=len(idx)), index=idx, name="A.US")
    a3_noise = rng.normal(0, 0.02, size=len(idx))
    a3 = pd.Series(0.5 * a1.values + a3_noise, index=idx, name="B.US")

    fake_assets = {
        "IDX.US": _FakeAsset("IDX.US", a1, currency="USD", name="Index"),
        "A.US": _FakeAsset("A.US", a2, currency="USD", name="Asset A"),
        "B.US": _FakeAsset("B.US", a3, currency="USD", name="Asset B"),
    }

    m_get_dict = mocker.patch(
        "okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets
    )
    m_currency_asset = mocker.patch(
        "okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset
    )

    yield {
        "index": idx,
        "series": {k: v for k, v in [("IDX.US", a1), ("A.US", a2), ("B.US", a3)]},
        "m_get_dict": m_get_dict,
        "m_currency_asset": m_currency_asset,
    }
