# from copy import deepcopy
# from pathlib import Path
#
# import pytest
# import okama as ok
#
# data_folder = Path(__file__).parent / "data"
#
# # Portfolios in AssetList
# @pytest.fixture(scope="package")
# def init_portfolio_values():
#     return dict(
#         assets=["RGBITR.INDX", "MCFTR.INDX"],  # index values are better as they are not changing (adjusted_close)
#         ccy="RUB",
#         first_date="2015-01",
#         last_date="2020-01",
#         inflation=True,
#         rebalancing_strategy=ok.Rebalance(period="year"),
#         symbol="pf1.PF",
#     )
#
# @pytest.fixture(scope="package")
# def portfolio_short_history(init_portfolio_values):
#     _portfolio_short_history = deepcopy(init_portfolio_values)
#     _portfolio_short_history["first_date"] = "2019-02"
#     return ok.Portfolio(**_portfolio_short_history)
#
# @pytest.fixture(scope="package")
# def portfolio_dividends(init_portfolio_values):
#     _portfolio_dividends = deepcopy(init_portfolio_values)
#     _portfolio_dividends["assets"] = ["SBER.MOEX", "T.US", "GNS.LSE"]
#     return ok.Portfolio(**_portfolio_dividends)
#
#
# @pytest.fixture(scope="class")
# def assets_from_db():
#     return ["USDRUB.CBR", "MCFTR.INDX"]
#
#
# @pytest.fixture(scope="class")
# def _init_asset_list(request, portfolio_short_history, portfolio_dividends, assets_from_db) -> None:
#     request.cls.asset_list_with_portfolio = ok.AssetList(
#         assets=[portfolio_short_history] + assets_from_db,
#         ccy="USD",
#     )
#
#     request.cls.asset_list_with_portfolio_dividends = ok.AssetList(
#         assets=[portfolio_dividends] + assets_from_db,
#         ccy="USD",
#     )
#
#     request.cls.asset_list = ok.AssetList(
#         assets=assets_from_db,
#         ccy="RUB",
#         first_date="2019-01",
#         last_date="2020-01",
#         inflation=True,
#     )
#     request.cls.asset_list_lt = ok.AssetList(
#         assets=assets_from_db,
#         ccy="RUB",
#         first_date="2003-03",
#         last_date="2020-01",
#         inflation=True,
#     )
#     request.cls.asset_list_st = ok.AssetList(
#         assets=assets_from_db,
#         ccy="RUB",
#         first_date="2019-01",
#         last_date="2019-05",
#         inflation=False,
#     )
#     request.cls.asset_list_no_infl = ok.AssetList(
#         assets=assets_from_db,
#         ccy="RUB",
#         first_date="2019-01",
#         last_date="2020-01",
#         inflation=False,
#     )
#     request.cls.currencies = ok.AssetList(
#         ["RUBUSD.FX", "EURUSD.FX", "CNYUSD.FX"],
#         ccy="USD",
#         first_date="2019-01",
#         last_date="2020-01",
#         inflation=True,
#     )
#     request.cls.spy = ok.AssetList(first_date="2000-01", last_date="2002-01", inflation=True)
#     request.cls.spy_rub = ok.AssetList(first_date="2000-01", last_date="2002-01", inflation=True, ccy="RUB")
#     request.cls.real_estate = ok.AssetList(
#         assets=["RUS_SEC.RE", "MOW_PR.RE"],
#         ccy="RUB",
#         first_date="2010-01",
#         last_date="2015-01",
#         inflation=True,
#     )
#


import pytest  # noqa: I001

import numpy as np
import pandas as pd

# Re-export helper classes for backward compatibility in tests
from tests.helpers.factories import (
    ListDefaults as _ListDefaults,
    FakeAsset as _FakeAsset,
    FakeCurrencyAsset as _FakeCurrencyAsset,
)


@pytest.fixture
def list_basic_patches(mocker):
    """Basic patching for two simple assets A.US and B.US with short series."""
    dm = _ListDefaults()

    # Patch ListMaker to return our prebuilt dict of Asset-like objects
    fake_assets = {
        "A.US": _FakeAsset("A.US", dm.ror_a, currency="USD"),
        "B.US": _FakeAsset("B.US", dm.ror_b, currency="USD"),
    }
    m_get_dict = mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)

    # Patch currency Asset used inside ListMaker.__init__ (self._currency)
    m_currency_asset = mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    yield {
        "defaults": dm,
        "m_get_dict": m_get_dict,
        "m_currency_asset": m_currency_asset,
    }


# Note: "synthetic_env" is now defined globally in tests/conftest.py.
# If you need custom environment in asset_list tests, create a new fixture with a distinct name.


@pytest.fixture
def synthetic_env2(mocker):
    """Alternative RNG seed and structure for robustness tests (24 months)."""
    rng = np.random.default_rng(20241004)
    idx = pd.period_range("2020-01", periods=24, freq="M")

    a1 = pd.Series(rng.normal(0.01 / 12, 0.05, size=len(idx)), index=idx, name="IDX.US")
    a2 = pd.Series(rng.normal(0.008 / 12, 0.04, size=len(idx)), index=idx, name="A.US")
    a3 = pd.Series(0.4 * a1.values + rng.normal(0, 0.02, size=len(idx)), index=idx, name="B.US")

    fake_assets = {
        "IDX.US": _FakeAsset("IDX.US", a1, currency="USD", name="Index"),
        "A.US": _FakeAsset("A.US", a2, currency="USD", name="Asset A"),
        "B.US": _FakeAsset("B.US", a3, currency="USD", name="Asset B"),
    }
    m_get_dict = mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)
    m_currency_asset = mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    yield {
        "index": idx,
        "series": {"IDX.US": a1, "A.US": a2, "B.US": a3},
        "m_get_dict": m_get_dict,
        "m_currency_asset": m_currency_asset,
    }
