"""Offline synthetic assets with a long history for FinPlan tests.

Plans span decades, so the shared 24-month `synthetic_env` fixture is too short
for the calendar backtest. This package-scoped fixture serves 600 months
(1976-01 through 2025-12) for any requested symbol, with no network access.
"""

import zlib  # noqa: I001

import numpy as np
import pandas as pd
import pytest

from tests.helpers.factories import FakeAsset, FakeCurrencyAsset, make_period_index


@pytest.fixture(scope="package", autouse=True)
def _finplan_offline_assets():
    idx = make_period_index(months=600, start="1976-01")

    def _series_for(symbol: str) -> pd.Series:
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        mu = 0.008 if symbol.startswith("EQ") else 0.003
        sigma = 0.045 if symbol.startswith("EQ") else 0.012
        return pd.Series(rng.normal(mu, sigma, size=len(idx)), index=idx, name=symbol)

    cache: dict[str, FakeAsset] = {}

    def _get_or_make(symbol: str) -> FakeAsset:
        if symbol not in cache:
            cache[symbol] = FakeAsset(symbol, _series_for(symbol), currency="USD")
        return cache[symbol]

    def _filtered_get_dict(symbols, first_date=None, last_date=None):
        result = {}
        for s in symbols:
            if hasattr(s, "symbol"):
                result[s.symbol] = s
            else:
                result[s] = _get_or_make(s)
        return result

    mp = pytest.MonkeyPatch()
    mp.setattr(
        "okama.common.make_asset_list.ListMaker._get_asset_obj_dict",
        staticmethod(_filtered_get_dict),
    )
    mp.setattr("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset)
    mp.setattr("okama.asset.Asset", FakeCurrencyAsset)
    try:
        yield {"index": idx}
    finally:
        mp.undo()


@pytest.fixture
def equity_portfolio():
    import okama as ok

    return ok.Portfolio(["EQ1.US", "EQ2.US"], weights=[0.7, 0.3], ccy="USD", inflation=False, symbol="eq.PF")


@pytest.fixture
def bond_portfolio():
    import okama as ok

    return ok.Portfolio(["BND1.US", "BND2.US"], weights=[0.5, 0.5], ccy="USD", inflation=False, symbol="bnd.PF")
