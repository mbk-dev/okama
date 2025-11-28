import socket
import random

import pytest
import okama as ok
from pathlib import Path
import numpy as np
import pandas as pd

# Helper classes and generators for Asset/currency mocks
from tests.helpers.factories import (
    FakeAsset,
    FakeCurrencyAsset,
    make_period_index,
    make_ror_series,
)

@pytest.fixture(scope="session", autouse=True)
def set_seed():
    """Set deterministic seeds for numpy and random across the test session."""
    np.random.seed(12345)
    random.seed(12345)


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Block network calls during tests to guarantee offline determinism."""
    mp = pytest.MonkeyPatch()

    def _deny(*args, **kwargs):
        raise AssertionError("Network calls are disabled during tests!")

    mp.setattr(socket, "socket", _deny)
    # Optionally block other entry points if needed:
    # import socket as _s
    # mp.setattr(_s, "create_connection", _deny)
    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def mock_macro():
    """Patch macro data access for *.INFL symbols with synthetic deterministic series."""
    from okama.api import data_queries as dq

    mp = pytest.MonkeyPatch()
    _orig_get_macro_ts = dq.QueryData.get_macro_ts
    _orig_get_symbol_info = dq.QueryData.get_symbol_info

    def _wrapped_get_macro_ts(symbol, first_date, last_date, period="M"):
        if isinstance(symbol, str) and symbol.endswith(".INFL"):
            idx = make_period_index(months=600, start="2000-01")
            s = make_ror_series(symbol, idx, base=0.004, amp=0.0015)
            if first_date or last_date:
                s = s.loc[first_date:last_date]
            if period != "M":
                raise AssertionError("Only monthly period is supported in tests")
            return s
        return _orig_get_macro_ts(symbol, first_date, last_date, period=period)

    def _wrapped_get_symbol_info(symbol):
        if isinstance(symbol, str) and symbol.endswith(".INFL"):
            ccy = symbol.split(".", 1)[0]
            return {
                "code": symbol,
                "name": f"Inflation {symbol}",
                "country": ccy,
                "currency": ccy,
                "type": "macro",
                "exchange": "N/A",
            }
        return _orig_get_symbol_info(symbol)

    mp.setattr(dq.QueryData, "get_macro_ts", staticmethod(_wrapped_get_macro_ts))
    mp.setattr(dq.QueryData, "get_symbol_info", staticmethod(_wrapped_get_symbol_info))
    yield
    mp.undo()



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
        "IDX.US": FakeAsset("IDX.US", a1, currency="USD", name="Index"),
        "A.US": FakeAsset("A.US", a2, currency="USD", name="Asset A"),
        "B.US": FakeAsset("B.US", a3, currency="USD", name="Asset B"),
    }

    # Return only requested symbols to keep shapes consistent with the user's input
    def _filtered_get_dict(symbols, first_date=None, last_date=None):
        """Return a dict with only the requested symbols/objects from fake_assets.

        Supports both symbol strings and already constructed Asset-like objects
        (with attribute `symbol`). This ensures that the number of assets in
        assets_ror matches the number of symbols passed to EfficientFrontier
        (and hence its bounds).

        Parameters
        ----------
        symbols : list
            List of symbols or Asset-like objects.
        first_date : str, optional
            First date parameter (ignored in mock, kept for signature compatibility).
        last_date : str, optional
            Last date parameter (ignored in mock, kept for signature compatibility).
        """
        result = {}
        for s in symbols:
            # If we receive an already constructed fake asset object (e.g., from plot_pair_ef)
            if hasattr(s, "symbol"):
                # Only replace if the symbol is in our fake_assets, otherwise pass through
                # This allows other tests to use custom asset objects for validation testing
                if s.symbol in fake_assets:
                    result[s.symbol] = s
                else:
                    # Pass through the original object for symbols not in fake_assets
                    # This is needed for tests that create their own test objects
                    from okama.common.make_asset_list import ListMaker
                    # Call the original unmocked method
                    original_result = ListMaker.__dict__['_get_asset_obj_dict'].__func__(
                        [s], first_date=first_date, last_date=last_date
                    )
                    result.update(original_result)
            else:
                # Otherwise, it's a symbol string -> take from our predefined fake_assets
                result[s] = fake_assets[s]
        return result

    m_get_dict = mocker.patch(
        "okama.common.make_asset_list.ListMaker._get_asset_obj_dict", side_effect=_filtered_get_dict
    )
    m_currency_asset = mocker.patch(
        "okama.common.make_asset_list.asset.Asset", side_effect=FakeCurrencyAsset
    )

    yield {
        "index": idx,
        "series": {k: v for k, v in [("IDX.US", a1), ("A.US", a2), ("B.US", a3)]},
        "m_get_dict": m_get_dict,
        "m_currency_asset": m_currency_asset,
    }
