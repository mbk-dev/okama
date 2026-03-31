import pandas as pd
import pytest

import okama as ok
from okama.common.make_asset_list import ListMaker
from tests.helpers.factories import (
    FakeAsset as _FakeAsset,
    FakeCurrencyAsset as _FakeCurrencyAsset,
    ListDefaults as _ListDefaults,
)


class DummyList(ListMaker):
    """A minimal concrete subclass for testing the abstract ListMaker."""

    def __repr__(self):  # pragma: no cover - simple implementation so the class can be instantiated
        return f"DummyList({len(self.symbols)} assets)"


@pytest.fixture()
def two_assets_env(mocker):
    """Two deterministic assets A.US and B.US for 3 months, base currency USD."""
    dm = _ListDefaults()

    fake_assets = {
        "A.US": _FakeAsset("A.US", dm.ror_a, currency="USD"),
        "B.US": _FakeAsset("B.US", dm.ror_b, currency="USD"),
    }
    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)
    # Currency object used inside ListMaker.__init__ for self._currency
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    return {"dm": dm}


def test_basic_init_and_properties(two_assets_env):
    dm = two_assets_env["dm"]
    lm = DummyList(["A.US", "B.US"], ccy="USD", inflation=False)

    # symbols/tickers
    assert lm.symbols == ["A.US", "B.US"]
    assert lm.tickers == ["A", "B"]

    # indexes and return data are taken from mocks
    ror = lm.assets_ror
    assert list(ror.columns) == ["A.US", "B.US"]
    assert list(ror.index) == list(dm.ror_index)

    # base currency
    assert lm.currency == "USD"


def test_duplicates_are_removed_and_order_preserved(mocker):
    dm = _ListDefaults()
    fake_assets = {
        "A.US": _FakeAsset("A.US", dm.ror_a, currency="USD"),
        "B.US": _FakeAsset("B.US", dm.ror_b, currency="USD"),
    }
    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    lm = DummyList(["A.US", "B.US", "A.US"], ccy="USD", inflation=False)
    # Order of first occurrence is preserved, duplicates removed
    assert lm.symbols == ["A.US", "B.US"]


def test_len_iter_getitem(two_assets_env):
    lm = DummyList(["A.US", "B.US"], ccy="USD", inflation=False)
    assert len(lm) == 2

    # __getitem__
    first = lm[0]
    assert hasattr(first, "symbol") and first.symbol in {"A.US", "B.US"}

    # __iter__
    seen = [obj.symbol for obj in lm]
    assert set(seen) == {"A.US", "B.US"}


def test_validate_period_ok_and_fail(synthetic_env):
    # 24 months -> pl.years == 2
    lm = DummyList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)

    # valid: period of 1 year
    lm._validate_period(1)

    # invalid: period exceeds available history in years
    with pytest.raises(ValueError):
        lm._validate_period(3)


def test_get_asset_obj_dict_raises_for_short_history():
    """Test that _get_asset_obj_dict raises ShortPeriodLengthError for assets with insufficient history.

    This test directly calls the validation logic by importing the original module
    to avoid interference from patches in other parallel tests (e.g., synthetic_env).
    """
    # Import the module directly to get the original unpatched version
    import importlib
    import okama.common.make_asset_list

    # Force reload to ensure we get a fresh unpatched version
    importlib.reload(okama.common.make_asset_list)
    from okama.common.make_asset_list import ListMaker as UnpatchedListMaker

    PL = ok.settings.PeriodLength

    class TinyAsset:
        def __init__(self, symbol: str):
            self.symbol = symbol
            self.ror = pd.Series(dtype=float)  # Empty series (length 0)
            self.pl = PL(0, 2)  # 0 years and 2 months -> should raise ShortPeriodLengthError

    # Create a list with TinyAsset objects directly (instead of strings)
    # This way we bypass the Asset(symbol) construction and use our test objects
    tiny_assets = [TinyAsset("TINY1.US"), TinyAsset("TINY2.US")]

    with pytest.raises(ok.common.error.ShortPeriodLengthError):
        UnpatchedListMaker._get_asset_obj_dict(tiny_assets)  # staticmethod


@pytest.fixture()
def _inflation_env_for_listmaker(mocker):
    """Minimal inflation environment for DummyList(inflation=True)."""
    idx = pd.period_range("2020-01", periods=12, freq="M")
    ror = pd.Series(0.01, index=idx, name="A.US")

    # assets
    fake_assets = {"A.US": _FakeAsset("A.US", ror, currency="USD")}
    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    # inflation: constant monthly values
    infl_monthly = pd.Series(0.002, index=idx.to_timestamp(how="end"), name="USD.INFL")

    class _FakeInflation:
        def __init__(self, symbol: str, first_date=None, last_date=None):
            self.symbol = symbol
            self.first_date = infl_monthly.index[0].to_period("M").to_timestamp(how="start")
            self.last_date = infl_monthly.index[-1].to_period("M").to_timestamp(how="start")
            self.values_monthly = infl_monthly.to_period("M")

    mocker.patch("okama.common.make_asset_list.macro.Inflation", side_effect=_FakeInflation)
    return {"idx": idx, "infl": infl_monthly.to_period("M")}


def test_inflation_true_sets_fields_and_aligns_index(_inflation_env_for_listmaker):
    lm = DummyList(["A.US"], ccy="USD", inflation=True)

    assert hasattr(lm, "inflation")
    assert hasattr(lm, "inflation_ts")
    assert isinstance(lm.inflation_ts, pd.Series)
    assert list(lm.inflation_ts.index) == list(lm.assets_ror.index)
