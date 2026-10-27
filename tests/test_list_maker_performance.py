"""ListMaker performance contracts with real validation and offline asset data."""

import warnings
from unittest.mock import PropertyMock

import numpy as np
import pandas as pd
import pytest

import okama as ok
from okama.common import make_asset_list
from okama.common.error import ShortPeriodLengthError
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset


def make_asset(symbol: str = "A.US", months: int = 24) -> FakeAsset:
    index = pd.period_range("2020-01", periods=months, freq="M")
    return FakeAsset(symbol, pd.Series(0.0, index=index, name=symbol))


def test_ready_assets_are_validated_without_starting_a_thread_pool(mocker) -> None:
    objects = [make_asset("B.US"), make_asset("A.US")]
    pool = mocker.spy(make_asset_list, "Parallel")
    result = make_asset_list.ListMaker._get_asset_obj_dict(objects)
    assert list(result) == ["B.US", "A.US"]
    assert all(result[obj.symbol] is obj for obj in objects)
    assert pool.call_count == 0


@pytest.mark.parametrize("months", [1, 2])
def test_ready_assets_still_reject_short_history(months: int) -> None:
    with pytest.raises(ShortPeriodLengthError):
        make_asset_list.ListMaker._get_asset_obj_dict([make_asset("A.US"), make_asset("B.US", months)])


def test_ready_assets_keep_last_duplicate_and_first_key_order() -> None:
    first, last = make_asset("B.US"), make_asset("B.US")
    middle = make_asset("A.US")
    result = make_asset_list.ListMaker._get_asset_obj_dict([first, middle, last])
    assert list(result) == ["B.US", "A.US"]
    assert result["B.US"] is last
    assert result["A.US"] is middle


def test_mixed_assets_keep_threaded_loading_and_input_order(mocker) -> None:
    ready, loaded = make_asset("B.US"), make_asset("A.US")
    load = mocker.patch("okama.common.make_asset_list.asset.Asset", return_value=loaded)
    pool = mocker.spy(make_asset_list, "Parallel")
    result = make_asset_list.ListMaker._get_asset_obj_dict([ready, "A.US"], "2020-01", "2021-12")
    assert list(result) == ["B.US", "A.US"]
    assert result["B.US"] is ready
    assert result["A.US"] is loaded
    load.assert_called_once_with("A.US", first_date="2020-01", last_date="2021-12")
    assert pool.call_count == 1


@pytest.fixture()
def yield_list(mocker) -> ok.AssetList:
    assets = [make_asset("A.US"), make_asset("B.US")]
    for asset in assets:
        asset.ror.index.name = "Date"
        asset.close_monthly = pd.Series(100.0, index=asset.ror.index, name=asset.symbol)
        asset.dividends = pd.Series(1.0, index=asset.ror.index, name=asset.symbol)
    mocker.patch("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset)
    return ok.AssetList(assets, ccy="USD", inflation=False)


def scalar_dividend_yield(asset_list: ok.AssetList) -> pd.DataFrame:
    """Original date-slice algorithm, independent of the new rolling calculation."""
    import warnings

    frame = {}
    dividends = asset_list._get_assets_dividends(remove_forecast=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
        for tick in asset_list.symbols:
            monthly = dividends[tick].copy()
            if monthly.sum() == 0:
                frame[tick] = monthly
                continue
            asset = asset_list.asset_obj_dict[tick]
            prices = asset.close_monthly.loc[asset_list.first_date : asset_list.last_date]
            if asset.currency != asset_list.currency:
                prices = asset_list._adjust_price_to_currency_monthly(prices, asset.currency)
            monthly.index = monthly.index.to_timestamp(how="start")
            result = pd.Series(dtype=float)
            for date in prices.index.to_timestamp(how="end"):
                date_without_nanoseconds = pd.Timestamp(date.to_pydatetime(warn=False))
                start = date_without_nanoseconds - pd.DateOffset(months=12) + pd.Timedelta(date.nanosecond, unit="ns")
                result.at[date] = monthly[start:date].sum() / prices.loc[:date].iloc[-1]
            result.index = result.index.to_period("M")
            frame[tick] = result
    return pd.DataFrame(frame)


def test_monthly_yield_avoids_one_date_offset_per_price_month(yield_list, mocker) -> None:
    offsets = mocker.spy(pd, "DateOffset")
    result = yield_list.dividend_yield
    assert result.loc["2021-01", "A.US"] == pytest.approx(0.12)
    assert offsets.call_count == 0


def test_monthly_yield_uses_exact_trailing_twelve_months(yield_list) -> None:
    asset = yield_list.asset_obj_dict["A.US"]
    asset.dividends = pd.Series(np.arange(1.0, 25.0), index=asset.ror.index, name="A.US")
    result = yield_list.dividend_yield
    assert result.loc["2020-01", "A.US"] == pytest.approx(0.01)
    assert result.loc["2020-12", "A.US"] == pytest.approx(0.78)
    assert result.loc["2021-01", "A.US"] == pytest.approx(0.90)
    assert result.loc["2021-03", "A.US"] == pytest.approx(1.14)
    pd.testing.assert_frame_equal(result, scalar_dividend_yield(yield_list), rtol=1e-12, atol=1e-12)


def test_yield_preserves_calendar_window_with_missing_months(yield_list) -> None:
    asset = yield_list.asset_obj_dict["A.US"]
    asset.dividends = pd.Series(np.arange(1.0, 25.0), index=asset.ror.index, name="A.US").drop(
        [pd.Period("2020-05", "M"), pd.Period("2020-06", "M")]
    )
    asset.close_monthly = asset.close_monthly.drop([pd.Period("2020-08", "M"), pd.Period("2020-09", "M")])
    result = yield_list.dividend_yield
    assert result.loc["2021-01", "A.US"] == pytest.approx(0.79)
    assert pd.isna(result.loc["2020-08", "A.US"])
    pd.testing.assert_frame_equal(result, scalar_dividend_yield(yield_list), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("values", [[0.0] * 24, [1.0, -1.0] * 12])
def test_zero_net_dividends_keep_existing_values_without_prices(yield_list, values: list[float]) -> None:
    for asset in yield_list.asset_obj_dict.values():
        asset.dividends = pd.Series(values, index=asset.ror.index, name=asset.symbol)
        asset.close_monthly = None
    expected = pd.DataFrame(dict.fromkeys(yield_list.symbols, values), index=yield_list.assets_ror.index)
    pd.testing.assert_frame_equal(yield_list.dividend_yield, expected)


def test_yield_skips_nan_dividends_within_the_calendar_window(yield_list) -> None:
    asset = yield_list.asset_obj_dict["A.US"]
    asset.dividends.iloc[2:5] = np.nan
    result = yield_list.dividend_yield
    assert result.loc["2020-12", "A.US"] == pytest.approx(0.09)
    pd.testing.assert_frame_equal(result, scalar_dividend_yield(yield_list), rtol=1e-12, atol=1e-12)


def test_yield_zero_price_keeps_infinite_result(yield_list) -> None:
    yield_list.asset_obj_dict["A.US"].close_monthly.iloc[-1] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = yield_list.dividend_yield
        assert np.isinf(result.iloc[-1, 0])
        pd.testing.assert_frame_equal(result, scalar_dividend_yield(yield_list), rtol=1e-12, atol=1e-12)


def test_yield_cache_is_reused(yield_list) -> None:
    first = yield_list.dividend_yield
    assert yield_list.dividend_yield is first


def test_yield_forecast_dividends_remain_excluded(mocker) -> None:
    current = pd.Period.now("M")
    index = pd.period_range(current - 12, current + 12, freq="M")
    asset = FakeAsset("A.US", pd.Series(0.0, index=index, name="A.US"))
    asset.dividends = pd.Series(1.0, index=index, name="A.US")
    asset.close_monthly = pd.Series(100.0, index=index, name="A.US")
    mocker.patch("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset)
    asset_list = ok.AssetList([asset], ccy="USD", inflation=False)
    result = asset_list.dividend_yield
    assert result.loc[current, "A.US"] == pytest.approx(0.12)
    assert result.loc[current + 12, "A.US"] == 0.0


def test_yield_preserves_foreign_currency_conversion(yield_list, mocker) -> None:
    class CurrencyWithPrices(FakeCurrencyAsset):
        def __init__(self, symbol: str, first_date: str | None = None, last_date: str | None = None) -> None:
            super().__init__(symbol, first_date, last_date)
            self.close_monthly = pd.Series(np.linspace(1.5, 2.5, len(self.ror)), index=self.ror.index, name=symbol)

    mocker.patch("okama.common.make_asset_list.asset.Asset", CurrencyWithPrices)
    yield_list.asset_obj_dict["A.US"].currency = "EUR"
    result = yield_list.dividend_yield
    pd.testing.assert_frame_equal(result, scalar_dividend_yield(yield_list), rtol=1e-12, atol=1e-12)


def test_yield_all_nan_twelve_month_window_sums_to_zero(yield_list) -> None:
    yield_list.asset_obj_dict["A.US"].dividends.iloc[-12:] = np.nan
    result = yield_list.dividend_yield
    assert result.iloc[-1, 0] == 0.0
    pd.testing.assert_frame_equal(result, scalar_dividend_yield(yield_list), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["duplicate_dividends", "duplicate_prices", "nullable_dtype"])
def test_nonstandard_monthly_data_keeps_scalar_semantics(case: str) -> None:
    import warnings

    index = pd.period_range("2020-01", periods=3, freq="M")
    dividends = pd.Series([1.0, 2.0, 3.0], index=index)
    prices = pd.Series(100.0, index=index)
    expected = pd.Series([0.01, 0.03, 0.06], index=index)
    if case == "duplicate_dividends":
        dividends = pd.Series([1.0, 2.0, 3.0], index=pd.PeriodIndex([index[0], index[0], index[1]]))
        prices = prices.iloc[:2]
        expected = pd.Series([0.03, 0.06], index=index[:2])
    elif case == "duplicate_prices":
        dividends = dividends.iloc[:2]
        prices = pd.Series([100.0, 200.0, 100.0], index=pd.PeriodIndex([index[0], index[0], index[1]]))
        expected = pd.Series([0.005, 0.03], index=index[:2])
    else:
        dividends = pd.Series([1.0, pd.NA, 3.0], dtype="Float64", index=index)
        expected = pd.Series([0.01, 0.01, 0.04], index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
        actual = make_asset_list.ListMaker._get_ltm_dividend_yield(dividends, prices)
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.filterwarnings("ignore:Setting a value on a Series via .at")
def test_infinite_dividends_keep_scalar_window_sums(yield_list) -> None:
    yield_list.asset_obj_dict["A.US"].dividends.iloc[0] = np.inf
    with np.errstate(invalid="ignore"):
        result = yield_list.dividend_yield
        expected = scalar_dividend_yield(yield_list)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore:Setting a value on a Series via .at")
def test_float32_dividends_keep_scalar_accumulation(yield_list) -> None:
    index = yield_list.assets_ror.index
    yield_list._assets_dividends_ts = pd.DataFrame(
        {tick: pd.Series(np.float32(0.1), index=index) for tick in yield_list.symbols}
    )
    pd.testing.assert_frame_equal(yield_list.dividend_yield, scalar_dividend_yield(yield_list), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("attribute", ["dividend_paying_years", "dividend_growing_years"])
@pytest.mark.parametrize("payments", [0.0, 1.0])
def test_dividend_streaks_keep_periods_and_values_without_pandas4_warning(yield_list, attribute, payments):
    for asset in yield_list.asset_obj_dict.values():
        asset.dividends = pd.Series(payments, index=asset.ror.index, name=asset.symbol)
    with warnings.catch_warnings():
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("error", pd.errors.Pandas4Warning)
        result = getattr(yield_list, attribute)
    assert list(result.columns) == ["A.US", "B.US"]
    assert isinstance(result.index, pd.PeriodIndex)
    assert result.index.name is None
    assert [str(x) for x in result.index] == (["2021"] if attribute == "dividend_growing_years" else ["2020", "2021"])
    expected = [0] if attribute == "dividend_growing_years" else ([1, 2] if payments else [0, 0])
    assert result["A.US"].tolist() == expected
    assert result["B.US"].tolist() == expected


@pytest.mark.parametrize("attribute", ["dividend_paying_years", "dividend_growing_years"])
def test_dividend_streaks_with_no_columns_keep_empty_frame(yield_list, mocker, attribute):
    empty = pd.DataFrame(index=pd.period_range("2020", periods=2, freq="Y", name="Date"))
    mocker.patch.object(ok.AssetList, "dividends_annual", new_callable=PropertyMock, return_value=empty)
    with warnings.catch_warnings():
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("error", pd.errors.Pandas4Warning)
        actual = getattr(yield_list, attribute)
    pd.testing.assert_frame_equal(actual, pd.DataFrame())


def test_dividend_forecasts_are_excluded_on_supported_pandas_versions(mocker):
    current = pd.Period.now("M")
    index = pd.period_range(current - 1, current + 1, freq="M")
    asset = FakeAsset("A.US", pd.Series(0.0, index=index, name="A.US"))
    asset.dividends = pd.Series([1.0, 2.0, 3.0], index=index, name="A.US")
    asset.close_monthly = pd.Series(100.0, index=index, name="A.US")
    mocker.patch("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset)
    asset_list = ok.AssetList([asset], ccy="USD", inflation=False)
    actual = asset_list._get_single_asset_dividends("A.US", remove_forecast=True)
    assert actual.index.equals(index)
    assert actual.tolist() == [1.0, 2.0, 0.0]


def test_scalar_dividend_yield_keeps_month_windows_without_nanosecond_warning():
    index = pd.period_range("2020-02", periods=15, freq="M")
    dividends = pd.Series(np.ones(15, dtype=np.float32), index=index, name="A.US")
    prices = pd.Series(100.0, index=index, name="A.US")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Discarding nonzero nanoseconds", category=UserWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("error", pd.errors.Pandas4Warning)
        actual = make_asset_list.ListMaker._get_ltm_dividend_yield(dividends, prices)
    expected = pd.Series([min(i, 12) / 100.0 for i in range(1, 16)], index=index)
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=1e-12)
