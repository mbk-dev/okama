from unittest.mock import call  # noqa: I001
import pytest

import pandas as pd

import okama as ok


class _DefaultMocks:
    def __init__(self, *, exchange: str = "NYSE"):
        self.allowed_namespaces = {"US", "FX", "INDX", "PIF"}
        self.symbol_info = {
            "code": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "country": "USA",
            "exchange": exchange,
            "currency": "USD",
            "type": "ETF",
            "isin": "US78462F1030",
        }
        # Minimal monthly ror series (PeriodIndex with monthly freq)
        self.ror_index = pd.period_range("2020-01", "2020-03", freq="M")
        self.ror = pd.Series([0.01, -0.02, 0.03], index=self.ror_index, name="SPY.US")


@pytest.fixture
def basic_patches(mocker):
    m_ns = mocker.patch("okama.asset.namespaces.get_assets_namespaces", return_value={"US", "FX", "INDX", "PIF"})
    m_info = mocker.patch("okama.asset.data_queries.QueryData.get_symbol_info")
    m_ror = mocker.patch("okama.asset.data_queries.QueryData.get_ror")
    dm = _DefaultMocks()
    m_info.return_value = dm.symbol_info
    m_ror.return_value = dm.ror
    yield {
        "m_namespaces": m_ns,
        "m_get_symbol_info": m_info,
        "m_get_ror": m_ror,
        "defaults": dm,
    }


def test_init_uses_mocked_queries(basic_patches):
    a = ok.Asset("SPY.US")
    dm = basic_patches["defaults"]
    # Asserts on fields filled from get_symbol_info
    assert a.ticker == dm.symbol_info["code"]
    assert a.name == dm.symbol_info["name"]
    assert a.country == dm.symbol_info["country"]
    assert a.exchange == dm.symbol_info["exchange"]
    assert a.currency == dm.symbol_info["currency"]
    assert a.type == dm.symbol_info["type"]
    assert a.isin == dm.symbol_info["isin"]
    assert a.inflation == f"{dm.symbol_info['currency']}.INFL"

    # Asserts on dates computed from ror index
    assert a.first_date == dm.ror_index[0].to_timestamp(how="start")
    assert a.last_date == dm.ror_index[-1].to_timestamp(how="start")


def test_price_calls_live_price(basic_patches, mocker):
    m_price = mocker.patch("okama.asset.data_queries.QueryData.get_live_price", return_value=123.45)
    a = ok.Asset("SPY.US")
    assert a.price == 123.45
    m_price.assert_called_once_with("SPY.US")


def test_close_calls_with_expected_periods(basic_patches, mocker):
    m_close = mocker.patch("okama.asset.data_queries.QueryData.get_close", return_value=pd.Series([1, 2, 3]))
    a = ok.Asset("SPY.US")
    _ = a.close_daily
    _ = a.close_monthly
    assert m_close.mock_calls == [
        call("SPY.US", period="D"),
        call("SPY.US", period="M"),
    ]


def test_adj_close_calls_with_expected_period(basic_patches, mocker):
    m_adj = mocker.patch("okama.asset.data_queries.QueryData.get_adj_close", return_value=pd.Series([10, 20]))
    a = ok.Asset("SPY.US")
    _ = a.adj_close
    m_adj.assert_called_once_with("SPY.US", period="D")


def test_dividends_empty_returns_zero_monthly_series(basic_patches, mocker):
    # Make dividends empty -> class should return zero monthly series between first/last dates
    mocker.patch("okama.asset.data_queries.QueryData.get_dividends", return_value=pd.Series(dtype=float))
    a = ok.Asset("SPY.US")
    div = a.dividends
    # For ror 2020-01 .. 2020-03, zero series should be for 2020-02 only (inclusive="neither")
    assert isinstance(div, pd.Series)
    assert len(div) == 1
    assert div.index[0].strftime("%Y-%m") == "2020-02"
    assert float(div.iloc[0]) == 0.0
    assert div.name == "SPY.US"


def test_dividends_aggregates_to_monthly(basic_patches, mocker):
    # Provide non-empty daily PeriodIndex dividends and check monthly aggregation
    daily_idx = pd.period_range("2020-02-01", periods=3, freq="D")
    daily_div = pd.Series([0.5, 0.25, 0.25], index=daily_idx, name="SPY.US")
    mocker.patch("okama.asset.data_queries.QueryData.get_dividends", return_value=daily_div)
    a = ok.Asset("SPY.US")
    div_m = a.dividends
    assert len(div_m) >= 1
    # All three days are in Feb 2020 -> sum to 1.0 in that month
    feb = div_m.loc["2020-02"]
    assert pytest.approx(float(feb)) == 1.0


def test_invalid_namespace_raises_value_error(mocker):
    mocker.patch("okama.asset.namespaces.get_assets_namespaces", return_value={"US"})
    with pytest.raises(ValueError):
        # Symbol with namespace not in allowed set -> error before any data query call
        ok.Asset("XYZ.EU")


def test_local_name_present(basic_patches):
    dm = basic_patches["defaults"]
    dm.symbol_info["local_name"] = "Сбербанк"
    basic_patches["m_get_symbol_info"].return_value = dm.symbol_info
    a = ok.Asset("SPY.US")
    assert a.local_name == "Сбербанк"
    assert a.info == dm.symbol_info  # raw payload stored
    assert "local_name" in repr(a)


def test_local_name_absent_is_none(basic_patches):
    # default symbol_info has no "local_name" key
    a = ok.Asset("SPY.US")
    assert a.local_name is None
    assert a.info["name"] == "SPDR S&P 500 ETF Trust"


@pytest.fixture
def cagr_asset_env(basic_patches):
    index = pd.period_range("2020-01", periods=36, freq="M")
    returns = pd.Series([0.01] * 24 + [0.02] * 12, index=index, name="SPY.US")
    basic_patches["m_get_symbol_info"].side_effect = lambda symbol: {
        **basic_patches["defaults"].symbol_info,
        "code": symbol.split(".")[0],
    }
    basic_patches["m_get_ror"].side_effect = lambda symbol, **kwargs: returns.rename(symbol).loc[
        kwargs["first_date"] : kwargs["last_date"]
    ]
    return returns


@pytest.mark.parametrize("period", [None, 1, 2, 3])
def test_asset_cagr_matches_single_asset_list(cagr_asset_env, period):
    a = ok.Asset("SPY.US", last_date="2022-12")
    al = ok.AssetList([a], ccy="USD", last_date="2022-12", inflation=False)
    result = a.get_cagr(period=period)
    assert isinstance(result, float)
    assert result == pytest.approx(al.get_cagr(period=period)["SPY.US"].iloc[-1])
    if period == 1:
        assert result == pytest.approx(1.02**12 - 1)


@pytest.mark.parametrize("period, error", [(0, ValueError), (-1, ValueError), (4, ValueError), (1.5, TypeError)])
def test_asset_cagr_rejects_invalid_period(cagr_asset_env, period, error):
    with pytest.raises(error):
        ok.Asset("SPY.US").get_cagr(period=period)


def test_asset_cagr_short_history_is_nan(basic_patches):
    assert pd.isna(ok.Asset("SPY.US").get_cagr())


@pytest.mark.parametrize("period", [None, 1])
def test_asset_real_cagr_uses_common_inflation_history(cagr_asset_env, mocker, period):
    inflation = pd.Series(0.005, index=cagr_asset_env.index[3:-2], name="USD.INFL")
    mocker.patch("okama.macro.namespaces.get_macro_namespaces", return_value={"INFL"})
    mocker.patch("okama.macro.data_queries.QueryData.get_macro_ts", return_value=inflation)
    a = ok.Asset("SPY.US")
    al = ok.AssetList([a], ccy="USD", inflation=True)
    assert a.get_cagr(period=period, real=True) == pytest.approx(
        al.get_cagr(period=period, real=True)["SPY.US"].iloc[-1]
    )
    assert a.last_date == pd.Timestamp("2022-12-01")


@pytest.mark.parametrize("month", ["2019-12", "2020-04"])
def test_monthly_close_error_names_requested_month_and_available_range(basic_patches, mocker, month):
    close = pd.Series([10.0, 20.0, 30.0], index=pd.period_range("2020-01", periods=3, freq="M"))
    mocker.patch("okama.asset.data_queries.QueryData.get_close", return_value=close)
    a = ok.Asset("SPY.US")
    with pytest.raises(KeyError) as error:
        a.get_close_monthly(month)
    assert month in str(error.value)
    assert "2020-01" in str(error.value)
    assert "2020-03" in str(error.value)


@pytest.mark.parametrize("month", ["2020-02", pd.Timestamp("2020-02-15"), pd.Period("2020-02", freq="M")])
def test_monthly_close_accessor_returns_scalar_and_preserves_series(basic_patches, mocker, month):
    close = pd.Series([10.0, 20.0, 30.0], index=pd.period_range("2020-01", periods=3, freq="M"))
    mocker.patch("okama.asset.data_queries.QueryData.get_close", return_value=close)
    a = ok.Asset("SPY.US")
    assert a.get_close_monthly(month) == 20.0
    pd.testing.assert_series_equal(a.close_monthly, close)
