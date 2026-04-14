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
