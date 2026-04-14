from unittest.mock import call  # noqa: I001

import pandas as pd
import pytest
import numpy as np

import okama as ok


class _MacroDefaults:
    def __init__(self):
        # Monthly PeriodIndex with 12 months
        self.m_idx = pd.period_range("2020-01", periods=12, freq="M")
        # Minimal daily PeriodIndex with 3 days
        self.d_idx = pd.period_range("2020-01-01", periods=3, freq="D")

        # Default time series per type
        # Inflation: 12 months, reproducible random values with seed
        rng = np.random.default_rng(seed=42)
        infl_values = rng.uniform(low=-0.005, high=0.015, size=12)
        self.infl_m = pd.Series(infl_values, index=self.m_idx, name="USD.INFL")

        # Monthly rate series (deterministic 12-month sequence)
        self.rate_m = pd.Series(0.05 + 0.001 * np.arange(12), index=self.m_idx, name="RUS_RUB.RATE")
        # Daily rate series keeps 3 deterministic days
        self.rate_d = pd.Series([0.051, 0.052, 0.053], index=self.d_idx, name="RUS_RUB.RATE")
        # Monthly indicator series (deterministic 12-month sequence)
        self.ind_m = pd.Series(30.0 + np.linspace(0.0, 1.1, 12), index=self.m_idx, name="USA_CAPE10.RATIO")

        # Default symbol info map
        self.symbol_info = {
            "USD.INFL": {
                "code": "USD",
                "name": "US Inflation Rate",
                "country": "USA",
                "currency": "USD",
                "type": "inflation",
            },
            "RUS_RUB.RATE": {
                "code": "RUS_RUB",
                "name": "Max deposit rates (RUB) in Russian banks",
                "country": "Russia",
                "currency": "RUB",
                "type": "rate",
            },
            "USA_CAPE10.RATIO": {
                "code": "USA_CAPE10",
                "name": "Cyclically adjusted price-to-earnings ratio CAPE10 for USA",
                "country": "USA",
                "currency": "USD",
                "type": "ratio",
            },
        }


@pytest.fixture()
def macro_patches(mocker):
    # Namespaces for Indicator._check_namespace
    m_ns = mocker.patch(
        "okama.api.namespaces.get_macro_namespaces",
        return_value={"INFL", "RATE", "RATIO"},
    )

    md = _MacroDefaults()

    def _get_symbol_info(sym):
        return md.symbol_info[sym]

    def _get_macro_ts(symbol, first_date, last_date, period):
        if symbol == "USD.INFL":
            return md.infl_m
        if symbol == "RUS_RUB.RATE":
            return md.rate_d if period == "D" else md.rate_m
        if symbol == "USA_CAPE10.RATIO":
            return md.ind_m
        raise KeyError(symbol)

    m_info = mocker.patch("okama.api.data_queries.QueryData.get_symbol_info", side_effect=_get_symbol_info)
    m_ts = mocker.patch("okama.api.data_queries.QueryData.get_macro_ts", side_effect=_get_macro_ts)

    yield {
        "m_namespaces": m_ns,
        "m_get_symbol_info": m_info,
        "m_get_macro_ts": m_ts,
        "defaults": md,
    }


def test_inflation_init_and_values_monthly(macro_patches):
    infl = ok.Inflation("USD.INFL")
    md = macro_patches["defaults"]

    # Fields from symbol info
    assert infl.ticker == "USD"
    assert infl.name == md.symbol_info["USD.INFL"]["name"]
    assert infl.country == "USA"
    assert infl.currency == "USD"
    assert infl.type == "inflation"

    # Dates computed from monthly index
    assert infl.first_date == md.m_idx[0].to_timestamp(how="start")
    assert infl.last_date == md.m_idx[-1].to_timestamp(how="start")

    # values_monthly comes from mock
    assert infl.values_monthly.equals(md.infl_m)


def test_set_values_monthly_happy_path(macro_patches):
    infl = ok.Inflation("USD.INFL")
    md = macro_patches["defaults"]
    next_month = md.m_idx[-1] + 1
    next_month_str = f"{next_month.start_time:%Y-%m}"
    infl.set_values_monthly(next_month_str, 0.03)
    assert infl.values_monthly[pd.Period(next_month_str, freq="M")] == 0.03
    # last_date should update to the newly added month
    assert infl.last_date == pd.Period(next_month_str, freq="M").to_timestamp(how="start")


def test_rate_values_daily_calls_period_D(macro_patches):
    rate = ok.Rate("RUS_RUB.RATE")
    _ = rate.values_daily
    m_ts = macro_patches["m_get_macro_ts"]
    assert call("RUS_RUB.RATE", None, None, period="D") in m_ts.mock_calls


def test_indicator_and_namespace_checking(macro_patches):
    # Indicator allowed when in allowed macro namespaces
    indicator = ok.Indicator("USA_CAPE10.RATIO")
    md = macro_patches["defaults"]
    assert indicator.values_monthly.equals(md.ind_m)

    # Invalid namespace for Indicator (uses get_macro_namespaces)
    with pytest.raises(ValueError):
        ok.Indicator("USD.INFL")


def test_macro_init_failing_namespaces(macro_patches):
    # Inflation should reject RATE
    with pytest.raises(ValueError, match=r"RATE is not in allowed namespaces"):
        ok.Inflation("RUS_RUB.RATE")
    # Rate should reject INFL
    with pytest.raises(ValueError, match=r"INFL is not in allowed namespaces"):
        ok.Rate("USD.INFL")


def test_inflation_describe_smoke(macro_patches):
    infl = ok.Inflation("USD.INFL")
    df = infl.describe(years=(1,))
    assert {"property", "period"}.issubset(df.columns)
    assert infl.symbol in df.columns
