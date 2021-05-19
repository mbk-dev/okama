from pytest import mark
import pandas as pd


@mark.asset
@mark.smoke
def test_get_symbol_data(init_asset_spy):
    assert init_asset_spy.name == "SPDR S&P 500 ETF Trust"
    assert init_asset_spy.country == "USA"
    assert init_asset_spy.currency == "USD"
    assert init_asset_spy.type == "ETF"
    assert init_asset_spy.inflation == "USD.INFL"
    assert init_asset_spy.first_date == pd.to_datetime("1993-02")


def test_price(init_asset_spy):
    assert type(init_asset_spy.price) == float


def test_dividends(init_asset_spy):
    assert init_asset_spy.dividends["2019"].iloc[-1] == 1.57


def test_nav_ts(init_asset_pif):
    assert init_asset_pif.nav_ts[0] == 101820352.18
