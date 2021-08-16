from pytest import mark
from pytest import approx
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


def test_close_daily(init_asset_spy):
    assert init_asset_spy.close_daily.loc['2000-01-20'] == 144.75


def test_close_monthly(init_asset_spy):
    assert init_asset_spy.close_monthly.loc['2000-01'] == 139.5625


def test_adj_close(init_asset_spy):
    assert init_asset_spy.adj_close.loc['2000-01-20'] == approx(97.0629, rel=1e-2)


def test_price(init_asset_spy):
    assert type(init_asset_spy.price) == float


def test_dividends(init_asset_spy):
    assert init_asset_spy.dividends["2019"].iloc[-1] == 1.57


def test_nav_ts(init_asset_pif):
    assert init_asset_pif.nav_ts[0] == 101820352.18
