from copy import deepcopy
from pathlib import Path

import pytest
import okama as ok

data_folder = Path(__file__).parent / "data"

# Portfolios in AssetList
@pytest.fixture(scope="package")
def init_portfolio_values():
    return dict(
        assets=["RGBITR.INDX", "MCFTR.INDX"],  # index values are better as they are not changing (adjusted_close)
        ccy="RUB",
        first_date="2015-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf1.PF",
    )

@pytest.fixture(scope="package")
def portfolio_short_history(init_portfolio_values):
    _portfolio_short_history = deepcopy(init_portfolio_values)
    _portfolio_short_history["first_date"] = "2019-02"
    return ok.Portfolio(**_portfolio_short_history)

@pytest.fixture(scope="package")
def portfolio_dividends(init_portfolio_values):
    _portfolio_dividends = deepcopy(init_portfolio_values)
    _portfolio_dividends["assets"] = ["SBER.MOEX", "T.US", "GNS.LSE"]
    return ok.Portfolio(**_portfolio_dividends)


@pytest.fixture(scope="class")
def assets_from_db():
    return ["USDRUB.CBR", "MCFTR.INDX"]


@pytest.fixture(scope="class")
def _init_asset_list(request, portfolio_short_history, portfolio_dividends, assets_from_db) -> None:
    request.cls.asset_list_with_portfolio = ok.AssetList(
        assets=[portfolio_short_history] + assets_from_db,
        ccy="USD",
    )

    request.cls.asset_list_with_portfolio_dividends = ok.AssetList(
        assets=[portfolio_dividends] + assets_from_db,
        ccy="USD",
    )

    request.cls.asset_list = ok.AssetList(
        assets=assets_from_db,
        ccy="RUB",
        first_date="2019-01",
        last_date="2020-01",
        inflation=True,
    )
    request.cls.asset_list_lt = ok.AssetList(
        assets=assets_from_db,
        ccy="RUB",
        first_date="2003-03",
        last_date="2020-01",
        inflation=True,
    )
    request.cls.asset_list_st = ok.AssetList(
        assets=assets_from_db,
        ccy="RUB",
        first_date="2019-01",
        last_date="2019-05",
        inflation=False,
    )
    request.cls.asset_list_no_infl = ok.AssetList(
        assets=assets_from_db,
        ccy="RUB",
        first_date="2019-01",
        last_date="2020-01",
        inflation=False,
    )
    request.cls.currencies = ok.AssetList(
        ["RUBUSD.FX", "EURUSD.FX", "CNYUSD.FX"],
        ccy="USD",
        first_date="2019-01",
        last_date="2020-01",
        inflation=True,
    )
    request.cls.spy = ok.AssetList(first_date="2000-01", last_date="2002-01", inflation=True)
    request.cls.spy_rub = ok.AssetList(first_date="2000-01", last_date="2002-01", inflation=True, ccy="RUB")
    request.cls.real_estate = ok.AssetList(
        assets=["RUS_SEC.RE", "MOW_PR.RE"],
        ccy="RUB",
        first_date="2010-01",
        last_date="2015-01",
        inflation=True,
    )


