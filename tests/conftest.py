from copy import deepcopy

import pytest
import okama as ok
from pathlib import Path

data_folder = Path(__file__).parent / "data"


# Asset
@pytest.fixture(scope="module")
def init_asset_spy():
    return ok.Asset(symbol="SPY.US")


@pytest.fixture(scope="module")
def init_asset_pif():
    return ok.Asset(symbol="0165-70287767.PIF")


# Asset List
@pytest.fixture(scope="class")
def assets_from_db():
    return ["RUB.FX", "MCFTR.INDX"]


@pytest.fixture(scope="class")
def _init_asset_list(
    request, portfolio_short_history, portfolio_dividends, assets_from_db
) -> None:
    request.cls.asset_list_with_portfolio = ok.AssetList(
        assets=[portfolio_short_history] + assets_from_db, ccy="USD",
    )

    request.cls.asset_list_with_portfolio_dividends = ok.AssetList(
        assets=[portfolio_dividends] + assets_from_db, ccy="USD",
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
    request.cls.spy = ok.AssetList(
        first_date="2000-01", last_date="2002-01", inflation=True
    )
    request.cls.spy_rub = ok.AssetList(
        first_date="2000-01", last_date="2002-01", inflation=True, ccy="RUB"
    )
    request.cls.real_estate = ok.AssetList(
        assets=["RUS_SEC.RE", "MOW_PR.RE"],
        ccy="RUB",
        first_date="2010-01",
        last_date="2015-01",
        inflation=True,
    )

# Portfolio
@pytest.fixture(scope="package")
def init_portfolio_values():
    return dict(
        assets=["RUB.FX", "MCFTR.INDX"],
        ccy="RUB",
        first_date="2015-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_period="year",
        symbol="pf1.PF",
    )


@pytest.fixture(scope="package")
def portfolio_rebalanced_year(init_portfolio_values):
    return ok.Portfolio(**init_portfolio_values)


@pytest.fixture(scope="package")
def portfolio_not_rebalanced(init_portfolio_values):
    _portfolio_not_rebalanced = deepcopy(init_portfolio_values)
    _portfolio_not_rebalanced["rebalancing_period"] = "none"
    return ok.Portfolio(**_portfolio_not_rebalanced)


@pytest.fixture(scope="package")
def portfolio_rebalanced_month(init_portfolio_values):
    _portfolio_rebalanced_month = deepcopy(init_portfolio_values)
    _portfolio_rebalanced_month["rebalancing_period"] = "month"
    return ok.Portfolio(**_portfolio_rebalanced_month)


@pytest.fixture(scope="package")
def portfolio_no_inflation(init_portfolio_values):
    _portfolio_no_inflation = deepcopy(init_portfolio_values)
    _portfolio_no_inflation["inflation"] = False
    _portfolio_no_inflation["rebalancing_period"] = "month"
    return ok.Portfolio(**_portfolio_no_inflation)


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


# Macro
@pytest.fixture(scope="class")
def _init_inflation(request):
    request.cls.infl_rub = ok.Inflation(symbol="RUB.INFL", last_date="2001-01")
    request.cls.infl_usd = ok.Inflation(symbol="USD.INFL", last_date="1923-01")
    request.cls.infl_eur = ok.Inflation(symbol="EUR.INFL", last_date="2006-02")


@pytest.fixture(scope="class")
def _init_rates(request):
    request.cls.rates_rub = ok.Rate(
        symbol="RUS_RUB.RATE", first_date="2015-01", last_date="2020-02"
    )


# Efficient Frontier Single Period
@pytest.fixture(scope="module")
def init_efficient_frontier_values1():
    return dict(
        assets=["SPY.US", "SBMX.MOEX"],
        ccy="RUB",
        first_date="2018-11",
        last_date="2020-02",
        inflation=True,
        n_points=2,
    )


@pytest.fixture(scope="module")
def init_efficient_frontier_values2():
    return dict(
        assets=['RUB.FX', 'EUR.FX', 'MCFTR.INDX'],
        ccy="RUB",
        first_date='2010-01',
        last_date='2020-01',
        inflation=True,
        n_points=20,
        full_frontier=True
    )


@pytest.fixture(scope="module")
def init_efficient_frontier(init_efficient_frontier_values1):
    return ok.EfficientFrontier(**init_efficient_frontier_values1)


@pytest.fixture(scope="module")
def init_efficient_frontier_bounds(init_efficient_frontier_values1):
    bounds = ((0.0, 0.5), (0.0, 1.0))
    return ok.EfficientFrontier(**init_efficient_frontier_values1, bounds=bounds)


@pytest.fixture(scope="module")
def init_efficient_frontier_three_assets(init_efficient_frontier_values2):
    return ok.EfficientFrontier(**init_efficient_frontier_values2)


# Efficient Frontier Multi-Period
@pytest.fixture(scope="module")
def init_efficient_frontier_reb():
    ls = ["SPY.US", "GLD.US"]
    return ok.EfficientFrontierReb(
        assets=ls,
        ccy="RUB",
        first_date="2019-01",
        last_date="2020-02",
        n_points=3,
        verbose=False,
        full_frontier=True,
    )
