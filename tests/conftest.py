from copy import deepcopy

import pytest
import okama as ok
from pathlib import Path

import okama.portfolios.cashflow_strategies
import okama.portfolios.dcf

data_folder = Path(__file__).parent / "data"


# Asset
@pytest.fixture(scope="module")
def init_asset_spy():
    return ok.Asset(symbol="SPY.US")


@pytest.fixture(scope="module")
def init_asset_eurusd():
    return ok.Asset(symbol="EURUSD.FX")


@pytest.fixture(scope="module")
def init_asset_berkshire():
    return ok.Asset(symbol="BRK.A.US")


@pytest.fixture(scope="module")
def init_asset_pif():
    return ok.Asset(symbol="0165-70287767.PIF")


@pytest.fixture(scope="module")
def init_asset_usdrub():
    return ok.Asset(symbol="RUB.FX")


# Asset List
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


# Macro
@pytest.fixture(scope="function")
def _init_inflation(request):
    request.cls.infl_rub = ok.Inflation(symbol="RUB.INFL", last_date="2001-01")
    request.cls.infl_usd = ok.Inflation(symbol="USD.INFL", last_date="1923-01")
    request.cls.infl_eur = ok.Inflation(symbol="EUR.INFL", last_date="2006-02")
    request.cls.infl_usd_less_year = ok.Inflation(symbol="USD.INFL", first_date="2006-01", last_date="2006-11")


@pytest.fixture(scope="class")
def _init_rates(request):
    request.cls.rates_rub = ok.Rate(symbol="RUS_RUB.RATE", first_date="2015-01", last_date="2020-02")
    request.cls.rates_cbr_rate = ok.Rate(symbol="RUS_CBR.RATE", first_date="2015-01", last_date="2020-02")
    request.cls.rates_ruonia = ok.Rate(symbol="RUONIA.RATE", first_date="2015-01", last_date="2020-02")


@pytest.fixture(scope="class")
def _init_indicator(request):
    request.cls.cape10_usd = ok.Indicator(symbol="USA_CAPE10.RATIO", first_date="2021-01", last_date="2022-02")


# Efficient Frontier Single Period
@pytest.fixture(scope="module")
def init_efficient_frontier_values1():
    return dict(
        assets=["MCFTR.INDX", "RGBITR.INDX"],
        ccy="RUB",
        first_date="2018-11",
        last_date="2020-02",
        inflation=True,
        n_points=2,
    )


@pytest.fixture(scope="module")
def init_efficient_frontier_values2():
    return dict(
        assets=["SPY.US", "AGG.US", "GLD.US"],
        ccy="USD",
        first_date="2010-01",
        last_date="2020-01",
        inflation=True,
        n_points=20,
        full_frontier=True,
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
    return ok.EfficientFrontierReb(
        assets=["SPY.US", "GLD.US"],
        ccy="USD",
        first_date="2019-01",
        last_date="2020-02",
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0, 1), (0, 1)),
        inflation=True,
        n_points=2,
        full_frontier=False,
    )


@pytest.fixture(scope="module")
def bounds_frontier_params():
    return dict(
        assets=["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"],
        ccy="USD",
        first_date="2019-01",
        last_date="2020-02",
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0, 1), (0, 1)),
        inflation=True,
    )


@pytest.fixture(scope="function")
def without_bounds_params():
    return dict(
        assets=["VOO.US", "GLD.US", "SCHA.US"],
        ccy="USD",
        first_date="2004-10",
        last_date="2020-10",
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


@pytest.fixture(scope="function")
def with_bounds_params():
    return dict(
        assets=["VOO.US", "GLD.US", "SCHA.US"],
        ccy="USD",
        first_date="2004-10",
        last_date="2020-10",
        rebalancing_strategy=ok.Rebalance(period="year"),
        bounds=((0, 0.4), (0, 1), (0, 1)),
    )


@pytest.fixture(scope="module")
def _min_ratio_asset_when_none_params():
    return dict(
        assets=["SPY.US", "GLD.US"],
        ccy="USD",
        last_date="2020-10",
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


@pytest.fixture(scope="module")
def _min_ratio_asset_when_not_none_params():
    return dict(
        assets=["SPY.US", "MCFTR.INDX"],
        ccy="RUB",
        last_date="2025-03",
        rebalancing_strategy=ok.Rebalance(period="year"),
    )


@pytest.fixture(scope="module")
def convex_frontier_params():
    return dict(
        assets=["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-11",
        rebalancing_strategy=ok.Rebalance(period="year"),
        n_points=5,
        verbose=True,
    )


@pytest.fixture(scope="module")
def nonconvex_frontier_params():
    return dict(
        assets=["SPY.US", "GLD.US", "VB.US", "RGBITR.INDX", "MCFTR.INDX"],
        ccy="RUB",
        first_date="2004-12",
        last_date="2020-12",
        rebalancing_strategy=ok.Rebalance(period="year"),
        n_points=5,
        verbose=True,
    )


@pytest.fixture(scope="module")
def init_bounds_frontier(bounds_frontier_params):
    return ok.EfficientFrontierReb(**bounds_frontier_params)


@pytest.fixture(scope="function")
def init_frontier_without_bounds(without_bounds_params):
    return ok.EfficientFrontierReb(**without_bounds_params)


@pytest.fixture(scope="function")
def init_frontier_with_bounds(with_bounds_params):
    return ok.EfficientFrontierReb(**with_bounds_params)


@pytest.fixture(scope="module")
def init_frontier_with_none(_min_ratio_asset_when_none_params):
    return ok.EfficientFrontierReb(**_min_ratio_asset_when_none_params)


@pytest.fixture(scope="module")
def init_frontier_with_not_none(_min_ratio_asset_when_not_none_params):
    return ok.EfficientFrontierReb(**_min_ratio_asset_when_not_none_params)


@pytest.fixture(scope="module")
def init_convex_frontier(convex_frontier_params):
    return ok.EfficientFrontierReb(**convex_frontier_params)


@pytest.fixture(scope="module")
def init_nonconvex_frontier(nonconvex_frontier_params):
    return ok.EfficientFrontierReb(**nonconvex_frontier_params)


# Rebalance
@pytest.fixture(scope="module")
def init_rebalance_no_rebalancing():
    return ok.Rebalance(
        period="none",
        abs_deviation=None,
        rel_deviation=None,
    )
