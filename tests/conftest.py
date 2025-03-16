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


# Portfolio
@pytest.fixture(scope="package")
def init_portfolio_values():
    return dict(
        assets=["RGBITR.INDX", "MCFTR.INDX"],  # index values are better as they are not changing (adjusted_close)
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


# DCF Scenarios
@pytest.fixture(scope="package")
def init_portfolio_dcf(init_portfolio_values):
    _portfolio_values = deepcopy(init_portfolio_values)
    _portfolio_dcf_values = dict(
        discount_rate=None,
        use_discounted_values=True,
    )
    return [_portfolio_values, _portfolio_dcf_values]


@pytest.fixture(scope="package")
def init_mc():
    return dict(distribution="t", period=10, number=100)


@pytest.fixture(scope="package")
def portfolio_dcf(init_portfolio_dcf):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = ok.PortfolioDCF(pf, **init_portfolio_dcf[1])
    return pf_dcf


@pytest.fixture(scope="package")
def portfolio_dcf_no_inflation(init_portfolio_dcf):
    values_list = deepcopy(init_portfolio_dcf)
    values_list[0]["inflation"] = False
    # Create Portfolio
    pf = ok.Portfolio(**values_list[0])
    pf_dcf = ok.PortfolioDCF(pf, **values_list[1])
    return pf_dcf


@pytest.fixture(scope="package")
def portfolio_dcf_discount_rate(init_portfolio_dcf):
    values_list = deepcopy(init_portfolio_dcf)
    values_list[1]["discount_rate"] = 0.08
    # Create Portfolio
    pf = ok.Portfolio(**values_list[0])
    pf_dcf = ok.PortfolioDCF(pf, **values_list[1])
    return pf_dcf


@pytest.fixture(scope="function")
def portfolio_dcf_indexation(init_portfolio_dcf, init_mc):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = ok.PortfolioDCF(pf, **init_portfolio_dcf[1])
    pf_dcf.set_mc_parameters(**init_mc)
    # Cash Flow
    ind = ok.IndexationStrategy(pf)  # create IndexationStrategy linked to the portfolio
    ind.initial_investment = 10_000  # add initial investments size
    ind.frequency = "year"  # set cash flow frequency
    ind.amount = -1_500  # set withdrawal size
    ind.indexation = "inflation"
    pf_dcf.cashflow_parameters = ind
    pf_dcf.use_discounted_values = False
    return pf_dcf


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
        rebalancing_period="year",    
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
        rebalancing_period="year",    
        bounds=((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0, 1), (0, 1)),
        inflation=True,
    )


@pytest.fixture(scope="module")
def without_bounds_params():
    return dict(
        assets=["GLD.US", "PGJ.US", "GC.COMM", "VB.US"],
        ccy="RUB",
        first_date="2004-12",
        last_date="2020-12",
        rebalancing_period="year",    
        inflation=True,
    )


@pytest.fixture(scope="module")
def with_bounds_params():
    return dict(
        assets=["GLD.US", "PGJ.US", "GC.COMM", "VB.US"],
        ccy="RUB",
        first_date="2004-12",
        last_date="2020-12",
        rebalancing_period="year",    
        bounds=((0, 1), (0, 1), (0, 1), (0, 0.4)),
        inflation=True,
    )


@pytest.fixture(scope="module")
def convex_frontier_params():
    return dict(
        assets=["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-11",
        rebalancing_period="year",
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
        rebalancing_period="year",
        n_points=5,
        verbose=True,
    )


@pytest.fixture(scope="module")
def init_bounds_frontier(bounds_frontier_params):
    return ok.EfficientFrontierReb(**bounds_frontier_params)


@pytest.fixture(scope="module")
def init_frontier_without_bounds(without_bounds_params):
    return ok.EfficientFrontierReb(**without_bounds_params)


@pytest.fixture(scope="module")
def init_frontier_with_bounds(with_bounds_params):
    return ok.EfficientFrontierReb(**with_bounds_params)


@pytest.fixture(scope="module")
def init_convex_frontier(convex_frontier_params):
    return ok.EfficientFrontierReb(**convex_frontier_params)


@pytest.fixture(scope="module")
def init_nonconvex_frontier(nonconvex_frontier_params):
    return ok.EfficientFrontierReb(**nonconvex_frontier_params)
