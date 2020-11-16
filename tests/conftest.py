import pytest
from okama.assets import Asset, AssetList, Portfolio
from okama.macro import Inflation, Rate
from okama.frontier import EfficientFrontier
from okama import Plots, EfficientFrontierReb


@pytest.fixture(scope='class')
def _init_asset(request):
    request.cls.spy = Asset(symbol='SPY.US')
    request.cls.otkr = Asset(symbol='0165-70287767.PIF')


@pytest.fixture(scope='class')
def _init_asset_list(request) -> None:
    request.cls.asset_list = AssetList(symbols=['RUB.FX', 'MCFTR.INDX'], curr='RUB',
                                       first_date='2019-01', last_date='2020-01', inflation=True)
    request.cls.currencies = AssetList(['RUBUSD.FX', 'EURUSD.FX', 'CNYUSD.FX'], curr='USD',
                                       first_date='2019-01', last_date='2020-01', inflation=True)
    request.cls.spy = AssetList(first_date='2000-01', last_date='2002-01', inflation=True)
    request.cls.real_estate = AssetList(symbols=['RUS_SEC.RE', 'MOW_PR.RE'], curr='RUB',
                                        first_date='2010-01', last_date='2015-01', inflation=True)


@pytest.fixture(scope='class')
def _init_portfolio(request):
    request.cls.portfolio = Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], curr='RUB',
                                      first_date='2015-01', last_date='2020-01', inflation=True)
    request.cls.portfolio_short_history = Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], curr='RUB',
                                                    first_date='2019-02', last_date='2020-01', inflation=True)
    request.cls.portfolio_no_inflation = Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], curr='RUB',
                                                   first_date='2015-01', last_date='2020-01', inflation=False)


@pytest.fixture(scope='class')
def _init_inflation(request):
    request.cls.infl_rub = Inflation(symbol='RUB.INFL', last_date='2001-01')
    request.cls.infl_usd = Inflation(symbol='USD.INFL', last_date='1923-01')
    request.cls.infl_eur = Inflation(symbol='EUR.INFL', last_date='2006-02')


@pytest.fixture(scope='class')
def _init_rates(request):
    request.cls.rates_rub = Rate(symbol='RUS_RUB.RATE', first_date='2015-01', last_date='2020-02')


@pytest.fixture(scope='module')
def init_plots():
    return Plots(symbols=['RUB.FX', 'EUR.FX', 'MCFTR.INDX'], curr='RUB', first_date='2010-01', last_date='2020-01')


@pytest.fixture(scope='module')
def init_efficient_frontier():
    ls = ['SPY.US', 'SBMX.MOEX']
    return EfficientFrontier(symbols=ls, curr='RUB', first_date='2018-11', last_date='2020-02', n_points=2)


@pytest.fixture(scope='module')
def init_efficient_frontier_bounds():
    ls = ['SPY.US', 'SBMX.MOEX']
    bounds = ((0, 0.5), (0, 1.))
    return EfficientFrontier(symbols=ls, curr='RUB', first_date='2018-11', last_date='2020-02', n_points=2, bounds=bounds)


@pytest.fixture(scope='class')
def _init_efficient_frontier_reb(request):
    ls = ['SPY.US', 'GLD.US']
    request.cls.ef = EfficientFrontierReb(symbols=ls, curr='RUB', first_date='2019-01', last_date='2020-02', n_points=2)
