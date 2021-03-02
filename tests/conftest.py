import pytest
import okama as ok
from pathlib import Path

data_folder = Path(__file__).parent / 'data'


@pytest.fixture(scope='class')
def _init_asset(request):
    request.cls.spy = ok.Asset(symbol='SPY.US')
    request.cls.otkr = ok.Asset(symbol='0165-70287767.PIF')


@pytest.fixture(scope='class')
def _init_asset_list(request) -> None:
    request.cls.asset_list = ok.AssetList(symbols=['RUB.FX', 'MCFTR.INDX'], ccy='RUB',
                                          first_date='2019-01', last_date='2020-01', inflation=True)
    request.cls.asset_list_lt = ok.AssetList(symbols=['RUB.FX', 'MCFTR.INDX'], ccy='RUB',
                                             first_date='2003-03', last_date='2020-01', inflation=True)
    request.cls.currencies = ok.AssetList(['RUBUSD.FX', 'EURUSD.FX', 'CNYUSD.FX'], ccy='USD',
                                          first_date='2019-01', last_date='2020-01', inflation=True)
    request.cls.spy = ok.AssetList(first_date='2000-01', last_date='2002-01', inflation=True)
    request.cls.real_estate = ok.AssetList(symbols=['RUS_SEC.RE', 'MOW_PR.RE'], ccy='RUB',
                                           first_date='2010-01', last_date='2015-01', inflation=True)


@pytest.fixture(scope='class')
def _init_portfolio(request):
    request.cls.portfolio = ok.Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], ccy='RUB',
                                         first_date='2015-01', last_date='2020-01', inflation=True)
    request.cls.portfolio_short_history = ok.Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], ccy='RUB',
                                                       first_date='2019-02', last_date='2020-01', inflation=True)
    request.cls.portfolio_no_inflation = ok.Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], ccy='RUB',
                                                      first_date='2015-01', last_date='2020-01', inflation=False)


@pytest.fixture(scope='class')
def _init_inflation(request):
    request.cls.infl_rub = ok.Inflation(symbol='RUB.INFL', last_date='2001-01')
    request.cls.infl_usd = ok.Inflation(symbol='USD.INFL', last_date='1923-01')
    request.cls.infl_eur = ok.Inflation(symbol='EUR.INFL', last_date='2006-02')


@pytest.fixture(scope='class')
def _init_rates(request):
    request.cls.rates_rub = ok.Rate(symbol='RUS_RUB.RATE', first_date='2015-01', last_date='2020-02')


@pytest.fixture(scope='module')
def init_plots():
    return ok.Plots(symbols=['RUB.FX', 'EUR.FX', 'MCFTR.INDX'], ccy='RUB', first_date='2010-01', last_date='2020-01')


@pytest.fixture(scope='module')
def init_efficient_frontier():
    ls = ['SPY.US', 'SBMX.MOEX']
    return ok.EfficientFrontier(symbols=ls, ccy='RUB', first_date='2018-11', last_date='2020-02', n_points=2)


@pytest.fixture(scope='module')
def init_efficient_frontier_bounds():
    ls = ['SPY.US', 'SBMX.MOEX']
    bounds = ((0, 0.5), (0, 1.))
    return ok.EfficientFrontier(symbols=ls, ccy='RUB', first_date='2018-11', last_date='2020-02', n_points=2, bounds=bounds)


@pytest.fixture(scope='module')
def init_efficient_frontier_reb():
    ls = ['SPY.US', 'GLD.US']
    return ok.EfficientFrontierReb(symbols=ls, ccy='RUB', first_date='2019-01', last_date='2020-02', n_points=3)
