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
    request.cls.asset_list_no_infl = ok.AssetList(symbols=['RUB.FX', 'MCFTR.INDX'], ccy='RUB',
                                                  first_date='2019-01', last_date='2020-01', inflation=False)
    request.cls.currencies = ok.AssetList(['RUBUSD.FX', 'EURUSD.FX', 'CNYUSD.FX'], ccy='USD',
                                          first_date='2019-01', last_date='2020-01', inflation=True)
    request.cls.spy = ok.AssetList(first_date='2000-01', last_date='2002-01', inflation=True)
    request.cls.real_estate = ok.AssetList(symbols=['RUS_SEC.RE', 'MOW_PR.RE'], ccy='RUB',
                                           first_date='2010-01', last_date='2015-01', inflation=True)


@pytest.fixture(scope='class')
def _init_portfolio_values():
    return dict(
        symbols=['RUB.FX', 'MCFTR.INDX'],
        ccy='RUB',
        first_date='2015-01',
        last_date='2020-01',
        inflation=True,
    )


@pytest.fixture(scope='class')
def _init_portfolio(request, _init_portfolio_values):
    request.cls.portfolio = ok.Portfolio(**_init_portfolio_values)

    _init_portfolio_values['inflation'] = False
    request.cls.portfolio_no_inflation = ok.Portfolio(**_init_portfolio_values)

    _init_portfolio_values['first_date'] = '2019-02'
    request.cls.portfolio_short_history = ok.Portfolio(**_init_portfolio_values)


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
def init_efficient_frontier_values():
    return dict(
        symbols=['SPY.US', 'SBMX.MOEX'],
        ccy='RUB',
        first_date='2018-11',
        last_date='2020-02',
        inflation=True,
        n_points=2
    )


@pytest.fixture(scope='module')
def init_efficient_frontier(init_efficient_frontier_values):
    return ok.EfficientFrontier(**init_efficient_frontier_values)


@pytest.fixture(scope='module')
def init_efficient_frontier_bounds(init_efficient_frontier_values):
    bounds = ((0.0, 0.5), (0.0, 1.))
    return ok.EfficientFrontier(**init_efficient_frontier_values, bounds=bounds)


@pytest.fixture(scope='module')
def init_efficient_frontier_reb():
    ls = ['SPY.US', 'GLD.US']
    return ok.EfficientFrontierReb(symbols=ls, ccy='RUB', first_date='2019-01', last_date='2020-02', n_points=3)
