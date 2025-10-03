from copy import deepcopy

import pytest
import okama as ok

import okama.portfolios.cashflow_strategies
import okama.portfolios.dcf

# Portfolio
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
def portfolio_rebalanced_year(init_portfolio_values):
    return ok.Portfolio(**init_portfolio_values)


@pytest.fixture(scope="package")
def portfolio_not_rebalanced(init_portfolio_values):
    _portfolio_not_rebalanced = deepcopy(init_portfolio_values)
    _portfolio_not_rebalanced["rebalancing_strategy"] = ok.Rebalance(period="none")
    return ok.Portfolio(**_portfolio_not_rebalanced)


@pytest.fixture(scope="package")
def portfolio_rebalanced_month(init_portfolio_values):
    _portfolio_rebalanced_month = deepcopy(init_portfolio_values)
    _portfolio_rebalanced_month["rebalancing_strategy"] = ok.Rebalance(period="month")
    return ok.Portfolio(**_portfolio_rebalanced_month)


@pytest.fixture(scope="package")
def portfolio_no_inflation(init_portfolio_values):
    _portfolio_no_inflation = deepcopy(init_portfolio_values)
    _portfolio_no_inflation["inflation"] = False
    _portfolio_no_inflation["rebalancing_strategy"] = ok.Rebalance(period="month")
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
def init_mc_students():
    return dict(distribution="t", period=10, number=100)


@pytest.fixture(scope="package")
def init_mc_normal_small():
    return dict(distribution="norm", period=1, number=10)


@pytest.fixture(scope="package")
def portfolio_dcf(init_portfolio_dcf):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **init_portfolio_dcf[1])
    return pf_dcf


@pytest.fixture(scope="package")
def portfolio_dcf_no_inflation(init_portfolio_dcf):
    values_list = deepcopy(init_portfolio_dcf)
    values_list[0]["inflation"] = False
    # Create Portfolio
    pf = ok.Portfolio(**values_list[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **values_list[1])
    return pf_dcf


@pytest.fixture(scope="package")
def portfolio_dcf_discount_rate(init_portfolio_dcf):
    values_list = deepcopy(init_portfolio_dcf)
    values_list[1]["discount_rate"] = 0.08
    # Create Portfolio
    pf = ok.Portfolio(**values_list[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **values_list[1])
    return pf_dcf


@pytest.fixture(scope="function")
def portfolio_dcf_indexation(init_portfolio_dcf, init_mc_students):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **init_portfolio_dcf[1])
    pf_dcf.set_mc_parameters(**init_mc_students)
    # Cash Flow
    ind = okama.portfolios.portfolio.cashflow_strategy.IndexationStrategy(pf)  # create IndexationStrategy linked to the portfolio
    ind.initial_investment = 10_000  # add initial investments size
    ind.frequency = "year"  # set cash flow frequency
    ind.amount = -1_500  # set withdrawal size
    ind.indexation = "inflation"
    pf_dcf.cashflow_parameters = ind
    pf_dcf.use_discounted_values = False
    return pf_dcf


@pytest.fixture(scope="function")
def portfolio_dcf_indexation_small(init_portfolio_dcf, init_mc_normal_small):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **init_portfolio_dcf[1])
    pf_dcf.set_mc_parameters(**init_mc_normal_small)
    # Cash Flow
    ind = okama.portfolios.portfolio.cashflow_strategy.IndexationStrategy(pf)  # create IndexationStrategy linked to the portfolio
    ind.initial_investment = 10_000  # add initial investments size
    ind.frequency = "month"  # set cash flow frequency
    ind.amount = -1_500 / 12  # set withdrawal size
    ind.indexation = "inflation"
    pf_dcf.cashflow_parameters = ind
    pf_dcf.use_discounted_values = False
    return pf_dcf


@pytest.fixture(scope="function")
def portfolio_dcf_percentage(init_portfolio_dcf, init_mc_students):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **init_portfolio_dcf[1])
    pf_dcf.set_mc_parameters(**init_mc_students)
    # Cash Flow
    pc = okama.portfolios.portfolio.cashflow_strategy.PercentageStrategy(pf)  # create IndexationStrategy linked to the portfolio
    pc.initial_investment = 100_000  # add initial investments size
    pc.frequency = "half-year"  # set cash flow frequency
    pc.percentage = 0.04  # set withdrawal size
    pf_dcf.cashflow_parameters = pc
    pf_dcf.use_discounted_values = True
    return pf_dcf


@pytest.fixture(scope="function")
def portfolio_dcf_time_series(init_portfolio_dcf, init_mc_students):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **init_portfolio_dcf[1])
    pf_dcf.set_mc_parameters(**init_mc_students)
    # Cash Flow
    d = {"2018-02": 2_000, "2024-03": -4_000}  # contribution  # withdrawal
    ts = okama.portfolios.portfolio.cashflow_strategy.TimeSeriesStrategy(pf)  # create TimeSeriesStrategy linked to the portfolio
    ts.time_series_dic = d  # use the dictionary to set cash flow
    ts.initial_investment = 1_000  # add initial investments size
    pf_dcf.cashflow_parameters = ts
    pf_dcf.use_discounted_values = False
    return pf_dcf


