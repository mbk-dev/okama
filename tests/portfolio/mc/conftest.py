from copy import deepcopy

import pytest
import okama as ok
from pathlib import Path

import okama.portfolios.cashflow_strategies
import okama.portfolios.dcf

data_folder = Path(__file__).parent / "data"


@pytest.fixture(scope="package")
def init_portfolio_values():
    return dict(
        assets=["MCFTR.INDX"],  # index values are better as they are not changing (adjusted_close)
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf1.PF",
    )

# DCF Scenarios
@pytest.fixture(scope="package")
def init_portfolio_dcf(init_portfolio_values):
    _portfolio_values = deepcopy(init_portfolio_values)
    _portfolio_dcf_values = dict(
        discount_rate=None,
    )
    return [_portfolio_values, _portfolio_dcf_values]


@pytest.fixture(scope="package")
def portfolio_dcf(init_portfolio_dcf):
    pf = ok.Portfolio(**init_portfolio_dcf[0])
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **init_portfolio_dcf[1])
    return pf_dcf


# @pytest.fixture(scope="package")
# def portfolio_dcf_no_inflation(init_portfolio_dcf):
#     values_list = deepcopy(init_portfolio_dcf)
#     values_list[0]["inflation"] = False
#     # Create Portfolio
#     pf = ok.Portfolio(**values_list[0])
#     pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, **values_list[1])
#     return pf_dcf

# Monte Carlo Scenarios
@pytest.fixture(scope="package")
def init_mc_students():
    return dict(distribution="t", distribution_parameters=(None, None, None), period=10, mc_number=100)


@pytest.fixture(scope="package")
def init_mc_normal_small():
    return dict(distribution="norm", distribution_parameters=(None, None), period=1, mc_number=10)


@pytest.fixture(scope="package")
def init_mc_lognormal_small():
    return dict(distribution="lognorm", distribution_parameters=(None, None, None), period=1, mc_number=10)


@pytest.fixture(scope="package")
def mc_normal_small(portfolio_dcf, init_mc_normal_small):
    mc = ok.MonteCarlo(parent=portfolio_dcf, **init_mc_normal_small)
    return mc

@pytest.fixture(scope="package")
def mc_lognormal_small(portfolio_dcf, init_mc_lognormal_small):
    mc = ok.MonteCarlo(parent=portfolio_dcf, **init_mc_lognormal_small)
    return mc

@pytest.fixture(scope="package")
def mc_students(portfolio_dcf, init_mc_students):
    mc = ok.MonteCarlo(parent=portfolio_dcf, **init_mc_students)
    return mc


