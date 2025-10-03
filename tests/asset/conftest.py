from copy import deepcopy

import pytest
import okama as ok
from pathlib import Path

import okama.portfolios.cashflow_strategies
import okama.portfolios.dcf

data_folder = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def init_asset_spy():
    return ok.Asset(symbol="SPY.US")


@pytest.fixture(scope="module")
def init_asset_eurusd():
    return ok.Asset(symbol="EURUSD.FX")


@pytest.fixture(scope="module")
def init_asset_berkshire():
    return ok.Asset(symbol="BRK.A.US")


# @pytest.fixture(scope="module")
# def init_asset_pif():
#     return ok.Asset(symbol="0165-70287767.PIF")


@pytest.fixture(scope="module")
def init_asset_usdrub():
    return ok.Asset(symbol="RUB.FX")


@pytest.fixture(scope="module")
def init_asset_index_mcftr():
    return ok.Asset(symbol="MCFTR.INDX")
