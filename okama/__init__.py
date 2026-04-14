"""
*okama* is a Python package developed for asset allocation and investments portfolio optimization tasks
according to Modern Portfolio Theory (MPT).
=====================================================================
The package is supplied with **free** «end of day» historical stock markets data and macroeconomic indicators through API.

Main features:

- Investment portfolio constrained Markowitz Mean-Variance Analysis (MVA) and optimization
- Rebalanced portfolio optimization
- Monte Carlo Simulations for financial assets and investment portfolios
- Popular risk metrics: VAR, CVaR, semi-deviation, variance and drawdowns
- Forecasting models according to normal and lognormal distribution
- Testing distribution on historical data
- Dividend yield and other dividend indicators for stocks
- Backtesting and comparing historical performance of broad range of assets and indexes in multiple currencies
- Methods to track the performance of index funds (ETF) and compare them with benchmarks
- Main macroeconomic indicators: inflation, central banks rates
- Matplotlib visualization scripts for the Efficient Frontier, Transition map and assets risk / return performance

"""

from importlib.metadata import version  # noqa: I001
from typing import Any

from okama.asset import Asset  # noqa: F401
from okama.asset_list import AssetList  # noqa: F401
from okama.portfolios.core import (
    Portfolio,  # noqa: F401
)
from okama.portfolios.mc import MonteCarlo  # noqa: F401
from okama.portfolios.dcf import PortfolioDCF  # noqa: F401
from okama.portfolios.cashflow_strategies import (
    CashFlow,  # noqa: F401
    IndexationStrategy,  # noqa: F401
    PercentageStrategy,  # noqa: F401
    TimeSeriesStrategy,  # noqa: F401
    VanguardDynamicSpending,  # noqa: F401
    CutWithdrawalsIfDrawdown,  # noqa: F401
)
from okama.macro import Inflation, Rate, Indicator  # noqa: F401
from okama.frontier.multi_period import EfficientFrontier  # noqa: F401
from okama.frontier.single_period import EfficientFrontierSingle  # noqa: F401
from okama.api.data_queries import QueryData  # noqa: F401
from okama.api.search import search  # noqa: F401
from okama.api.api_methods import API  # noqa: F401
import okama.api.namespaces
from okama.api.namespaces import symbols_in_namespace  # noqa: F401
from okama.common.helpers.helpers import Float, Frame, Date  # noqa: F401
from okama.common.helpers.rebalancing import Rebalance  # noqa: F401
import okama.settings


def __getattr__(name: str) -> Any:
    """
    Lazily expose selected API metadata at the package level.

    Parameters
    ----------
    name : str
        Public attribute name requested from the ``okama`` package.

    Returns
    -------
    Any
        Cached namespace metadata for supported attribute names.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported dynamic package attribute.

    Notes
    -----
    Supported dynamic attributes include ``okama.namespaces``,
    ``okama.assets_namespaces``, ``okama.macro_namespaces``, and
    ``okama.no_dividends_namespaces``.

    ``okama.namespaces`` returns a dictionary of available namespace codes and
    their descriptions.
    """
    if name == "namespaces":
        return okama.api.namespaces.get_namespaces()
    elif name == "assets_namespaces":
        return okama.api.namespaces.get_assets_namespaces()
    elif name == "macro_namespaces":
        return okama.api.namespaces.get_macro_namespaces()
    elif name == "no_dividends_namespaces":
        return okama.api.namespaces.no_dividends_namespaces()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__version__ = version("okama")
