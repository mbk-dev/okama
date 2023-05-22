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
from importlib.metadata import version

from okama.asset import Asset
from okama.asset_list import AssetList
from okama.portfolio import Portfolio
from okama.macro import Inflation, Rate, Indicator
from okama.frontier.multi_period import EfficientFrontierReb
from okama.frontier.single_period import EfficientFrontier
from okama.api.data_queries import QueryData
from okama.api.search import search
from okama.api.api_methods import API
import okama.api.namespaces
from okama.api.namespaces import symbols_in_namespace
from okama.common.helpers.helpers import Float, Frame, Rebalance, Date
import okama.settings


def __getattr__(name):
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
