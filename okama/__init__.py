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
from importlib_metadata import version

from okama.asset import Asset
from okama.asset_list import AssetList
from okama.portfolio import Portfolio
from okama.macro import Inflation, Rate
from okama.frontier.multi_period import EfficientFrontierReb
from okama.frontier.single_period import EfficientFrontier
from okama.api.data_queries import QueryData
from okama.api.search import search
from okama.api.api_methods import API
from okama.api.namespaces import (
    namespaces,
    assets_namespaces,
    macro_namespaces,
    symbols_in_namespace,
)
from okama.common.helpers.helpers import Float, Frame, Rebalance, Date
import okama.settings

__version__ = version('okama')
