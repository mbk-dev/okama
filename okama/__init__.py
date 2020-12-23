"""
*okama* is a Python package developed for asset allocation and investments portfolio optimization tasks
according to Modern Portfolio Theory (MPT).
=====================================================================
The package is supplied with **free** «end of day» historical stock markets data and macroeconomic indicators through API.

Main features:

- Investment portfolio constrained Markowitz Mean-Variance Analysis (MVA) and optimization
- Rebalanced portfolio optimization
- Monte Carlo Simulations for financial assets and investment portfolios
- Popular risk metrics: VAR, CVaR, semidiviation, variance and drawdowns
- Forecasting models according to normal and lognormal distribution
- Testing distribution on historical data
- Dividend yield and other dividend indicators for stocks
- Backtesting and comparing historical performance of broad range of assets and indexes in multiple currencies
- Methods to track the perfomance of index funds (ETF) and compare them with benchmarks
- Main macroeconomic indicators: inflation, central banks rates
- Matplotlib visualization scripts for the Efficient Frontier, Transition map and assets risk / return performance

"""

from okama.assets import Asset, AssetList, Portfolio
from okama.macro import Inflation, Rate
from okama.frontier import EfficientFrontier
from okama.frontier_reb import EfficientFrontierReb
from okama.plots import Plots
from okama.data import QueryData, API, search
from okama.helpers import Float, Frame, Rebalance, Date
from okama.settings import namespaces
import okama.settings

__version__ = '0.91'
