<p align="center">
  <img src="https://raw.githubusercontent.com/mbk-dev/okama/images/images/Okama2.jpg" alt="okama — investment portfolio analysis and optimization library" width="600">
</p>

# Okama

[![Documentation Status](https://img.shields.io/readthedocs/okama.svg?style=popout)](https://okama.readthedocs.io/)
[![Python](https://img.shields.io/pypi/pyversions/okama.svg)](https://www.python.org/)
[![PyPI Latest Release](https://img.shields.io/pypi/v/okama.svg)](https://pypi.org/project/okama/)
[![Downloads](https://static.pepy.tech/badge/okama)](https://pepy.tech/project/okama)
[![Coverage](https://coveralls.io/repos/github/mbk-dev/okama/badge.svg?branch=master)](https://coveralls.io/github/mbk-dev/okama?branch=master)
[![License](https://img.shields.io/pypi/l/okama.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbk-dev/okama/blob/master/examples/01%20howto.ipynb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

_okama_ is a Python library for investment portfolio analysis and optimization. It applies concepts commonly used in quantitative finance.

_okama_ provides access to **free** end-of-day historical market data and macroeconomic indicators through an API.

> ...entities should not be multiplied without necessity
>
> -- <cite>William of Ockham (c. 1287–1347)</cite>

## Table of contents

- [Okama main features](#okama-main-features)
- [Financial data and macroeconomic indicators](#financial-data-and-macroeconomic-indicators)
  - [End of day historical data](#end-of-day-historical-data)
  - [Currencies](#currencies)
  - [Macroeconomic indicators](#macroeconomic-indicators)
  - [Other historical data](#other-historical-data)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Examples](#examples)
- [Documentation](#documentation)
- [Financial Widgets](#financial-widgets)
- [MCP server](#mcp-server)
- [Roadmap](#roadmap)
- [Contributing to okama](#contributing-to-okama)
- [Communication](#communication)

## Okama main features

- Investment portfolio constrained Markowitz Mean-Variance Analysis (MVA) and optimization
- Rebalanced portfolio optimization with constraints (multi-period Efficient Frontier)
- Advanced rebalancing strategies: Rebalancing-bands (threshold-based), Calendar-based or hybrid
- Investment portfolios with complex contributions / withdrawals cash flows (DCF)
- Money-weighted internal rate of return (IRR/MWRR) for portfolio cash flows — on historical data and across Monte Carlo forecast paths
- Monte Carlo Simulations for financial assets and investment portfolios, reproducible with a random `seed`
- Forecasting with popular theoretical distributions: normal, lognormal and Student's (T)
- Degrees of freedom optimization for Student's t-distribution to fit well at a given confidence level
- Testing distributions on historical data
- Popular risk metrics: VAR, CVaR, semi-deviation, variance and drawdowns
- Different financial ratios: CAPE10, Sharpe ratio, Sortino ratio, Diversification ratio
- Dividend yield and other dividend indicators for stocks
- Backtesting and comparing historical performance of a broad range of assets and indexes in multiple currencies
- Methods to track the performance of index funds (ETF) and compare them with benchmarks
- Main macroeconomic indicators: inflation, central banks rates
- Matplotlib visualization scripts for the Efficient Frontier, Transition map and assets risk / return performance

## Financial data and macroeconomic indicators

### End of day historical data

- Stocks and ETF for main world markets
- Mutual funds
- Commodities
- Stock indexes

### Currencies

- FX currencies
- Crypto currencies
- Central bank exchange rates

### Macroeconomic indicators
For many countries (China, USA, United Kingdom, European Union, Russia, Israel etc.):  

- Inflation
- Central bank rates
- CAPE10 (Shiller P/E) Cyclically adjusted price-to-earnings ratios

### Other historical data

- Real estate prices
- Top bank rates

## Installation

### Requirements

- Python **3.11** or newer
- Core dependencies: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [scipy](https://scipy.org/) (plus `matplotlib`, `pyarrow`, `statsmodels`, `arch` and others). See [pyproject.toml](pyproject.toml) for the full list of dependencies and version constraints.

### Install from PyPI

```bash
pip install okama
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add okama          # add to a uv-managed project
uv pip install okama  # or pip-style install into the active environment
```

### Install the latest development version from GitHub

```bash
git clone -b dev https://github.com/mbk-dev/okama.git
cd okama
poetry install
```

## Getting started

> [!NOTE]
> All examples below are written for Jupyter Notebook / IPython. In a plain Python interpreter, wrap the displayed objects in `print(...)`.

### 1. Compare several assets from different stock markets. Get USD-adjusted performance

```python
import okama as ok

x = ok.AssetList(['SPY.US', 'BND.US', 'DBXD.XFRA'], ccy='USD')
x
```
![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi01.jpg)

Get the main parameters for the set:
```python
x.describe()
```
![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi02.jpg)

Get the assets accumulated return, plot it and compare with the USD inflation:
```python
x.wealth_indexes.plot()
```
![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi03.jpg)

### 2. Create a dividend stocks portfolio with base currency EUR

```python
weights = [0.3, 0.2, 0.2, 0.2, 0.1]
assets = ['T.US', 'XOM.US', 'FRE.XFRA', 'SNW.XFRA', 'LKOH.MOEX']
pf = ok.Portfolio(assets, weights=weights, ccy='EUR')
pf.table
```
![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi04.jpg)

Plot the dividend yield of the portfolio (adjusted to the base currency).

```python
pf.dividend_yield.plot()
```
![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi05.png)

### 3. Draw an Efficient Frontier for 2 popular ETF: SPY and GLD

```python
ls = ['SPY.US', 'GLD.US']
curr = 'USD'
last_date = '2020-10'
# Rebalancing period is one year (default value)
frontier = ok.EfficientFrontier(ls, last_date=last_date, ccy=curr, rebalancing_strategy=ok.Rebalance(period='year'))
frontier.names
```
![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi06.jpg)

Get the Efficient Frontier points for rebalanced portfolios and plot the chart with the assets risk/CAGR points:
```python
import matplotlib.pyplot as plt

points = frontier.ef_points

fig = plt.figure(figsize=(12,6))
fig.subplots_adjust(bottom=0.2, top=1.5)
frontier.plot_assets(kind='cagr')  # plots the assets points on the chart
ax = plt.gca()
ax.plot(points.Risk, points.CAGR) 
```
![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi07.jpg)

### 4. Get a Transition Map for allocations

```python
ls = ['SPY.US', 'GLD.US', 'BND.US']
ok.EfficientFrontier(ls, ccy='USD').plot_transition_map(x_axe='risk')
```
![Transition map](https://raw.githubusercontent.com/mbk-dev/okama/images/images/readmi08.jpg)

## Examples

More examples are available as [Jupyter Notebooks](https://github.com/mbk-dev/okama/tree/master/examples):

1. [howto](https://github.com/mbk-dev/okama/blob/master/examples/01%20howto.ipynb) — main features of the _okama_ package: `Asset`, `AssetList`, and `Portfolio` objects.
2. [index funds performance](https://github.com/mbk-dev/okama/blob/master/examples/02%20index%20funds%20perfomance.ipynb) — compare ETFs and mutual funds with their benchmarks: tracking difference, tracking error, beta, and correlation.
3. [investment portfolios](https://github.com/mbk-dev/okama/blob/master/examples/03%20investment%20portfolios.ipynb) — portfolio properties and comparison of multiple portfolios.
4. [investment portfolios with DCF](https://github.com/mbk-dev/okama/blob/master/examples/04%20investment%20portfolios%20with%20DCF.ipynb) — portfolio strategies with cash flows (withdrawals and contributions), backtesting and longevity forecasts with Monte Carlo simulation.
5. [macroeconomics](https://github.com/mbk-dev/okama/blob/master/examples/05%20macroeconomics%20-%20inflation%20rates.ipynb) — historical inflation, key rates, and other macroeconomic indicators.
6. [efficient frontier single period](https://github.com/mbk-dev/okama/blob/master/examples/06%20efficient%20frontier%20single%20period.ipynb) — classic Markowitz frontiers with monthly rebalanced portfolios (`EfficientFrontierSingle`).
7. [efficient frontier multi-period](https://github.com/mbk-dev/okama/blob/master/examples/07%20efficient%20frontier%20multi-period.ipynb) — multi-period optimization with custom rebalancing frequencies or without rebalancing.
8. [backtesting distribution](https://github.com/mbk-dev/okama/blob/master/examples/08%20backtesting%20distribution.ipynb) — backtest portfolio return distributions with Jarque-Bera, Kolmogorov-Smirnov, and related diagnostics.
9. [financial database](https://github.com/mbk-dev/okama/blob/master/examples/09%20financial%20database.ipynb) — query the okama database for stocks, ETFs, mutual funds, indexes, currencies, and macroeconomic data.
10. [forecasting](https://github.com/mbk-dev/okama/blob/master/examples/10%20forecasting.ipynb) — forecast portfolio performance with normal, lognormal, Student's t, and historical distributions.
11. [rebalancing portfolio](https://github.com/mbk-dev/okama/blob/master/examples/11%20rebalancing%20portfolio.ipynb) — compare calendar-based, threshold-based, and hybrid portfolio rebalancing strategies.

> [!TIP]
> You can try the notebooks on Google Colab without installing anything: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbk-dev/okama/blob/master/examples/01%20howto.ipynb)

## Documentation

The official documentation is hosted on readthedocs.org: [https://okama.readthedocs.io/](https://okama.readthedocs.io/)

## Financial Widgets
[okama-dash](https://github.com/mbk-dev/okama-dash) repository has interactive financial widgets (multi-page web application) 
built with the _okama_ package and [Dash (plotly)](https://github.com/plotly/dash) framework. Working example is available at 
[okama.io](https://okama.io/).

![](https://raw.githubusercontent.com/mbk-dev/okama/images/images/main_page.jpg) 

## MCP server

[okama-mcp](https://github.com/mbk-dev/okama-mcp) is an MCP (Model Context Protocol) server that exposes
the _okama_ toolkit to AI assistants — Claude Desktop, Claude Code, Cursor, and any other MCP-compatible
client. Ask the AI to backtest a portfolio, build an efficient frontier, or run a Monte Carlo retirement
forecast — it calls _okama_ directly, no Python code needed.

```bash
uvx okama-mcp stdio  # run straight from PyPI
```

okama-mcp is free and open source — no hosted service, no registration; you run it yourself, locally or
on your own server. See [mcp.okama.io](https://mcp.okama.io) for installation and client configuration.

## Roadmap

The plan for _okama_ is to add more functions that will be useful to investors and asset managers.

- Add support for a series of investment portfolios (a financial plan comprising multiple investment strategies, each active until a specific date, after which it transitions to another).
- Add multidimensional Monte Carlo with Ledoit-Wolf shrinkage
- Add Omega ratio to EfficientFrontier and Portfolio classes.
- Add Black-Litterman asset allocation 
- Add different utility functions for optimizers: IRR, portfolio survival period, semi-deviation, VaR, CVaR, drawdowns etc.
- Add more functions based on suggestions from users.

## Contributing to okama

Contributions are *most welcome*. Have a look at the [Contribution Guide](https://github.com/mbk-dev/okama/blob/master/CONTRIBUTING.md) for more.  
Feel free to ask questions on [Discussions](https://github.com/mbk-dev/okama/discussions).  
As contributors and maintainers to this project, you are expected to abide by okama's code of conduct. More information can be found at: [Contributor Code of Conduct](https://github.com/mbk-dev/okama/blob/master/CODE_OF_CONDUCT.md)

## Communication

For basic usage questions (e.g., "_Is XXX currency supported by okama?_") and for sharing ideas please use [GitHub Discussions](https://github.com/mbk-dev/okama/discussions).
Russian language community is available at [okama.io forums](https://community.okama.io).
