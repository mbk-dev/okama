
[![Documentation Status](https://img.shields.io/readthedocs/okama.svg?style=popout)](http://okama.readthedocs.io/)
[![Python](https://img.shields.io/badge/python-v3-brightgreen.svg)](https://www.python.org/)
[![PyPI Latest Release](https://img.shields.io/pypi/v/okama.svg)](https://pypi.org/project/okama/)
[![Coverage](https://coveralls.io/repos/github/mbk-dev/okama/badge.svg?branch=master)](https://coveralls.io/github/mbk-dev/okama?branch=master)
[![License](https://img.shields.io/pypi/l/okama.svg)](https://opensource.org/licenses/MIT)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbk-dev/okama/blob/master/examples/01%20howto.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Okama

_okama_ is a library with investment portfolio analyzing & optimization tools. CFA recommendations are used in quantitative finance.

_okama_ goes with **free** «end of day» historical stock markets data and macroeconomic indicators through API.
>...entities should not be multiplied without necessity
>
> -- <cite>William of Ockham (c. 1287–1347)</cite>

## Table of contents

- [Okama main features](#okama-main-features)
- [Financial data and macroeconomic indicators](#financial-data-and-macroeconomic-indicators)
  - [End of day historical data](#end-of-day-historical-data)
  - [Macroeconomic indicators](#macroeconomic-indicators)
  - [Other historical data](#other-historical-data)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Documentation](#documentation)
- [Financial Widgets](#financial-widgets)
- [RoadMap](#roadmap)
- [Contributing to okama](#contributing-to-okama)
- [Communication](#communication)

## Okama main features

- Investment portfolio constrained Markowitz Mean-Variance Analysis (MVA) and optimization
- Rebalanced portfolio optimization with constraints (multi-period Efficient Frontier)
- Monte Carlo Simulations for financial assets and investment portfolios
- Popular risk metrics: VAR, CVaR, semi-deviation, variance and drawdowns
- Different financial ratios: Sharpe ratio, Sortino ratio, Diversification ratio 
- Forecasting models according to normal, lognormal and other popular distributions
- Testing distribution on historical data
- Dividend yield and other dividend indicators for stocks
- Backtesting and comparing historical performance of broad range of assets and indexes in multiple currencies
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
For many countries (USA, United Kingdom, European Union, Russia, Israel etc.):  

- Inflation
- Central bank rates
- CAPE10 (Shiller P/E) Cyclically adjusted price-to-earnings ratios

### Other historical data

- Real estate prices
- Top bank rates

## Installation

`pip install okama`

The latest development version can be installed directly from GitHub:

`pip install git+https://github.com/mbk-dev/okama@dev`


## Getting started

### 1. Compare several assets from different stock markets. Get USD-adjusted performance

```python
import okama as ok

x = ok.AssetList(['SPY.US', 'BND.US', 'DBXD.XFRA'], ccy='USD')
x  # all examples are for Jupyter Notebook/iPython. For raw Python interpreter use 'print(x)' instead.

```
![](../images/images/readmi01.jpg?raw=true) 

Get the main parameters for the set:
```python
x.describe()
```
![](../images/images/readmi02.jpg?raw=true) 

Get the assets accumulated return, plot it and compare with the USD inflation:
```python
x.wealth_indexes.plot()
```
![](../images/images/readmi03.jpg?raw=true) 

### 2. Create a dividend stocks portfolio with base currency EUR

```python
weights = [0.3, 0.2, 0.2, 0.2, 0.1]
assets = ['T.US', 'XOM.US', 'FRE.XFRA', 'SNW.XFRA', 'LKOH.MOEX']
pf = ok.Portfolio(assets, weights=weights, ccy='EUR')
pf.table
```
![](../images/images/readmi04.jpg?raw=true) 

Plot the dividend yield of the portfolio (adjusted to the base currency).

```python
pf.dividend_yield.plot()
```
![](../images/images/readmi05.png?raw=true) 

### 3. Draw an Efficient Frontier for 2 popular ETF: SPY and GLD
```python
ls = ['SPY.US', 'GLD.US']
curr = 'USD'
last_date='2020-10'
# Rebalancing periods is one year (default value)
frontier = ok.EfficientFrontierReb(ls, last_date=last_date, ccy=curr, rebalancing_period='year')
frontier.names
```
![](../images/images/readmi06.jpg?raw=true) 

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
![](../images/images/readmi07.jpg?raw=true)   
<nowiki>*</nowiki> - *rebalancing period is one year*.

### 4. Get a Transition Map for allocations
```python
ls = ['SPY.US', 'GLD.US', 'BND.US']
map = ok.EfficientFrontier(ls, ccy='USD').plot_transition_map(x_axe='risk')
```
![](../images/images/readmi08.jpg?v23-11-2020,raw=true "Transition map")  

More examples are available in form of [Jupyter Notebooks](https://github.com/mbk-dev/okama/tree/master/examples).

## Documentation

The official documentation is hosted on readthedocs.org: [https://okama.readthedocs.io/](https://okama.readthedocs.io/)

## Financial Widgets
[okama-dash](https://github.com/mbk-dev/okama-dash) repository has interactive financial widgets (multi-page web application) 
build with _okama_ package and [Dash (plotly)](https://github.com/plotly/dash) framework. Working example is available at 
[okama.io](https://okama.io/).

![](https://github.com/mbk-dev/okama-dash/blob/images/images/main_page.jpg?raw=true) 

## RoadMap

The plan for _okama_ is to add more functions that will be useful to investors and asset managers.

- Add Omega ratio to EfficientFrontier, EfficientFrontierReb and Portfolio classes.
- Add withdrawals as an attribute of Portfolio class.
- Add Black-Litterman asset allocation 
- Accelerate optimization for multi-period Efficient Frontier: minimize_risk and maximize_risk methods of EfficientFrontierReb class.
- Make a single EfficientFrontier class for all optimizations: single-period or multu-period with rebalancing period as a parameter.
- Add different utility functions for optimizers: semi-deviation, VaR, CVaR, drawdowns etc.
- Add more functions based on suggestion of users.

## Contributing to okama

Contributions are *most welcome*. Have a look at the [Contribution Guide](https://github.com/mbk-dev/okama/blob/master/CONTRIBUTING.md) for more.  
Feel free to ask questions on [Discussuions](https://github.com/mbk-dev/okama/discussions).  
As contributors and maintainers to this project, you are expected to abide by okama' code of conduct. More information can be found at: [Contributor Code of Conduct](https://github.com/mbk-dev/okama/blob/master/CODE_OF_CONDUCT.md)

## Communication

For basic usage questions (e.g., "_Is XXX currency supported by okama?_") and for sharing ideas please use [GitHub Discussions](https://github.com/mbk-dev/okama/discussions/3).
Russian language community is available at [okama.io forums](https://community.okama.io).

## License

MIT
