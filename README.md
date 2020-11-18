# Okama

_okama_ is a Python package developed for asset allocation and investments portfolio optimization tasks according to Modern Portfolio Theory.

All classes and methods of okama are supplied with **free** «end of day» historical stock markets data and macroeconomic indicators through API.

## Okama main features

- Investment portfolio constrained Markowitz Mean-Variance Analysis (MVA) and optimization
- Rebalanced portfolio optimization
- Monte Carlo Simulations for financial assets and investment portfolios
- Popular risk metrics: VAR, CVaR, semidiviation, variance and drawdowns
- Forecasting models according to normal and lognormal distribution
- Testing distribution on historical data
- Dividend yield and other dividend indicators for stocks
- Backtesting and comparing historical performance of broad range of assets and indexes in multiple currencies
- Main macroeconomic indicators: inflation, central banks rates
- Matplotlib visualization scripts for the Efficient Frontier, Transition map and assets risk / return performance

## Financial data and macroeconomic indicators

### End of day historical data

- Stocks and ETF for main world markets
- Mutual funds
- Commodities
- Currencies
- Stock indexes

### Macroeconomic indicators

- Inflation
- Central bank rates

### Other historical data

- Real estate prices
- Top bank rates

## Table of contents

## Installation

`pip install okama`

`conda install okama`

## Getting started

### 1. Compare several assets from different stock markets. Get the USD-adjusted perfomance:

```python
import okama as ok
x = ok.AssetList(['SPY.US', 'BND.US', 'DBXD.XETR'], curr='USD')
print(x)

```
![](../images/images/image1.jpg?raw=true) 

Get the main parameters for the set:
```python
x.describe(tickers=False)
```
![](../images/images/image2.jpg?raw=true) 

Get the assets accumulated return, plot it and compare with the USD inflation:
```python
x.wealth_indexes.plot()
```
![](../images/images/image3.jpg?raw=true) 

### 2. Create a dividend stocks portfolio with base currency EUR.
```python
weights = [0.3, 0.2, 0.2, 0.2, 0.1]
assets = ['T.US', 'XOM.US', 'FRE.XETR', 'SNW.XETR', 'LKOH.MOEX']
pf = ok.Portfolio(assets, weights=weights, curr='EUR')
print(pf)
```
img

Plot the dividend yield for each group of assets (based on stock currency).
```python
pf.dividend_yield.plot()
```
img

### 3. Draw an Efficient Frontier for 2 poular ETF: SPY and GLD.
```python
ls = ['SPY.US', 'GLD.US']
curr = 'USD'
frontier = ok.EfficientFrontierReb(ls, last_date='2020-10', curr=curr, reb_period='Y')  # Rebalancing periods is one year (dafault value)
frontier.names
```
img

Get the Efficient Frontier points and plot the chart with the assets risk/CAGR points:
```python
points = frontier.ef_points

fig = plt.figure(figsize=(12,6))
fig.subplots_adjust(bottom=0.2, top=1.5)
ok.Plots(ls, curr=curr).plot_assets(kind='cagr')  # plots the assets points on the chart
ax = plt.gca()
ax.plot(points.Risk, points.CAGR) 
```
img

### 4. Get a Transition Map for allocations.
```python
ls = ['SPY.US', 'GLD.US', 'BND.US']
map = ok.Plots(ls, curr='USD').plot_transition_map(cagr=False)
```

More examples in Jupyter Notebooks:

-

## Communication

To communicate with the Okama developer community, create a Github issue or use the Okama mailing list. Please be respectful in your communications with the Okama community.
For basic usage questions (e.g., "Is XXX currency supported by okama?"), please use **the community forums** instead.

## Issues

We encourage you to report issues using the [Github tracker](https://github.com/mbk-dev/okama/issues). We welcome all kinds of issues, especially those related to correctness, documentation, performance, and feature requests.

## Contributing to okama

All contributions, bug reports, bug fixes, documentation improvements, enhancements, frontend implementation and ideas are welcome.

## License

MIT
