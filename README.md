<!-- buttons -->
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://pypi.org/project/okama/">
        <img src="https://img.shields.io/badge/pypi-v0.98-brightgreen.svg"
            alt="pypi"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
</p>

<!-- content -->

# Okama

_okama_ is a Python package developed for asset allocation and investment portfolio optimization tasks according to Modern Portfolio Theory (MPT).

The package is supplied with **free** «end of day» historical stock markets data and macroeconomic indicators through API.
>...entities should not be multiplied without necessity
>
> -- <cite>William of Ockham (c. 1287–1347)</cite>
## Okama main features

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

## Installation

`pip install okama`

## Getting started

### 1. Compare several assets from different stock markets. Get USD-adjusted performance

```python
import okama as ok
x = ok.AssetList(['SPY.US', 'BND.US', 'DBXD.XETR'], ccy='USD')
print(x)

```
![](../images/images/readmi01.jpg?raw=true) 

Get the main parameters for the set:
```python
x.describe(tickers=False)
```
![](../images/images/readmi02.jpg?raw=true) 

Get the assets accumulated return, plot it and compare with the USD inflation:
```python
x.wealth_indexes.plot()
```
![](../images/images/readmi03.jpg?raw=true) 

### 2. Create a dividend stocks portfolio with base currency EUR

```python
import okama.portfolio

weights = [0.3, 0.2, 0.2, 0.2, 0.1]
assets = ['T.US', 'XOM.US', 'FRE.XETR', 'SNW.XETR', 'LKOH.MOEX']
pf = okama.portfolio.Portfolio(assets, weights=weights, ccy='EUR')
print(pf)
```
![](../images/images/readmi04.jpg?raw=true) 

Plot the dividend yield for each group of assets (based on stock currency).
```python
pf.dividend_yield.plot()
```
![](../images/images/readmi05.jpg?raw=true) 

### 3. Draw an Efficient Frontier for 2 poular ETF: SPY and GLD
```python
ls = ['SPY.US', 'GLD.US']
curr = 'USD'
frontier = ok.EfficientFrontierReb(ls, last_date='2020-10', ccy=curr, reb_period='year')  # Rebalancing periods is one year (dafault value)
frontier.names
```
![](../images/images/readmi06.jpg?raw=true) 

Get the Efficient Frontier points for rebalanced portfolios and plot the chart with the assets risk/CAGR points:
```python
points = frontier.ef_points

fig = plt.figure(figsize=(12,6))
fig.subplots_adjust(bottom=0.2, top=1.5)
ok.Plots(ls, ccy=curr).plot_assets(kind='cagr')  # plots the assets points on the chart
ax = plt.gca()
ax.plot(points.Risk, points.CAGR) 
```
![](../images/images/readmi07.jpg?raw=true)   
<nowiki>*</nowiki> - *rebalancing period is one year*.

### 4. Get a Transition Map for allocations
```python
ls = ['SPY.US', 'GLD.US', 'BND.US']
map = ok.Plots(ls, ccy='USD').plot_transition_map(cagr=False)
```
![](../images/images/readmi08.jpg?v23-11-2020,raw=true "Transition map")  

More examples are available in [Jupyter Notebooks](https://github.com/mbk-dev/okama/tree/master/notebooks).

## Communication

For basic usage questions (e.g., "_Is XXX currency supported by okama?_") and for sharing ideas please use [GitHub Discussions](https://github.com/mbk-dev/okama/discussions/3).
Russian language community is available at [okama.io forums](https://community.okama.io/c/python-okama).
## Issues

We encourage you to report issues using the [Github tracker](https://github.com/mbk-dev/okama/issues). We welcome all kinds of issues, especially those related to correctness, documentation, performance, and feature requests.

## Contributing to okama

All contributions, bug reports, bug fixes, documentation improvements, enhancements, frontend implementation and ideas are welcome.

## License

MIT
