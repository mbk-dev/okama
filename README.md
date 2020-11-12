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

1. Get information about a single asset (or an index):
_import okama as ok_

```python
x = ok.Asset(&#39;SPY.US&#39;)

x.name
'''

1. Study the performance and compare the performance of several assets
2. Backtest the portfolio allocation
 Portfolio is an AssetList with weights.
3. Draw an Efficient Frontier
4. Get a Transition Map for allocations

More examples in Jupyter Notebooks:

-

## Communication

To communicate with the Okama developer community, create a Github issue or use the Okama mailing list. Please be respectful in your communications with the Okama community.

## Issues

We encourage you to report issues using the [Github tracker](https://github.com/mbk-dev/okama/issues). We welcome all kinds of issues, especially those related to correctness, documentation, performance, and feature requests.

## Contributing to okama

All contributions, bug reports, bug fixes, documentation improvements, enhancements, frontend implementation and ideas are welcome.

## License

MIT
