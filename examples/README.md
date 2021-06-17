# _Okama_ examples
Jupyter Notebooks in this directory demonstrate basic use of _okama_.

1. [howto](https://github.com/mbk-dev/okama/blob/master/examples/01%20howto.ipynb)
Main features of _okama_ package: Assets, AssetsList and Portfolio objects.
   
2. [index funds performance](https://github.com/mbk-dev/okama/blob/master/examples/02%20index%20funds%20perfomance.ipynb)
Compare ETF, mutual funds to their indexes. See the _tracking difference_, _tracking error_ and _beta_. 
   Calculate correlation. 
   
3. [efficient frontier single period](https://github.com/mbk-dev/okama/blob/master/examples/03%20efficient%20frontier%20single%20period.ipynb)
EfficientFrontier class can be used for "classic" Markowitz frontiers where all portfolios are rebalanced monthly 
   (single period optimization). It's the most easy and fast way to draw an Efficient Frontier.
   
4. [efficient frontier multi-period](https://github.com/mbk-dev/okama/blob/master/examples/04%20efficient%20frontier%20multi-period.ipynb)
Multi-period approach uses portfolios with custom rebalancing (annually, quarterly etc.) or not rebalanced at all.
   
5. [backtesting distribution](https://github.com/mbk-dev/okama/blob/master/examples/05%20backtesting%20distribution.ipynb)
Examples for "backtesting" the distribution of portfolio returns on the historical data. 
   Portfolio performance can be tested according to Jarque-Bera, Kolmogorov-Smirnov 
   tests for normal, lognormal and other types of probability distributions.
   
6. [forecasting](https://github.com/mbk-dev/okama/blob/master/notebooks/06%20forecasting.ipynb)
Examples of forecasting investment portfolio performance according to normal, lognormal or historical distribution.
