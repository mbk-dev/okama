.. figure:: /images/ef.png
    :scale: 60 %
    :align: center

.. raw:: html

    <embed>
        <p align="center">
            <a href="https://www.python.org/">
                <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
                    alt="python"></a> &nbsp;
            <a href="https://pypi.org/project/okama/">
                <img src="https://img.shields.io/badge/pypi-v1.0.1-brightgreen.svg"
                    alt="pypi"></a> &nbsp;
            <a href='https://coveralls.io/github/mbk-dev/okama?branch=master'>
                <img src='https://coveralls.io/repos/github/mbk-dev/okama/badge.svg?branch=master'
                alt='Coverage Status' /></a>
            <a href="https://opensource.org/licenses/MIT">
                <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
                    alt="MIT license"></a> &nbsp;
        </p>
    </embed>

.. meta::
   :title: Okama Documentation: Python library for investments
   :description lang=en: okama is a Python library with investment portfolio analyzing & optimization tools
   :keywords lang=en: okama, python, investments, portfolio optimization, quantitative finance, asset allocation, rebalancing, efficient frontier, financial assets

===================
Okama Documentation
===================

*okama* is a library with investment portfolio analyzing & optimization tools. CFA recommendations are used in quantitative finance.

*okama* goes with **free** «end of day» historical stock markets data and macroeconomic indicators through API.

    ...entities should not be multiplied without necessity

    *-- William of Ockham (c. 1287–1347)*

Okama main features
*******************

* Investment portfolio constrained Markowitz Mean-Variance Analysis (MVA) and optimization
* Rebalanced portfolio optimization with constraints (multi-period Efficient Frontier)
* Monte Carlo Simulations for financial assets and investment portfolios
* Popular risk metrics: VAR, CVaR, semi-deviation, variance and drawdowns
* Forecasting models according to normal and lognormal distribution
* Testing distribution on historical data
* Dividend yield and other dividend indicators for stocks
* Backtesting and comparing historical performance of broad range of assets and indexes in multiple currencies
* Methods to track the performance of index funds (ETF) and compare them with benchmarks
* Main macroeconomic indicators: inflation, central banks rates
* Matplotlib visualization scripts for the Efficient Frontier, Transition map and assets risk / return performance

Financial data and macroeconomic indicators
*******************************************
*okama* can be used with free financial data available through API.

End of day historical data
==========================

* Stocks and ETF for main world markets
* Mutual funds
* Commodities
* Currencies
* Stock indexes

Macroeconomic indicators
========================

* Inflation for many countries (USA, United Kingdom, European Union, Russia etc.)
* Central bank rates

Other historical data
=====================

* Real estate prices
* Top bank rates


Installation
************

Okama can be installed from `PyPI <https://pypi.org/project/okama/>`_:

.. code:: text

    pip install okama

The latest development version can be installed directly from GitHub:

.. code:: text

    pip install git+https://github.com/mbk-dev/okama@dev

.. warning::

    The development version of *okama* can have technical and financial issues. Please use carefully at your own risk.

.. toctree::
    :maxdepth: 1
    :caption: Quick Start

    /jupyter/quickstart

.. toctree::
    :maxdepth: 1
    :caption: Index Funds Performance

    /jupyter/funds

.. toctree::
    :maxdepth: 1
    :caption: Investment Portfolios

    /jupyter/portfolio

.. autosummary::
    :toctree: stubs
    :template: custom-class-template.rst
    :caption: Main Classes

    okama.Asset
    okama.AssetList
    okama.Portfolio

.. autosummary::
    :toctree: stubs
    :template: custom-class-template-no-inherited.rst

    okama.EfficientFrontier
    okama.EfficientFrontierReb
    okama.Plots


Indices and tables
******************

* :ref:`genindex`
* :ref:`search`
