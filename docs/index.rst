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
                <img src="https://img.shields.io/pypi/v/okama.svg"
                    alt="PyPI Latest Release"></a> &nbsp;
            <a href='https://coveralls.io/github/mbk-dev/okama?branch=master'>
                            <img src='https://coveralls.io/repos/github/mbk-dev/okama/badge.svg?branch=master'
                            alt='Coverage Status' /></a>
            <a href="https://opensource.org/licenses/MIT">
                <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
                    alt="MIT license"></a> &nbsp;
            <a href="https://colab.research.google.com/github/mbk-dev/okama/blob/master/examples/01%20howto.ipynb">
                <img src="https://colab.research.google.com/assets/colab-badge.svg"
                    alt="MIT license"></a> &nbsp;
        </p>
    </embed>

.. meta::
   :title: Okama Documentation: Python library for investments
   :description lang=en: okama is a Python library for investment portfolio analysis and optimization
   :keywords lang=en: okama, python, investments, portfolio optimization, quantitative finance, asset allocation, rebalancing, efficient frontier, financial assets

===================
Okama Documentation
===================

*okama* is a Python library for investment portfolio analysis and optimization. It applies concepts commonly used in quantitative finance.

*okama* provides access to **free** end-of-day historical market data and macroeconomic indicators through an API.

    ...entities should not be multiplied without necessity

    *-- William of Ockham (c. 1287–1347)*

Okama main features
*******************

* Constrained Markowitz Mean-Variance Analysis (MVA) and portfolio optimization
* Multi-period Efficient Frontier optimization with rebalancing constraints
* Investment portfolios with contribution and withdrawal cash flows (DCF)
* Monte Carlo simulations for financial assets and investment portfolios
* Popular risk metrics: VaR, CVaR, semideviation, variance, and drawdowns
* Forecasting models based on normal, lognormal, and Student's t distributions
* Distribution fitting and goodness-of-fit testing on historical data
* Dividend yield and other dividend indicators for stocks
* Backtesting and comparing the historical performance of a broad range of assets and indexes in multiple currencies
* Methods for tracking the performance of index funds (ETFs) and comparing them with benchmarks
* Main macroeconomic indicators: inflation, central bank rates, and financial ratios
* Matplotlib visualizations for the Efficient Frontier, Transition Map, and asset risk/return performance

Financial data and macroeconomic indicators
*******************************************
*okama* can work with free financial data available through its API.

End of day historical data
==========================

* Stocks and ETFs for major world markets
* Mutual funds
* Commodities
* Currencies
* Stock indexes

Macroeconomic indicators
========================

For several countries, including the USA, the United Kingdom, the European Union, Russia, and Israel:

* Inflation
* Central bank rates
* CAPE10 (Shiller P/E), or cyclically adjusted price-to-earnings ratios

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

    git clone https://github.com/mbk-dev/okama@dev
    poetry install

.. warning::

    The development version of *okama* may have technical and financial issues. Use it carefully and at your own risk.

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

Main
****

Assets & Portfolio
==================

.. autosummary::
    :toctree: stubs
    :template: custom-class-template.rst
    :caption: Main

    okama.Asset
    okama.AssetList
    okama.Portfolio
    okama.Rebalance

Cash Flows & DCF
================

.. autosummary::
    :toctree: stubs
    :template: custom-class-template.rst

    okama.PortfolioDCF
    okama.MonteCarlo
    okama.IndexationStrategy
    okama.PercentageStrategy
    okama.TimeSeriesStrategy
    okama.VanguardDynamicSpending
    okama.CutWithdrawalsIfDrawdown

Efficient Frontier
==================

.. autosummary::
    :toctree: stubs
    :template: custom-class-template-no-inherited.rst

    okama.EfficientFrontier

Macroeconomics
==============

.. autosummary::
    :toctree: stubs
    :template: custom-class-template.rst
    :caption: Macroeconomics

    okama.Inflation
    okama.Rate
    okama.Indicator

Data Access & Search
====================

Use these helpers to discover available namespaces and find supported symbols before creating assets or portfolios.

:py:func:`okama.search`
    Search symbols by ticker, name, or ISIN.

:py:func:`okama.symbols_in_namespace`
    Return all symbols available in a namespace.

:py:data:`okama.namespaces`
    Returns a dictionary of available data namespaces and their descriptions.

.. toctree::
    :hidden:
    :caption: Data Access & Search

    stubs/okama.search
    stubs/okama.symbols_in_namespace
    okama.namespaces

Indices and tables
******************

* :ref:`genindex`
* :ref:`search`
