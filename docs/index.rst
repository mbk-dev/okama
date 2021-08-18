.. raw:: html

    <meta prefix="og: http://ogp.me/ns#" property="og:title" content="Okama: Python package for investments" />
    <meta prefix="og: http://ogp.me/ns#" property="og:description" content="Investment portfolio and stocks analyzing tools for Python with free historical data" />

    <embed>
        <p align="center">
            <a href="https://www.python.org/">
                <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
                    alt="python"></a> &nbsp;
            <a href="https://pypi.org/project/okama/">
                <img src="https://img.shields.io/badge/pypi-v1.0.0-brightgreen.svg"
                    alt="pypi"></a> &nbsp;
            <a href='https://coveralls.io/github/mbk-dev/okama?branch=master'>
                <img src='https://coveralls.io/repos/github/mbk-dev/okama/badge.svg?branch=master'
                alt='Coverage Status' /></a>
            <a href="https://opensource.org/licenses/MIT">
                <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
                    alt="MIT license"></a> &nbsp;
        </p>
    </embed>

==============================
Welcome to okama documentation
==============================

*okama* is a Python package developed for asset allocation and investment portfolio optimization tasks according to Modern Portfolio Theory (MPT).

The package is supplied with **free** «end of day» historical stock markets data and macroeconomic indicators through API.

    ...entities should not be multiplied without necessity

    *-- William of Ockham (c. 1287–1347)*

Installation
------------

Okama can be installed from _PyPI:https://pypi.org/project/okama/ :

.. code:: text

    pip install okama

The latest development version can be installed directly from GitHub:

.. code:: text

    pip install git+https://github.com/mbk-dev/okama@dev

.. warning::

    The development version of *okama* can have technical and financial issues. Please use carefully at your own risk.

Class Overview
--------------


.. autosummary::
    :toctree: stubs

    okama.Asset
    okama.AssetList
    okama.Portfolio


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
    :maxdepth: 2
    :caption: Content

    source/test
