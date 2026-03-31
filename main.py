import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

tickers = [
    "VNQ.US",
    "DBXD.XFRA",
    "MCFTR.INDX",
]  # we can create lists of assets and portfolio containing general type of assets and **indexes**
w = [0.5, 0.25, 0.25]
currency = "USD"

y = ok.Portfolio(tickers, ccy=currency, weights=w)

print(y)
