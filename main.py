import warnings

import pandas as pd
import matplotlib.pyplot as plt

import okama as ok

import os

# os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
# warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

tickers = [
    "SPY.US",
    "AGG.US",
    "GC.COMM",
]  # we can create lists of assets and portfolio containing general type of assets and **indexes**
w = [0.7, 0.15, 0.15]
currency = "USD"

y = ok.Portfolio(tickers, ccy=currency, weights=w, inflation=True)

y.dcf.discount_rate = None
y.dcf.wealth_index(discounting="pv", include_negative_values=False).plot()

plt.show()