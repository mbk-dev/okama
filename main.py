# import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
# warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

ls = ["SPY.US", "BND.US", "GC.COMM", "EUR.FX"]
currency = "EUR"  # base currency

# x = ok.Inflation("EUR.INFL", first_date="2019-01-01", last_date="2020-01-01")

x = ok.AssetList(first_date="2019-01-01", last_date="2020-01-01")  # first_date and last_date limits the Rate of Return time series

print(x)

