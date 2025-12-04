# import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
# warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

al = ["SP500TR.INDX", "VNQ.US", "GC.COMM", "USDEUR.FX"]

x = ok.AssetList(al, last_date="2025-11")  # first_date and last_date limits the Rate of Return time series

print(x.kstest(distr="lognorm"))

