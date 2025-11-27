# import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
# warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

ls_m = ["SPY.US", "GLD.US", "PGJ.US", "RGBITR.INDX", "MCFTR.INDX"]
curr_rub = "RUB"

# x = ok.EfficientFrontier(
#     assets=ls_m,
#     first_date="2005-01",
#     last_date="2020-11",
#     ccy=curr_rub,
#     # rebalancing_strategy=ok.Rebalance(period="year"),  # set rebalancing period to one year
#     n_points=20,
#     verbose=False,
# )

x = ok.EfficientFrontier(
    assets=ls_m,
    first_date="2005-01",
    last_date="2020-11",
    ccy=curr_rub,
    n_points=40,
    # rebalancing_strategy=ok.Rebalance(period="year"),  # set rebalancing period to one year
)

ef_points = x.ef_points

print(ef_points[["Mean return", "CAGR"]])

