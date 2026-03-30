import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

ls = ["SPY.US", "GLD.US"]
# ls = ['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM', 'RUS_PR.RE']

rb = ok.Rebalance(period='year')

x = ok.EfficientFrontier(
    assets=ls,
    ccy="RUB",
    last_date="2022-01",
    inflation=False,
    rebalancing_strategy=rb,
    n_points=80,
    verbose=True,
    full_frontier=True
)

df = x.ef_points
print(df)

fig, ax = plt.subplots(figsize=(12, 10))

# Plot the Efficient Frontiers
ax.plot(df.Risk, df.CAGR, label="Граница эффективности с учетом ребалансировки")

# Plot the assets
x.plot_assets(kind="cagr")

# Set labels
ax.set_xlabel("Risk (Standard Deviation)")
ax.set_ylabel("CAGR")
ax.legend()
plt.show()

