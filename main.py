import warnings

import matplotlib.pyplot as plt
import pandas as pd
import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

l = ["SPY.US", "GLD.US", "AGG.US"]
w=[0.20, 0.30, 0.50]

pf = ok.Portfolio(l, weights=w, ccy="EUR", rebalancing_period="none")



# # Fixed Percentage strategy
# pc = ok.PercentageStrategy(pf)
# pc.initial_investment = 10_000
# pc.frequency = "year"
# pc.percentage = -0.08

# Fixed Amount strategy
ind = ok.IndexationStrategy(pf)
ind.initial_investment = 10_000
ind.frequency = "year"
ind.amount = -500
ind.indexation = "inflation"

# # TimeSeries strategy
# d = {
#     "2025-02": 1_000,
#     "2029-03": -2_000,
# }
#
# ts = ok.TimeSeriesStrategy(pf)
# ts.initial_investment = 10_000
# ts.time_series_dic = d

# Assign a strategy
pf.dcf.cashflow_parameters = ind
pf.dcf.discount_rate = 0.10
pf.dcf.use_discounted_values = True

# Set Monte Carlo
pf.dcf.set_mc_parameters(distribution="t", period=50, number=100)

pf.dcf.plot_forecast_monte_carlo(backtest=True)

plt.yscale("log")  # log or linear
plt.legend("")
# plt.savefig('time_series.png')
plt.show()
