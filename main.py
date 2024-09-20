import warnings

import matplotlib.pyplot as plt
import pandas as pd
import okama as ok

import os
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

pf = ok.Portfolio(
    assets=["MCFTR.INDX", "RUCBTRNS.INDX"],
    weights=[.3, .7],
    inflation=True,
    # first_date="2014-01",
    ccy="RUB",
    rebalancing_period="year",
)

# Fixed Percentage strategy
pc = ok.PercentageStrategy(pf)
pc.frequency = "year"
pc.percentage = -0.08

# Fixed Amount strategy
ind = ok.IndexationStrategy(pf)
ind.initial_investment = 100
ind.frequency = "year"
ind.amount = -0.5 * 12
ind.indexation = "inflation"

# TimeSeries strategy
d = {
    "2025-02": 1_000,
    "2029-03": -2_000,
}

ts = ok.TimeSeriesStrategy(pf)
ts.initial_investment = 1_000
ts.time_series_dic = d

# Assign a strategy
pf.dcf.cashflow_parameters = ind
pf.dcf.discount_rate = 0.10
pf.dcf.use_discounted_values = False

# df = pf.dcf.wealth_index

# Set Monte Carlo
pf.dcf.set_mc_parameters(
    distribution="t",
    period=50,
    number=100
)

# w = pf.dcf.find_the_largest_withdrawals_size(
#     min_amount=-100_000,
#     max_amount=-5_000,
#     withdrawal_steps=10,
#     confidence_level=0.50,
#     goal="survival_period",  # survival_period maintain_balance
#     target_survival_period=25
# )
# print(w)


df = pf.dcf.wealth_with_assets
df.plot()
# print("portfolio balance \n", df.iloc[-1, :].describe())

# pf.dcf.plot_forecast_monte_carlo(backtest=True)



# s = pf.dcf.monte_carlo_survival_period(threshold=.05)
# print("survival period \n", s.describe())


#
# print(pf.dcf.distribution)
# print(pf.dcf.mc_number)
# s = pf.dcf.monte_carlo_wealth
# print(s.iloc[-1].quantile(50/100))

# plt.figure(figsize=(20, 12))
plt.yscale("log")  # log or linear
# df.plot()
# plt.savefig('time_series.png')
plt.show()
