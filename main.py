import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os


os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

pf = ok.Portfolio(
    ["SPY.US", "AGG.US", "GLD.US"],
    weights=[0.60, 0.35, 0.05],
    ccy="USD",
    inflation=True,
    last_date="2024-01",
    rebalancing_strategy=ok.Rebalance(period="year"),
    symbol="My_portfolio.PF",
)

pf.dcf.use_discounted_values = True
# # Percentage CF strategy
# cf_strategy = ok.PercentageStrategy(pf)  # create PercentageStrategy linked to the portfolio
#
# cf_strategy.initial_investment = 1_000  # initial investments size
# cf_strategy.frequency = "year"  # withdrawals frequency
# cf_strategy.percentage = -0.12

# Indexation CF strategy
cf_strategy = ok.IndexationStrategy(pf)
cf_strategy.initial_investment = 10_000_000
cf_strategy.amount = -12_000 * 12
cf_strategy.frequency = "year"

d = {
    "2026-02": 10_000_000,
    "2029-03": -20_000_000,
}

cf_strategy.time_series_dic = d

pf.dcf.cashflow_parameters = cf_strategy  # assign the cash flow strategy to portfolio

pf.dcf.set_mc_parameters(distribution="norm", period=30, number=400)  # simulation period in years

df = pf.dcf.monte_carlo_wealth_fv

df.plot(legend=False)
plt.yscale('log')
plt.show()

sp = pf.dcf.monte_carlo_survival_period()
print(sp.quantile(25 / 100), " years")

wealth_pv = pf.dcf.monte_carlo_wealth_pv.iloc[-1].describe()
wealth_fv = pf.dcf.monte_carlo_wealth_fv.iloc[-1].describe()

print(f"{wealth_pv=}", f"{wealth_fv=}")


