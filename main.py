import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

import okama.portfolios.cashflow_strategies

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

rs = ok.Rebalance(
    period="year",
    abs_deviation=0.10,
    rel_deviation=0.40
)
weights = [0.12, 0.21, 0.42, 0.25]
pf = ok.Portfolio(
    ['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM'],
    weights=weights,
    first_date='2014-06',
    ccy="RUB",
    inflation=True,
    rebalancing_strategy=rs,
    symbol="My_portfolio.PF",
)
pf.dcf.discount_rate = 0.09
# Percentage CF strategy
cf_strategy = ok.PercentageStrategy(pf)  # create PercentageStrategy linked to the portfolio

cf_strategy.initial_investment = 83_000_000  # initial investments size
cf_strategy.frequency = "year"  # withdrawals frequency
cf_strategy.percentage = -0.40

# # Indexation CF strategy
# cf_strategy = ok.IndexationStrategy(pf)
#
# cf_strategy.initial_investment = 83_000_000
# cf_strategy.frequency = "year"
# cf_strategy.amount = 1_500_000 * 12
# cf_strategy.indexation = 0.09

d = {
    "2015-06": -35_000_000,
}

cf_strategy.time_series_dic = d
cf_strategy.time_series_discounted_values = False

pf.dcf.cashflow_parameters = cf_strategy  # assign the cash flow strategy to portfolio

pf.dcf.set_mc_parameters(
    distribution="norm",
    period=15,
    number=100
)

wi = pf.dcf.wealth_index_fv
# cf = pf.dcf.cash_flow_ts_pv.resample("Y").sum()


wi.plot(legend=False)
# cf.plot(kind="bar", legend=False)
plt.yscale('linear')  # linear or log
plt.show()

# df = pf.dcf.monte_carlo_wealth_fv
# print(df)


# sp = pf.dcf.monte_carlo_survival_period()
# print(sp.quantile(25 / 100), " years")

# mc_wealth_pv = pf.dcf.monte_carlo_wealth_pv.iloc[-1].describe([.05, .10, .20, .50])
# print(f"{mc_wealth_pv=}")

# print(pf.dcf.wealth_index.iloc[-1, :])


