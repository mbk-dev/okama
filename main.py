import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

rs = ok.Rebalance(
    period="year",
    # abs_deviation=0.10,
    # rel_deviation=0.40
)
weights = [0, 0, 1, 0]
pf = ok.Portfolio(
    ['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM'],
    weights=weights,
    # first_date='2014-06',
    ccy="RUB",
    inflation=True,
    rebalancing_strategy=rs,
    symbol="My_portfolio.PF",
)
pf.dcf.discount_rate = 0.09
# # Percentage CF strategy
# cf_strategy = ok.PercentageStrategy(pf)  # create PercentageStrategy linked to the portfolio
#
# cf_strategy.initial_investment = 83_000_000  # initial investments size
# cf_strategy.frequency = "year"  # withdrawals frequency
# cf_strategy.percentage = -0.09

# # Indexation CF strategy
# cf_strategy = ok.IndexationStrategy(pf)
#
# cf_strategy.initial_investment = 10_000_000
# cf_strategy.frequency = "year"
# cf_strategy.amount = 10_000_000 * 0.05
# cf_strategy.indexation = 0.09

# Cut Whithdrawals if Drawdown CWID strategy
cf_strategy = ok.CutWithdrawalsIfDrawdown(pf)

cf_strategy.initial_investment = 10_000_000
cf_strategy.frequency = "year"
cf_strategy.amount = -10_000_000 * 0.05
cf_strategy.indexation = 0.09
cf_strategy.crash_threshold_reduction = [
    (.10, .20),
    (.20, .50),
    (.40, 1),
]

# d = {
#     "2015-06": -35_000_000,
# }
#
# cf_strategy.time_series_dic = d
# cf_strategy.time_series_discounted_values = False

# # VDS strategy
# cf_strategy = ok.VanguardDynamicSpending(pf)
# cf_strategy.initial_investment = 1_000_000
# cf_strategy.percentage = -0.08
# cf_strategy.indexation = 0.09
# # cf_strategy.min_max_annual_withdrawal = 10_000_000 / 5,  10_000_000 / 10 # 20%, 10%
# cf_strategy.floor_ceiling = -0.10, 0.20
# # cf_strategy.time_series_dic = d
# # cf_strategy.time_series_discounted_values = False

pf.dcf.cashflow_parameters = cf_strategy  # assign the cash flow strategy to portfolio

# w = cf_strategy.calculate_withdrawal_size(
#     last_withdrawal=0,
#     balance=cf_strategy.initial_investment,
#     number_of_periods=1
# )

# print(pf.dcf.cashflow_parameters)
# # pf.dcf.set_mc_parameters(
# #     distribution="norm",
# #     period=15,
# #     number=100
# # )
# print(pf.dcf.cashflow_parameters._crash_threshold_reduction_series)

# wi = pf.dcf.wealth_index(discounting="pv", include_negative_values=False)
cf = pf.dcf.cash_flow_ts(discounting="pv", remove_if_wealth_index_negative=True).resample("Y").sum()
# wi = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=False)
# cf = pf.dcf.monte_carlo_cash_flow(discounting="pv", remove_if_wealth_index_negative=True)
# print(cf)
# print(cf.pct_change())
# wi.plot(
#     # kind="bar",
#     legend=False
# )
# plt.yscale('linear')  # linear or log
# plt.show()
#
# df = cf[0]
cf.plot(
    kind="bar",
    legend=False
)
plt.yscale('linear')  # linear or log
plt.show()

# print(df[df != 0])

# df = pf.dcf.monte_carlo_wealth_fv
# print(df)


# sp = pf.dcf.monte_carlo_survival_period()
# print(sp.quantile(25 / 100), " years")

# mc_wealth_pv = pf.dcf.monte_carlo_wealth_pv.iloc[-1].describe([.05, .10, .20, .50])
# print(f"{mc_wealth_pv=}")

# print(pf.dcf.wealth_index.iloc[-1, :])


