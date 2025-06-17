import warnings

import matplotlib.pyplot as plt
import pandas as pd
import okama as ok

import os


os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)
ef = ok.EfficientFrontierReb(
    ['SPY.US', 'AGG.US', 'GLD.US'],
    rebalancing_strategy=ok.Rebalance(period='year'),
    ccy='RUB',
    first_date='2020-01', last_date='2025-03', full_frontier=True, verbose=True)
# w = ef.minimize_risk(0.184914755913651)
# print(w)
ef.rebalancing_strategy.period = "month"
df_reb_year = ef.ef_points
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the Efficient Fronrier
ax.plot(df_reb_year.Risk, df_reb_year.CAGR, label="Annually rebalanced")
# ax.plot(df_not_reb.Risk, df_not_reb.CAGR, label="Not rebalanced")

# Plot the aseets
ef.plot_assets(kind="cagr")

plt.show()

# al = ok.AssetList(['MCFTR.INDX'], ccy='RUB', inflation=False, first_date='2025-01')
# wealth = al.wealth_indexes
# cum_return = al.get_cumulative_return()
#
# print(f'{wealth.iloc[-1]=}')
# print(f'{cum_return=}')


# portfolio = ['RUCBTRNS.INDX', 'RGBITR.INDX', 'MCFTR.INDX', 'GC.COMM' ] # Список активов
# weights = [.22, .09, .45, .24]
#
# pf = ok.Portfolio(
#     portfolio,
#     weights=weights,
#     ccy='RUB',
#     rebalancing_period='year',
#     last_date = '2024-12'
# )
#
# print(pf.dividend_yield_annual)


# assets = ['RGBITR.INDX', 'MCFTR.INDX', 'GC.COMM']
# weights = [0.60,   0.35, 0.05]
# assets = ['RGBITR.INDX', 'MCFTR.INDX']
# weights = [0.50,  0.50]
# pf = ok.Portfolio(assets, weights=weights,
#                   first_date="2015-01",
#                   last_date="2020-01",
#                   ccy='RUB',
#                   inflation=True)
# pf.rebalancing_strategy = ok.Rebalance(
#     period="none",
#     abs_deviation=0.10,
#     rel_deviation=None
# )
# print(pf.rebalancing_strategy)
# # pf = ok.Portfolio(
# #     assets,
# #     weights=weights,
# #     ccy='USD',
# #     rebalancing_strategy=ok.Rebalance(period="none", abs_deviation=None, rel_deviation=0.050),
# #     inflation=False
# # )
# ev = pf.rebalancing_strategy.wealth_ts(
#     target_weights=weights,
#     ror=pf.assets_ror,
#     calculate_assets_wealth_indexes=True
# ).events
# print(ev)


# pf.dcf.discount_rate = 0.09

# ind = ok.IndexationStrategy(pf)
# ind.initial_investment = 2_000_000 * 90
# ind.amount = -4000 * 90
# ind.frequency = "month"
# ind.indexation = 0.09

# pf.dcf.cashflow_parameters = ind

# Fixed Percentage strategy
# pc = ok.PercentageStrategy(pf)
# pc.initial_investment = 10_000
# pc.frequency = "year"
# pc.percentage = -0.55
#
# pf.dcf.cashflow_parameters = pc
#
# pf.dcf.set_mc_parameters(
#     distribution="norm",
#     period=50,
#     number=100
# )
#
# solution = pf.dcf.find_the_largest_withdrawals_size(
#     goal="maintain_balance_pv",
#     percentile=20,
#     tolerance_rel=0.05,
#     threshold=0.05,
#     iter_max=20
# )
#
# print(
#     solution
# )

