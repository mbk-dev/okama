import warnings

import matplotlib.pyplot as plt
import pandas as pd
import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

ef = ok.EfficientFrontierReb(['SPY.US', 'MCFTR.INDX'], ccy='RUB', last_date='2025-03', bounds=None, full_frontier=True)
df1_reb_year = ef.ef_points

fig, ax = plt.subplots(figsize=(12, 10))

# Plot the Efficient Frontiers
ax.plot(df1_reb_year.Risk, df1_reb_year.CAGR, label="Annually rebalanced")
# ax.plot(df1_not_reb.Risk, df1_not_reb.CAGR, label="Not rebalanced")

# Plot the assets
ef.plot_assets(kind="cagr")
plt.show()


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


# assets = ['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM']
# weights = [0.16, 0.40,  0.25, 0.19]
# pf = ok.Portfolio(assets, weights=weights, ccy='RUB', rebalancing_period='year', inflation=False)
# pf.dcf.discount_rate = 0.09
#
# ind = ok.IndexationStrategy(pf)
# ind.initial_investment = 2_000_000 * 90
# ind.amount = -4000 * 90
# ind.frequency = "month"
# ind.indexation = 0.09
#
# pf.dcf.cashflow_parameters = ind
#
# pf.dcf.set_mc_parameters(
#     distribution="norm",
#     period=50,
#     number=1000
# )

# df = pf.dcf.monte_carlo_survival_period()
# print(df.describe())

# w, err = pf.dcf.find_the_largest_withdrawals_size(
#     goal="maintain_balance_pv",
#     percentile=20,
#     tolerance_rel=0.01,
# )
#
# print(w, err)



# ror = pf.assets_ror

# x = ok.Rebalance(
#     period='none', abs_deviation=0.10, rel_deviation=None
# )
#
#
# (x.wealth_ts(target_weights=pf.weights, ror=ror)[1] * 100).plot(figsize=[14, 8])
# plt.legend(["Индекс Мосбиржи", "Индекс ОФЗ"])
# plt.show()
