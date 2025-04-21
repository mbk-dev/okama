import warnings

import matplotlib.pyplot as plt
import pandas as pd
import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

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


# assets = ['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM']
# weights = [0.16, 0.40,  0.25, 0.19]
assets = ['SPY.US', 'AGG.US']
weights = [0.60,  0.40]
pf = ok.Portfolio(assets, weights=weights, ccy='USD', rebalancing_period='year', inflation=False)
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


ror = pf.assets_ror

x = ok.Rebalance(
    period='none', abs_deviation=0.10, rel_deviation=0.05
)


(x.wealth_ts(target_weights=pf.weights, ror=ror) * 100).plot(figsize=[14, 8])
# plt.legend(["Индекс Мосбиржи", "Индекс ОФЗ"])
plt.show()
