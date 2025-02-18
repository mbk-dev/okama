import warnings

import matplotlib.pyplot as plt
import pandas as pd
import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

portfolio = ['RUCBTRNS.INDX', 'RGBITR.INDX', 'MCFTR.INDX', 'GC.COMM' ] # Список активов
weights = [.22, .09, .45, .24]

pf = ok.Portfolio(
    portfolio,
    weights=weights,
    ccy='RUB',
    rebalancing_period='year',
    last_date = '2024-12'
)

print(pf.dividend_yield_annual)


# assets = ['MCFTR.INDX', 'RGBITR.INDX']
# weights = [0.60, 0.40]
# pf = ok.Portfolio(assets, weights=weights, ccy='RUB', rebalancing_period='year', inflation=False)
#
# ind = ok.IndexationStrategy(pf)
# ind.initial_investment = 4_000  # the initial investments size
# ind.amount = -10  # set withdrawal/contribution size
# ind.frequency = "month"  # set cash flow frequency TODO: add parameter
# ind.indexation = None  # set indexation size
# pf.dcf.cashflow_parameters = ind
#
# pf.dcf.set_mc_parameters(
#     distribution="norm",
#     period=2,
#     number=100
# )
#
# df = pf.dcf.monte_carlo_survival_period()
# print(df)







# ror = pf.assets_ror

# x = ok.Rebalance(
#     period='none', abs_deviation=0.10, rel_deviation=None
# )
#
#
# (x.wealth_ts(target_weights=pf.weights, ror=ror)[1] * 100).plot(figsize=[14, 8])
# plt.legend(["Индекс Мосбиржи", "Индекс ОФЗ"])
# plt.show()
