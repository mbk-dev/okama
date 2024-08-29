import warnings

import matplotlib.pyplot as plt
import okama as ok

import os
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.simplefilter(action='ignore', category=FutureWarning)

a = ok.Portfolio(["SPY.US"])
b = ok.Portfolio(["GC.COMM"])
c = ok.Portfolio(["RGBITR.INDX"])

x = ok.AssetList([a, b, c])

y = ok.EfficientFrontier(assets=["SPY.US", "GC.COMM", c], ccy='RUB', n_points=100)


y.plot_pair_ef()

plt.show()

# pf = ok.Portfolio(
#     assets=['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM'],
#     weights=[0.10, 0.51,  0.19, .20],
#     inflation=True,
#     ccy="RUB",
#     rebalancing_period="year",
# )
# # pf.dcf.cashflow_parameters.initial_investment = 10_300_000
#
# # Set cashflow
# pf.dcf.set_cashflow_parameters(
#     initial_investment=10_300_000,
#     method="fixed_amount",
#     frequency="month",
#     amount=-50_000,
#     indexation=pf.dcf.discount_rate
# )
#
# # Set Monte Carlo
# pf.dcf.mc.period = 50
# pf.dcf.mc.number = 100
# pf.dcf.mc.distribution = "t"
#
# w = pf.dcf.find_the_largest_withdrawals_size(
#     min_amount=-100_000,
#     max_amount=-5_000,
#     withdrawal_steps=10,
#     confidence_level=0.50,
#     goal="survival_period",  # survival_period maintain_balance
#     target_survival_period=25
# )
# print(w)


# df = pf.dcf.monte_carlo_wealth

# pf.dcf.plot_forecast_monte_carlo(backtest=True)

# plt.show()
# pf.dcf.set_mc_parameters(
#     distribution="t",
#     period=10,
#     mc_number=10
# )
#
# print(pf.dcf.distribution)
# print(pf.dcf.mc_number)
# s = pf.dcf.monte_carlo_wealth
# print(s.iloc[-1].quantile(50/100))
