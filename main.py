import warnings

import matplotlib.pyplot as plt
import pandas as pd
import okama as ok

import os


os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)


# ef = ok.EfficientFrontierReb(
#     ['SPY.US', 'AGG.US', 'GLD.US'],
#     rebalancing_strategy=ok.Rebalance(period='year'),
#     ccy='USD',
#     first_date='2020-01', last_date='2025-03', full_frontier=True, verbose=True)
# glob = ef.global_max_return_portfolio
# cagr = glob['CAGR']
# print(cagr)
#
# pf = ef.minimize_risk(cagr)
# print(pf)

# w = ef.minimize_risk(0.184914755913651)
# print(w)



# Plot the Efficient Fronrier
# ax.plot(df_reb_year.Risk, df_reb_year.CAGR, label="Annually rebalanced")
# ax.plot(df_not_reb.Risk, df_not_reb.CAGR, label="Not rebalanced")

# Plot the aseets
# ef.plot_assets(kind="cagr")

# plt.show()

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

pf2 = ok.Portfolio(first_date="2015-01", last_date="2024-10", rebalancing_strategy=ok.Rebalance(period="year"))
d = {
    "2018-02": 2_000,  # contribution
    "2024-03": -4_000  # withdrawal
}
ts = ok.TimeSeriesStrategy(pf2)
ts.time_series_dic = d  # use the dictionary to set cash flow
ts.initial_investment = 1_000  # add initial investments size (optional)
pf2.dcf.cashflow_parameters = ts
print(pf2.dcf.wealth_index)
# assets = ['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM']
# weights = [0.16,          0.40,             0.25,        0.19]
# # assets = ['RGBITR.INDX', 'MCFTR.INDX']
# # weights = [0.50,  0.50]
# pf = ok.Portfolio(assets, weights=weights, ccy='RUB', inflation=False)
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
# pc.percentage = -0.12
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
#     tolerance_rel=0.10,
#     threshold=0.05,
#     iter_max=20
# )
#
# print(
#     solution
# )

