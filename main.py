import matplotlib.pyplot as plt

import okama as ok

weights = [0.30, 0.20, 0.50]
portf = ok.Portfolio(
    ["SPY.US", "BND.US", "GLD.US"],
    ccy="RUB",
    weights=weights,
    inflation=False,
    rebalancing_period="year",
    first_date="2015-01",
    last_date="2020-12"
)

ret = portf.get_cumulative_return()
portf.wealth_index_with_assets.plot()
print(ret)
print(portf.ror.iloc[0])
print(portf.assets_ror.iloc[0])



# ## Portfolio WithDrawls
# weights = [0.32, 0.31, 0.18, 0.19]
# portf = ok.Portfolio(
#     ["RGBITR.INDX", "RUCBTRNS.INDX", "MCFTR.INDX", "GC.COMM"],
#     ccy="RUB",
#     weights=weights,
#     inflation=True,
#     symbol="retirement_portf.PF",
#     rebalancing_period="year",
#     cashflow=-200_000,
#     initial_amount=44_000_000,
#     discount_rate=None,
# )
#
# print(portf.discount_rate)
# print(portf)
# print(f"{portf.get_cagr()}")
# print(f"{portf.dcf.initial_amount_pv=}, {portf.dcf.cashflow_pv=}")
# print(f"{portf.dcf.survival_period=}")
# print(f"{portf.dcf.survival_date=}")
# portf.dcf.wealth_index.plot()
#
# portf.dcf.plot_forecast_monte_carlo(distr="norm", years=30, backtest=True, n=100)
#
# s_periods = portf.dcf.monte_carlo_survival_period(distr="lognorm", years=25, n=100)
# print(f"median {s_periods.quantile(50 / 100)}")
# print(f"1st percentile {s_periods.quantile(1 / 100)}")
# print(f"99th percentile {s_periods.quantile(99 / 100)}")
# print(f"min {s_periods.min()}")
# print(f"max {s_periods.max()}")
# print(s_periods.describe(percentiles=[0.01, 0.5, 0.99]))
# print(s_periods.mode())

# Rolling / Expanding Risk

# al = ok.AssetList(["DJI.INDX", "BND.US"], inflation=True)
# print(al.describe())
# al.get_rolling_risk_annual(window=12 * 20).plot()

plt.show()
