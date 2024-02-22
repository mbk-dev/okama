import matplotlib.pyplot as plt

import okama as ok
# Portfolio WithDrawls
# weights = [0.32, 0.31,  0.18, .19]
# portf = ok.Portfolio(['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM'],
#                      ccy="RUB",
#                      weights=weights,
#                      inflation=False,
#                      symbol="retirement_portf.PF",
#                      rebalancing_period='year',
#                      cashflow=-150_000,
#                      initial_amount=39_000_000,
#                      discount_rate=0.01
#                      )

# print(portf.discount_rate)
# print(portf)
# print(f"{portf.get_cagr()}")
# print(f"{portf.initial_amount_pv=}, {portf.cashflow_pv=}")
# print(f"{portf.survival_period=}")
# print(f"{portf.survival_date=}")
# portf.wealth_index.plot()
#
# portf.plot_forecast_monte_carlo(distr="norm", years=30, backtest=True, n=100)

# s_periods = portf.monte_carlo_survival_period(distr="lognorm", years=25, n=10)
# print(f"медиана {s_periods.quantile(50 / 100)}")
# print(f"первый порцентиль {s_periods.quantile(1 / 100)}")
# print(f"99й порцентиль {s_periods.quantile(99 / 100)}")
# print(f"минимум {s_periods.min()}")
# print(f"среднее {s_periods.mean()}")

# plt.show()

# Rolling / Expanding Risk
# al = ok.AssetList(['DJI.INDX',
#                    'BND.US'
#                    ])
# print(al)
# al.get_rolling_risk_annual(window=12*20).plot()
# # al.get_rolling_cagr(window=12*20, real=True).plot()
#
# plt.show()

# pf = ok.Portfolio(['SPY.US',
#                    'BND.US'
#                    ])
rf3 = ok.Portfolio(
    ["BND.US", "VTI.US", "VXUS.US"],
    weights=[0.40, 0.40, 0.20],
    rebalancing_period="year",
)
print(rf3.recovery_period)
