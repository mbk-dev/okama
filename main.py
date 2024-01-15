import matplotlib.pyplot as plt

import okama as ok

weights = [0.32, 0.31,  0.18, .19]
portf = ok.Portfolio(['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM'],
                     ccy="RUB",
                     weights=weights,
                     inflation=True,
                     symbol="retirement_portf.PF",
                     rebalancing_period='year',
                     cashflow=-400_000,
                     initial_amount=39_000_000
                     )
# print(portf)
# print(f"{portf.get_cagr()}")
# print(f"{portf.initial_amount_pv=}, {portf.cashflow_pv=}")
# print(f"{portf.survival_period=}")
# print(f"{portf.survival_date=}")
# portf.wealth_index.plot()

# portf.plot_forecast_monte_carlo(distr="norm", years=30, backtest=False, n=1000)

# plt.show()

s_periods = portf.monte_carlo_survival_period(distr="lognorm", years=25, n=1000)
print(f"медиана {s_periods.quantile(50 / 100)}")
print(f"первый порцентиль {s_periods.quantile(1 / 100)}")
print(f"99й порцентиль {s_periods.quantile(99 / 100)}")
print(f"минимум {s_periods.min()}")
print(f"среднее {s_periods.mean()}")
