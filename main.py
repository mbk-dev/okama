import matplotlib.pyplot as plt

import okama as ok

weights = [0.32, 0.31,  0.18, .19]
portf = ok.Portfolio(['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM'],
                     ccy="RUB",
                     weights=weights,
                     inflation=True,
                     symbol="retirement_portf.PF",
                     rebalancing_period='year',
                     cashflow=-200_000,
                     initial_amount=39_000_000
                     )
# print(portf)
# print(f"{portf.get_cagr()}")
# print(f"{portf.initial_amount_pv=}, {portf.cashflow_pv=}")
# print(f"{portf.survival_period=}")
# print(f"{portf.survival_date=}")
# portf.wealth_index.plot()

portf.plot_forecast_monte_carlo(distr="norm", years=10, backtest=True, n=2)

plt.show()
