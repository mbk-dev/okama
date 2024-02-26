import matplotlib.pyplot as plt

import okama as ok

# Portfolio WithDrawls
weights = [0.32, 0.31, 0.18, 0.19]
portf = ok.Portfolio(
    ["RGBITR.INDX", "RUCBTRNS.INDX", "MCFTR.INDX", "GC.COMM"],
    ccy="RUB",
    weights=weights,
    inflation=True,
    symbol="retirement_portf.PF",
    rebalancing_period="year",
    cashflow=-200_000,
    initial_amount=39_000_000,
    discount_rate=None,
)

print(portf.discount_rate)
print(portf)
print(f"{portf.get_cagr()}")
print(f"{portf.dcf.initial_amount_pv=}, {portf.dcf.cashflow_pv=}")
print(f"{portf.dcf.survival_period=}")
print(f"{portf.dcf.survival_date=}")
portf.dcf.wealth_index.plot()

portf.dcf.plot_forecast_monte_carlo(distr="norm", years=30, backtest=True, n=100)

s_periods = portf.dcf.monte_carlo_survival_period(distr="lognorm", years=25, n=10)
print(f"median {s_periods.quantile(50 / 100)}")
print(f"1st percentile {s_periods.quantile(1 / 100)}")
print(f"99th percentile {s_periods.quantile(99 / 100)}")
print(f"min {s_periods.min()}")
print(f"max {s_periods.mean()}")

# Rolling / Expanding Risk

al = ok.AssetList(["DJI.INDX", "BND.US"], inflation=True)
print(al.describe())
al.get_rolling_risk_annual(window=12 * 20).plot()

plt.show()
