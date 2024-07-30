import warnings

import matplotlib.pyplot as plt
import okama as ok

warnings.filterwarnings("ignore")

pf = ok.Portfolio(
    assets=["MCFTR.INDX", "AGG.US", "GLD.US"],
    weights=[0.60, 0.35, 0.05],
    inflation=True,
    ccy="RUB",
    rebalancing_period="year",
    initial_amount=300_000,
    cashflow=-2_000,
)

# Set cashflow
pf.dcf.set_cashflow_parameters(
    method="fixed_amount",
    frequency="quarter",
    amount=-2_000 * 3,
    indexation=pf.discount_rate
)

# Set Monte Carlo
pf.dcf.mc.period = 5
pf.dcf.mc.number = 200
pf.dcf.mc.distribution = "norm"


df = pf.dcf.monte_carlo_wealth

pf.dcf.plot_forecast_monte_carlo(backtest=True)

plt.show()
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
