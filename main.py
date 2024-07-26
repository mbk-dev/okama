import warnings

import okama as ok

warnings.filterwarnings("ignore")

pf = ok.Portfolio(
    assets=["MCFTR.INDX", "AGG.US", "GLD.US"],
    weights=[0.60, 0.35, 0.05],
    ccy="RUB",
    rebalancing_period="year",
    initial_amount=300_000,
    cashflow=-2_000,
)

pf.dcf.set_mc_parameters(
    distribution="t",
    period=10,
    mc_number=100
)

print(pf.dcf.distribution)
print(pf.dcf.mc_number)
s = pf.dcf.monte_carlo_wealth
print(s.iloc[-1].quantile(50/100))
