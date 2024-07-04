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

portf.plot_percentiles_fit("t")

plt.show()
