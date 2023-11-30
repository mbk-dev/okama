import matplotlib.pyplot as plt
import okama as ok

# indexes = ["RGBITR.INDX", "MCFTR.INDX", "GC.COMM"]
# ef = ok.EfficientFrontier(indexes, ccy="RUB", full_frontier=True, inflation=False, n_points=50)
# ef.plot_cml(rf_return=0.15, y_axe="cagr")
# plt.show()

pf = ok.Portfolio(["SPY.US", "BND.US"], weights=[.5, .5], rebalancing_period="monthly")
pf.wealth_index_with_assets.plot()
plt.show()
