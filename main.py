import matplotlib.pyplot as plt
import okama as ok

indexes = ["RGBITR.INDX", "MCFTR.INDX", "GC.COMM"]
ef = ok.EfficientFrontier(indexes, ccy="RUB", full_frontier=True, inflation=False, n_points=50)
ef.plot_cml(rf_return=0.15, y_axe="cagr")
plt.show()
