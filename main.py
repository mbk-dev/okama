import matplotlib.pyplot as plt
import okama as ok

al = ok.AssetList(['SP500TR.INDX', 'SPY.US', 'VOO.US'])

# al.tracking_error(rolling_window=12).plot()
al.index_beta(rolling_window=12).plot()
plt.show()
