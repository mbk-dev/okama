import okama as ok
import matplotlib.pyplot as plt

pf = ok.Portfolio(['SP500TR.INDX', 'MCFTR.INDX'], ccy='USD')
pf.rebalancing_period = 'year'
pf.wealth_index_with_assets.plot()
plt.show()

