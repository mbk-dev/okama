import matplotlib.pyplot as plt
import okama as ok

weights = [0.32, 0.31,  0.18, .19]
portf = ok.Portfolio(['RGBITR.INDX', 'RUCBTRNS.INDX', 'MCFTR.INDX', 'GC.COMM'], ccy="RUB", weights=weights, inflation=True, symbol="portf.PF", rebalancing_period='year')
portf.describe()
