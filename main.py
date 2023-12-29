import matplotlib.pyplot as plt
import okama as ok

indexes = ["RGBITR.INDX", "RUCBTRNS.INDX", "MCFTR.INDX", "GC.COMM"]
bounds = [(0, 0.7), (0, 0.7), (0, 1), (0, 0.19)]
ef = ok.EfficientFrontier(indexes, ccy="RUB", full_frontier=False, bounds=bounds, inflation=False)
print(ef.get_tangency_portfolio(rf_return=0.08))
## TODO: replace ValueError: Weights sum is not equal to one. with custom error mesage

