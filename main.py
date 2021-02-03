import okama as ok

ls = ['AGG.US', 'GLD.US', 'SPY.US']
y = ok.EfficientFrontierReb(symbols=ls, n_points=2, verbose=True)
ef = y.ef_points
print(y.ef_points)

