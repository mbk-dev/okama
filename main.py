import okama as ok

ls = ['SPY.US', 'GLD.US']
curr = 'USD'
y = ok.EfficientFrontierReb(symbols=ls, last_date='2020-10', curr=curr, reb_period='Y', n_points=5)
b = y.ef_points
print(b)

