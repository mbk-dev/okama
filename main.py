import okama as ok

ls2 = ['SPY.US', 'BND.US']
curr = 'USD'
two_assets = ok.EfficientFrontier(assets=ls2, ccy=curr, n_points=100)
print(two_assets)
