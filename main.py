import okama as ok

ls4 = ['SPY.US', 'BND.US', 'GLD.US', 'VNQ.US']
curr = 'USD'
four_assets = ok.EfficientFrontier(symbols=ls4, curr=curr, n_points=100)
ok.Plots(ls4, curr=curr).plot_pair_ef()
