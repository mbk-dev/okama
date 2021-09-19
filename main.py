import okama as ok

ls_m = ['SPY.US', 'GLD.US']
curr_rub = 'USD'
x = ok.EfficientFrontierReb(assets=ls_m)
monte_carlo = x.get_monte_carlo(n=100)  # it can take some time ...
print(monte_carlo)
