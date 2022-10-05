import okama as ok

x = ok.EfficientFrontier(['SPY.US', 'BND.US'])
print(x.get_monte_carlo(10))
