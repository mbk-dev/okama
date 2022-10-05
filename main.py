import okama as ok

x = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[0.1, 0.3, 0.6], rebalancing_period='year')
print(x.weights_ts.tail())
