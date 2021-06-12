import okama as ok
x = ok.Portfolio(['SPY.US', 'AGG.US'], weights=[0.2, 0.8], inflation=False)
print(x.get_rolling_cumulative_return())
