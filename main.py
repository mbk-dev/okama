import okama as ok

x = ok.Portfolio(['SPY.US', 'BND.US'], weights=[0.5, 0.5], ccy='RUB')

print(x.get_sharpe_ratio(rf_return=0.07))
