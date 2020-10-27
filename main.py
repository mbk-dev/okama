import okama as ok
ls = ['SPY.US', 'GLD.US']
x = ok.EfficientFrontierReb(symbols=ls)
range = x._target_risk_range

print(x.maximize_return_constraints_mysty(range[0]))
# print(x._maximize_return(range[0]))
