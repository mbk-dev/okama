import okama as ok

pf = ok.Portfolio(['SPY.US', 'AGG.US'], weights=[.7, .3])

print(pf.diversification_ratio)
