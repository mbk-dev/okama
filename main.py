import okama as ok

weights = [.18, 0.12, 0.38, 0.12, 0.15, 0.05]
sum(weights)

portf = ok.Portfolio(['TIP.US', 'LQD.US', 'SPY.US', 'PGJ.US', 'GLD.US', 'REIT.INDX'],
                     ccy='USD',
                     weights=weights,
                     inflation=True,
                     symbol="portf.PF")
print(portf.percentile_inverse_cagr(distr='hist', years=1, score=0))
