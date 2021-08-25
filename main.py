import okama as ok

ccy = 'RUB'
ls = ['RGBITR.INDX', 'MCFTR.INDX', 'SPY.US', 'PGJ.US', 'GLD.US']
cons_weights=[0.65, 0.06, 0.19, 0.05, 0.05]
port_cons_no_infl = ok.Portfolio(ls, ccy=ccy, weights=cons_weights, inflation=False)
print(port_cons_no_infl.percentile_from_history(years=8))
