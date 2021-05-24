import okama as ok
pf1 = ok.Portfolio(['GLD.US', 'MSFT.US'], weights=[.1, .9], ccy='RUB', symbol='pf1.PF')
pf2 = ok.Portfolio(['GLD.US', 'MSFT.US'], weights=[.5, .5], ccy='RUB')
x = ok.AssetList([pf1, pf2, 'SPY.US'])
print(x.__len__())
