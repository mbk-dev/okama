import okama as ok

ls = ['MNG.LSE', 'GNS.LSE']
pf = ok.AssetList(ls)
print(pf.dividend_yield)
