import okama as ok

x = ok.AssetList(assets=['NSEI.INDX'])
print(x.first_date)
print(x.period_length)

y = ok.Asset('INRUSD.FX')
print(y.first_date)
