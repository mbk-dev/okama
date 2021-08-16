import okama as ok

x = ok.Portfolio(['LKOH.MOEX', 'T.US'], ccy='USD', rebalancing_period='none')
y = ok.AssetList(['SPY.US', x])
print(y.assets_dividend_yield)
