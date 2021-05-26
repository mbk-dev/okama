import okama as ok
x = ok.AssetList(['T.US'], ccy='RUB')
condition = x._get_asset_dividends('T.US').values != 0
print(x._get_asset_dividends('T.US')[condition])
