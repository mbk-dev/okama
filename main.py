import numpy as np
import okama as ok

asset_list = ok.AssetList(assets=['MSFT.US'], ccy='USD')

print(asset_list.dividends_annual)


x = asset_list.get_dividend_mean_growth_rate(period=20)
# x.replace([np.inf, -np.inf], 0, inplace=True)

print(f'Growth rate:', x)
