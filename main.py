import okama as ok

asset_list = ok.AssetList(assets=['RUB.FX', 'MCFTR.INDX'], ccy='RUB', first_date='2019-01', last_date='2020-01', inflation=True)
print(asset_list.get_cumulative_return(period='ytd', real=True))
