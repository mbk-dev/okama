import okama as ok

tk = ['FXGD.MOEX', 'FXRD.MOEX']
al = ok.AssetList(assets=tk, ccy='USD', inflation=True, last_date='2021-10')

print(al.wealth_indexes)
