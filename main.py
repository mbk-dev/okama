import okama as ok

funds1 = ['MCFTR.INDX', '0177-71671092.PIF']
curr = 'RUB'

x1 = ok.AssetList(funds1, ccy=curr)

# r = x1.get_rolling_cagr(window=12*3)
print(x1.tracking_difference(rolling_window=12*3))
