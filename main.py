import okama as ok

funds1 = ['MCFTR.INDX', '0177-71671092.PIF', '0890-94127385.PIF']
curr = 'RUB'
x1 = ok.AssetList(funds1, curr=curr)
x1.tracking_difference_annualized
