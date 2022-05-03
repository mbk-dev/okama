import okama as ok

x = ok.AssetList(['SPY.US'], ccy='USD')
print(x.get_cumulative_return(period='YTD'))
