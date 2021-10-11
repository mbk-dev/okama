import okama as ok

al = ok.AssetList(['VOO.US', 'BND.US'])
print(al.get_sharpe_ratio(rf_return=0.02))
