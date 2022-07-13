import okama as ok

x = ok.AssetList(["SPY.US", "BND.US", "GLD.US"], inflation=False)
print(x.assets_last_dates)
