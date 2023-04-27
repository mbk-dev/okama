import okama as ok

al = ok.AssetList(['SPY.US', ok.Portfolio()])

for a in al:
    print(a.symbol)
