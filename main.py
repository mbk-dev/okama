import okama as ok

al = ok.AssetList(['SPY.US', ok.Portfolio()])

print(al[0].symbol)

for a in al:
    print(a.symbol)
