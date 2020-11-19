import okama as ok

weights = [0.3, 0.2, 0.2, 0.2, 0.1]
assets = ['T.US', 'XOM.US', 'JNJ.US', 'SBERP.MOEX', 'LKOH.MOEX']
y = ok.Portfolio(assets, weights=weights, curr='RUB')
print(y.dividend_yield)
