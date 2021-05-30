import okama as ok
x = ok.Portfolio(['SBERP.MOEX'], ccy='RUB', inflation=True)
print(x.assets_dividend_yield)
