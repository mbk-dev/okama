import okama as ok

x = ok.Portfolio(symbols=['SBER.MOEX', 'T.US'], ccy='RUB', last_date='2020-01', inflation=True)
# print(x.get_cagr('YTD'))
print(x.describe())

