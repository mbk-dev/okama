import okama as ok
pf = ok.Portfolio(['T.US', 'XOM.US'], weights=[0.8, 0.2], first_date='2010-01', last_date='2021-01', ccy='USD')
print(pf.dividend_yield)
