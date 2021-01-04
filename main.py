import okama as ok
import pandas as pd

x = ok.Portfolio(symbols=['RUB.FX', 'MCFTR.INDX'], ccy='RUB',
                                      first_date='2015-01', last_date='2020-01', inflation=True)
print(x.describe())
print('\n\n*******************')
print(pd.read_pickle('tests/data/portfolio_description.pkl'))