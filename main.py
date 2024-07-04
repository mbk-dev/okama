import warnings

import matplotlib.pyplot as plt
import okama as ok

warnings.filterwarnings("ignore")

pf = ok.Portfolio(
    assets=['MCFTR.INDX', 'AGG.US', 'GLD.US'],
    weights=[.60, .35, .05],
    ccy='RUB',
    rebalancing_period='year',
    initial_amount=300_000,
    cashflow=-2_000
)


print(pf.describe())

