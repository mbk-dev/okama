import okama as ok

portf = ok.Portfolio(["SBER.MOEX", "T.US", "GNS.LSE"],
                     first_date="2015-01",
                     last_date="2020-01",
                     ccy="RUB",
                     rebalancing_period='year'
                     )
print(portf.dividends[-1])
