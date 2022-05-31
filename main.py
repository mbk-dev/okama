import okama as ok

rate = ok.Rate(symbol="RUONIA.RATE", first_date="2015-01", last_date="2020-02")
print(rate.values_daily)
