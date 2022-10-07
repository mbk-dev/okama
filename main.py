import okama as ok

lsz = ["MCFTR.INDX", "SPY.US", "VB.US", "RGBITR.INDX", "GLD.US"]
f_date = "2004-12"
l_date = "2020-12"
curr = "RUB"

a_reb = ok.EfficientFrontierReb(
    lsz, ccy=curr, first_date=f_date, last_date=l_date, n_points=40, rebalancing_period="year", verbose=True
)
annual_reb = a_reb.ef_points
print(annual_reb)
