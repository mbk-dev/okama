import okama as ok

close_ts = ok.Asset('TSPX.MOEX').close_monthly
print(close_ts['2022-01':])
