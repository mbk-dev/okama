import okama as ok
x = ok.Portfolio(['SPY.US', 'SBERP.MOEX'], weights=[0.2, 0.8])
print(x.recovery_period)
