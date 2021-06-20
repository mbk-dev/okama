import okama as ok

rf3 = ok.Portfolio(['BND.US', 'VTI.US', 'VXUS.US'], weights=[0.40, 0.40, 0.20], rebalancing_period='year')
rf3.symbol = '1 1.PF'
print(rf3)
