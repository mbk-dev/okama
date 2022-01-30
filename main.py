import okama as ok

ls3 = ['MCFTR.INDX', 'RGBITR.INDX', 'GC.COMM']
y = ok.EfficientFrontier(assets=ls3, ccy='USD', n_points=10)

print(y.mdp_points)
