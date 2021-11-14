import okama as ok

tk = ['VCIT.US',
 'VTIP.US',
 'TIP.US',
 'SCHP.US',
 'FXRU.MOEX',
 'HACK.US',
 'JETS.US',
 'XAR.US',
 'QQQ.US',
 'VBR.US',
 'VBK.US',
 'PGJ.US',
 'VOO.US',
 'IHAK.US',
 'VNQ.US',
 'SLV.US',
 'GLDM.US']

wt = [0.14,
 0.13,
 0.03,
 0.29,
 0.05,
 0.03,
 0.01,
 0.01,
 0.01,
 0.04,
 0.03,
 0.01,
 0.02,
 0.02,
 0.03,
 0.03,
 0.12]

pf = ok.Portfolio(assets=tk, weights=wt, ccy='USD', inflation=True)

print(pf.dividend_yield)
