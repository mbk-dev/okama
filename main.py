import okama as ok

pf = ok.Portfolio()

sc = pf.percentile_inverse_cagr(score=0, distr='hist')
print(sc)
