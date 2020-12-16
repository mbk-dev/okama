import okama as ok

ln = ok.Portfolio(['EDV.US'], inflation=False)
ln.plot_forecast(distr='lognorm', years=2)
