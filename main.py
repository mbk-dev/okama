import okama as ok
init_portfolio_values = dict(
        assets=['RUB.FX', 'MCFTR.INDX'],
        ccy='RUB',
        first_date='2015-01',
        last_date='2020-01',
        inflation=True,
        rebalancing_period='year',
        symbol='pf1.PF',
    )
init_portfolio_values['rebalancing_period'] = 'month'
portfolio_rebalanced_month = ok.Portfolio(**init_portfolio_values)
print(portfolio_rebalanced_month.forecast_monte_carlo_cagr(years=10, distr='lognorm', n=100, percentiles=[50]))
