import okama as ok

ls = ['SPY.US', 'GLD.US']
curr = 'USD'
y = ok.EfficientFrontierReb(assets=ls,
                            first_date='2004-12',
                            last_date='2020-10',
                            ccy=curr,
                            rebalancing_period='year',  # set rebalancing period to one year
                            ticker_names=True,  # use tickers in DataFrame column names (can be set to False to show full assets names instead tickers)
                            n_points=2,  # number of points in the Efficient Frontier
                            full_frontier=True,
                            verbose=True)  # verbose mode is ON to show progress while the EF points are calcualted

df_reb_year = y.ef_points


print(df_reb_year.head(5))
