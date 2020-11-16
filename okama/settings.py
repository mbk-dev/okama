default_ticker = 'SPY.US'
default_tickers_list = ['SPY.US', 'BND.US']  # required in frontier.py
default_macro = 'RUB.INFL'

EOD_url = 'https://eodhistoricaldata.com/api/eod/'
EOD_div_url = 'https://eodhistoricaldata.com/api/div/'
EOD_live = 'https://eodhistoricaldata.com/api/real-time/'
eod_search_url = 'https://eodhistoricaldata.com/api/search/'
eod_exchanges_url = 'https://eodhistoricaldata.com/api/exchanges/'

# Namespaces
#
assets_namespaces = ('US', 'XETR', 'MOEX', 'PIF', 'FX', 'INDX', 'COMM', 'RE')
macro_namespaces = ('INFL', 'RATE')

no_dividends_namespaces = ('PIF', 'INDX', 'FX', 'COMM', 'RE') + macro_namespaces

namespaces = {'US': 'US Stock Exchanges and mutual funds',
              'XETR': 'Frankfurt Stock Exchange',
              'MOEX': 'Moscow Exchange',
              'PIF': 'Russian mutual funds',
              'FX': 'FOREX currency market',
              'INDX': 'Indexes',
              'COMM': 'Commodities prices',
              'RE': 'Real estate prices',
              'INFL': 'Inflation',
              'RATE': 'Bank deposit rates'
              }

