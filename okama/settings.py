default_ticker = 'SPY.US'
default_tickers_list = ['SPY.US', 'BND.US']  # required in frontier.py
default_macro = 'RUB.INFL'

EOD_url = 'https://eodhistoricaldata.com/api/eod/'
EOD_div_url = 'https://eodhistoricaldata.com/api/div/'
EOD_live = 'https://eodhistoricaldata.com/api/real-time/'
eod_search_url = 'https://eodhistoricaldata.com/api/search/'
eod_exchanges_url = 'https://eodhistoricaldata.com/api/exchanges/'

# Assets
#
eod_namespaces = ('US', 'MCX', 'F', 'RUFUND', 'FOREX', 'INDX', 'COMM')  # F - Frankfurt Exchange
sql_tickers = ('EURRUB.FOREX', 'RUB.FOREX', 'MCFTR.INDX')  # CNYRUB.FORX is also stored (probably from cbr.ru) up to 2020/03

# Real Estate symbols
real_estate_symbols = {'RUS_SEC.RE': 'Russian real estate. Secondary market.',
                       'MOW_SEC.RE': 'Moscow real estate. Secondary market.',
                       'RUS_PR.RE': 'Russian real estate. Primary market.',
                       'MOW_PR.RE': 'Moscow real estate. Primary market.'}
# OKID indexes
okid_indexes = {'OKID.INDX': 'RUS_RUB.RATE',
                'OKID10.INDX': 'RUS_RUB_TOP10.RATE',
                'OKID_USD.INDX': 'RUS_USD.RATE',
                'OKID_EUR.INDX': 'RUS_EUR.RATE'}

# Macroeconomic indicators
#
rates_symbols = ('RUS_RUB.RATE', 'RUS_RUB_TOP10.RATE', 'RUS_USD.RATE', 'RUS_EUR.RATE')
inflation_list = ['RUB.INFL', 'USD.INFL', 'EUR.INFL']

# Namespaces
#
assets_namespaces = ('US', 'F', 'MCX', 'RUFUND', 'FOREX', 'INDX', 'COMM', 'RE')
macro_namespaces = ('INFL', 'RATE')
namespaces = assets_namespaces + macro_namespaces

