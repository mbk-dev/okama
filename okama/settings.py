from collections import namedtuple

# tickers
default_ticker = "SPY.US"
default_namespace = "US"
default_tickers_list = ["SPY.US", "BND.US"]  # required in single_period.py

# Default Macro Symbols
default_macro_inflation = "RUB.INFL"
default_macro_rate = "RUS_CBR.RATE"
default_macro_indicator = "USA_CAPE10.RATIO"

DEFAULT_DISCOUNT_RATE = 0.05

_MONTHS_PER_YEAR = 12
PeriodLength = namedtuple("PeriodLength", "years months")
