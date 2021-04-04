from collections import namedtuple

default_ticker = "SPY.US"
default_namespace = "US"
default_tickers_list = ["SPY.US", "BND.US"]  # required in single_period.py
default_macro = "RUB.INFL"

_MONTHS_PER_YEAR = 12
PeriodLength = namedtuple("PeriodLength", "years months")
