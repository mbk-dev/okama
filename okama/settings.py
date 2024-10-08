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
DEFAULT_INITIAL_INVESTMENT = 1000.0

_MONTHS_PER_YEAR = 12
PeriodLength = namedtuple("PeriodLength", "years months")

# From Pandas resamples alias: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
frequency_mapping = {"none": "none", "year": "Y", "half-year": "2Q", "quarter": "Q", "month": "M"}
frequency_periods_per_year = {"none": 0, "year": 1, "half-year": 2, "quarter": 4, "month": 12}
