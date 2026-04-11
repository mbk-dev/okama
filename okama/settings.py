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

# Distributions
distributions = ("norm", "lognorm", "t")

# Period frequencies for PeriodIndex resampling (Y, Q, M are valid period aliases)
frequency_mapping = {"none": "none", "year": "Y", "half-year": "2Q", "quarter": "Q", "month": "M"}
# Offset aliases for pd.Grouper (YE, QE, ME are the pandas 3 offset aliases)
grouper_frequency_mapping = {"none": "none", "year": "YE", "half-year": "2QE", "quarter": "QE", "month": "ME"}
frequency_periods_per_year = {"none": 0, "year": 1, "half-year": 2, "quarter": 4, "month": 12}
