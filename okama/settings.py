import os
from collections import namedtuple

# API
api_url: str = os.getenv("OKAMA_API_URL", "https://api.okama.io")
api_default_timeout: int = int(os.getenv("OKAMA_API_TIMEOUT", "10"))  # seconds

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


def resolve_n_jobs() -> int:
    """Resolve the joblib ``n_jobs`` for EfficientFrontier parallel loops.

    Guards against multiplicative nested process parallelism (issue #94): the
    frontier loops use a process (loky) backend, so running them inside another
    parallel context multiplies worker counts (``N`` outer x ``N`` inner) and
    can exhaust RAM. This collapses to a single job whenever the caller is
    already inside a parallel context:

    - a pytest-xdist worker, signalled by the ``PYTEST_XDIST_WORKER`` env var;
    - an active joblib pool, signalled by a backend nesting level above zero
      (checked via ``joblib.parallel.get_active_backend``; the backend *type*
      is not a reliable signal because the top-level default is already a
      non-sequential loky backend).

    Otherwise it returns the configured degree of parallelism from the
    ``OKAMA_N_JOBS`` environment variable, defaulting to ``-1`` (all cores).
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return 1
    try:
        from joblib.parallel import get_active_backend

        backend, _ = get_active_backend()
        if getattr(backend, "nesting_level", 0) > 0:
            return 1
    except Exception:
        pass
    return int(os.getenv("OKAMA_N_JOBS", "-1"))
