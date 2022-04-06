import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from okama import settings


class API:
    """
    Set of methods to get data from API.
    """
    # TODO: introduce 'from' & 'to' for dates.

    api_url = "http://api.okama.io:5000"
    default_timeout = 5  # secondsq

    endpoint_ror = "/api/ts/ror/"
    endpoint_symbol = "/api/symbol/"
    endpoint_search = "/api/search/"
    endpoint_live_price = "/api/live_price/"
    endpoint_adjusted_close = "/api/ts/adjusted_close/"
    endpoint_close = "/api/ts/close/"
    endpoint_dividends = "/api/ts/dividends/"
    endpoint_nav = "/api/ts/nav/"
    endpoint_macro = "/api/ts/macro/"
    # namespaces endpoints
    endpoint_namespaces = "/api/namespaces/"
    endpoint_assets_namespaces = "/api/assets_namespaces/"
    endpoint_macro_namespaces = "/api/macro_namespaces/"
    endpoint_no_dividends_namespaces = "/api/no_dividends_namespaces/"

    @classmethod
    def connect(
        cls,
        endpoint: str = endpoint_ror,
        symbol: str = settings.default_ticker,
        first_date: str = "1900-01-01",
        last_date: str = "2100-01-01",
        period: str = "d",
    ) -> str:
        session = requests.session()
        retry_strategy = Retry(total=3,
                               backoff_factor=0.1,
                               status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        request_url = cls.api_url + endpoint + symbol
        params = {"first_date": first_date, "last_date": last_date, "period": period}
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        try:
            r = session.get(request_url, params=params, verify=False, timeout=cls.default_timeout)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            if r.status_code == 404:
                raise requests.exceptions.HTTPError(f"{symbol} is not found in the database.", 404) from errh
            raise requests.exceptions.HTTPError(
                f"HTTP error fetching data for {symbol}:",
                r.status_code,
                r.reason,
                request_url,
            ) from errh
        return r.text

    @classmethod
    def get_ror(
        cls,
        symbol: str = settings.default_ticker,
        first_date: str = "1900-01-01",
        last_date: str = "2100-01-01",
        period: str = "m",
    ):
        return cls.connect(
            endpoint=cls.endpoint_ror,
            symbol=symbol,
            first_date=first_date,
            last_date=last_date,
            period=period,
        )

    @classmethod
    def get_adjusted_close(
        cls,
        symbol: str = settings.default_ticker,
        first_date: str = "1900-01-01",
        last_date: str = "2100-01-01",
        period: str = "m",
    ):
        return cls.connect(
            endpoint=cls.endpoint_adjusted_close,
            symbol=symbol,
            first_date=first_date,
            last_date=last_date,
            period=period,
        )

    @classmethod
    def get_close(
        cls,
        symbol: str = settings.default_ticker,
        first_date: str = "1900-01-01",
        last_date: str = "2100-01-01",
        period: str = "m",
    ):
        return cls.connect(
            endpoint=cls.endpoint_close,
            symbol=symbol,
            first_date=first_date,
            last_date=last_date,
            period=period,
        )

    @classmethod
    def get_dividends(
        cls,
        symbol: str = settings.default_ticker,
        first_date: str = "1900-01-01",
        last_date: str = "2100-01-01",
    ):
        return cls.connect(
            endpoint=cls.endpoint_dividends,
            symbol=symbol,
            first_date=first_date,
            last_date=last_date,
        )

    @classmethod
    def get_nav(
        cls,
        symbol: str = settings.default_ticker,
        first_date: str = "1900-01-01",
        last_date: str = "2100-01-01",
        period: str = "m",
    ):
        return cls.connect(
            endpoint=cls.endpoint_nav,
            symbol=symbol,
            first_date=first_date,
            last_date=last_date,
            period=period,
        )

    @classmethod
    def get_macro(
        cls,
        symbol: str = settings.default_ticker,
        first_date: str = "1900-01-01",
        last_date: str = "2100-01-01",
    ):
        """
        Get macro time series (monthly).
        """
        return cls.connect(
            endpoint=cls.endpoint_macro,
            symbol=symbol,
            first_date=first_date,
            last_date=last_date,
            period="m",
        )

    @classmethod
    def get_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_namespaces, symbol="")

    @classmethod
    def get_symbols_in_namespace(cls, namespace: str = settings.default_namespace):
        return cls.connect(endpoint=cls.endpoint_namespaces, symbol=namespace)

    @classmethod
    def get_assets_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_assets_namespaces, symbol="")

    @classmethod
    def get_macro_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_macro_namespaces, symbol="")

    @classmethod
    def get_no_dividends_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_no_dividends_namespaces, symbol="")

    @classmethod
    def get_symbol_info(cls, symbol: str):
        return cls.connect(endpoint=cls.endpoint_symbol, symbol=symbol)

    @classmethod
    def search(cls, search_string: str):
        return cls.connect(endpoint=cls.endpoint_search, symbol=search_string)

    @classmethod
    def get_live_price(cls, symbol: str):
        return cls.connect(endpoint=cls.endpoint_live_price, symbol=symbol)
