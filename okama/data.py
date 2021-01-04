from typing import Dict
from functools import lru_cache

from io import StringIO
import json

import requests
import pandas as pd
import numpy as np

from .settings import default_ticker


def search(search_string: str) -> json:
    string_response = API.search(search_string)
    return json.loads(string_response)


@lru_cache()
def get_namespaces():
    string_response = API.get_namespaces()
    return json.loads(string_response)


@lru_cache()
def get_assets_namespaces():
    string_response = API.get_assets_namespaces()
    return json.loads(string_response)


@lru_cache()
def get_macro_namespaces():
    string_response = API.get_macro_namespaces()
    return json.loads(string_response)


@lru_cache()
def no_dividends_namespaces():
    string_response = API.get_no_dividends_namespaces()
    return json.loads(string_response)


class QueryData:
    """
    Set of methods to select a source and get_ts the data.
    """

    @staticmethod
    def get_symbol_info(symbol: str) -> Dict[str, str]:
        json_input = API.get_symbol_info(symbol)
        return json.loads(json_input)

    @staticmethod
    def csv_to_series(csv_input: str, period: str) -> pd.Series:
        ts = pd.read_csv(StringIO(csv_input),
                         delimiter=',',
                         index_col=0,
                         parse_dates=[0],
                         dtype={1: np.float64},
                         engine='python')
        if not ts.empty:
            ts.index = ts.index.to_period(period.upper())
            ts = ts.squeeze('columns')
        return ts

    @staticmethod
    def get_macro_ts(symbol: str, first_date: str = '1913-01-01', last_date: str = '2100-01-01') -> pd.Series:
        """
        Requests API for Macroeconomic indicators time series (monthly data).
        - Inflation time series
        - Bank rates time series
        """
        csv_input = API.get_macro(symbol=symbol, first_date=first_date, last_date=last_date)
        return QueryData.csv_to_series(csv_input, period='M')

    @staticmethod
    def get_ror(symbol: str, first_date: str = '1913-01-01', last_date: str = '2100-01-01', period='M') -> pd.Series:
        """
        Requests API for rate of return time series.
        """
        csv_input = API.get_ror(symbol=symbol, first_date=first_date, last_date=last_date, period=period)
        return QueryData.csv_to_series(csv_input, period)

    @staticmethod
    def get_nav(symbol: str, first_date: str = '1913-01-01', last_date: str = '2100-01-01', period='M') -> pd.Series:
        """
        NAV time series for funds (works for PIF namespace only).
        """
        csv_input = API.get_nav(symbol=symbol, first_date=first_date, last_date=last_date, period=period)
        return QueryData.csv_to_series(csv_input, period=period)

    @staticmethod
    def get_close(symbol: str, first_date: str = '1913-01-01', last_date: str = '2100-01-01', period='M') -> pd.Series:
        """
        Gets 'close' time series for a ticker.
        """
        csv_input = API.get_close(symbol=symbol, first_date=first_date, last_date=last_date, period=period)
        return QueryData.csv_to_series(csv_input, period)

    @staticmethod
    def get_adj_close(symbol: str, first_date: str = '1913-01-01', last_date: str = '2100-01-01', period='M') -> pd.Series:
        """
        Gets 'adjusted close' time series for a ticker.
        """
        csv_input = API.get_adjusted_close(symbol=symbol, first_date=first_date, last_date=last_date, period=period)
        return QueryData.csv_to_series(csv_input, period)

    @staticmethod
    def get_dividends(symbol: str, first_date: str = '1913-01-01', last_date: str = '2100-01-01',) -> pd.Series:
        """
        Dividends time series daily data (dividend payment day should be considered).
        """
        if symbol.split('.', 1)[-1] not in no_dividends_namespaces():
            csv_input = API.get_dividends(symbol, first_date=first_date, last_date=last_date)
            ts = QueryData.csv_to_series(csv_input, period='D')
        else:
            # make empty time series when no dividends
            ts = pd.Series(dtype=float)
            ts.rename(symbol, inplace=True)
        return ts

    @staticmethod
    def get_live_price(symbol: str) -> float:
        price = API.get_live_price(symbol)
        return float(price)


class API:
    """
    Set of methods to data from API.
    TODO: introduce 'from' & 'to' for dates.
    """

    api_url = 'http://185.63.191.70:5000'

    endpoint_ror = '/api/ts/ror/'
    endpoint_symbol = '/api/symbol/'
    endpoint_search = '/api/search/'
    endpoint_live_price = '/api/live_price/'
    endpoint_adjusted_close = '/api/ts/adjusted_close/'
    endpoint_close = '/api/ts/close/'
    endpoint_dividends = '/api/ts/dividends/'
    endpoint_nav = '/api/ts/nav/'
    endpoint_macro = '/api/ts/macro/'
    # namespaces endpoints
    endpoint_namespaces = '/api/namespaces/'
    endpoint_assets_namespaces = '/api/assets_namespaces/'
    endpoint_macro_namespaces = '/api/macro_namespaces/'
    endpoint_no_dividends_namespaces = '/api/no_dividends_namespaces/'

    @classmethod
    def connect(cls,
                endpoint: str = endpoint_ror,
                symbol: str = default_ticker,
                first_date: str = '1900-01-01',
                last_date: str = '2100-01-01',
                period: str = 'd',
                ) -> str:
        session = requests.session()
        request_url = cls.api_url + endpoint + symbol
        params = {'first_date': first_date, 'last_date': last_date, 'period': period}
        r = session.get(request_url, params=params, verify=False)
        if r.status_code != requests.codes.ok:
            raise Exception(f'Error fetching data for {symbol}:', r.status_code, r.reason, request_url)
        return r.text

    @classmethod
    def get_ror(cls,
                symbol: str = default_ticker,
                first_date: str = '1900-01-01',
                last_date: str = '2100-01-01',
                period: str = 'm'):
        return cls.connect(endpoint=cls.endpoint_ror,
                           symbol=symbol,
                           first_date=first_date,
                           last_date=last_date,
                           period=period)

    @classmethod
    def get_adjusted_close(cls,
                           symbol: str = default_ticker,
                           first_date: str = '1900-01-01',
                           last_date: str = '2100-01-01',
                           period: str = 'm'):
        return cls.connect(endpoint=cls.endpoint_adjusted_close,
                           symbol=symbol,
                           first_date=first_date,
                           last_date=last_date,
                           period=period)

    @classmethod
    def get_close(cls,
                  symbol: str = default_ticker,
                  first_date: str = '1900-01-01',
                  last_date: str = '2100-01-01',
                  period: str = 'm'):
        return cls.connect(endpoint=cls.endpoint_close,
                           symbol=symbol,
                           first_date=first_date,
                           last_date=last_date,
                           period=period)

    @classmethod
    def get_dividends(cls,
                      symbol: str = default_ticker,
                      first_date: str = '1900-01-01',
                      last_date: str = '2100-01-01'):
        return cls.connect(endpoint=cls.endpoint_dividends,
                           symbol=symbol,
                           first_date=first_date,
                           last_date=last_date)

    @classmethod
    def get_nav(cls,
                symbol: str = default_ticker,
                first_date: str = '1900-01-01',
                last_date: str = '2100-01-01',
                period: str = 'm'):
        return cls.connect(endpoint=cls.endpoint_nav,
                           symbol=symbol,
                           first_date=first_date,
                           last_date=last_date,
                           period=period)

    @classmethod
    def get_macro(cls,
                  symbol: str = default_ticker,
                  first_date: str = '1900-01-01',
                  last_date: str = '2100-01-01'):
        """
        Get macro time series (monthly).
        """
        return cls.connect(endpoint=cls.endpoint_macro,
                           symbol=symbol,
                           first_date=first_date,
                           last_date=last_date,
                           period='m')

    @classmethod
    def get_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_namespaces, symbol='')

    @classmethod
    def get_assets_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_assets_namespaces, symbol='')

    @classmethod
    def get_macro_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_macro_namespaces, symbol='')

    @classmethod
    def get_no_dividends_namespaces(cls):
        return cls.connect(endpoint=cls.endpoint_no_dividends_namespaces, symbol='')

    @classmethod
    def get_symbol_info(cls, symbol: str):
        return cls.connect(endpoint=cls.endpoint_symbol, symbol=symbol)

    @classmethod
    def search(cls, search_string: str):
        return cls.connect(endpoint=cls.endpoint_search, symbol=search_string)

    @classmethod
    def get_live_price(cls, symbol: str):
        return cls.connect(endpoint=cls.endpoint_live_price, symbol=symbol)


namespaces = get_namespaces()
assets_namespaces = get_assets_namespaces()
macro_namespaces = get_macro_namespaces()