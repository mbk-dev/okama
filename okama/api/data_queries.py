from typing import Dict

from io import StringIO
import json

import pandas as pd
import numpy as np

from okama.api import api_methods, namespaces


class QueryData:
    """
    Set of methods to get symbols data from api_methods.API.
    """

    @staticmethod
    def get_symbol_info(symbol: str) -> Dict[str, str]:
        json_input = api_methods.API.get_symbol_info(symbol)
        return json.loads(json_input)

    @staticmethod
    def csv_to_series(csv_input: str, period: str) -> pd.Series:
        ts = pd.read_csv(
            StringIO(csv_input),
            delimiter=",",
            index_col=0,
            parse_dates=[0],
            dtype={1: np.float64},
            engine="python",
        )
        if not ts.empty:
            ts.index = ts.index.to_period(period.upper())
            ts = ts.squeeze("columns")
        return ts

    @staticmethod
    def get_macro_ts(
        symbol: str, first_date: str = "1913-01-01", last_date: str = "2100-01-01"
    ) -> pd.Series:
        """
        Requests api_methods.API for Macroeconomic indicators time series (monthly data).
        - Inflation time series
        - Bank rates time series
        """
        csv_input = api_methods.API.get_macro(
            symbol=symbol, first_date=first_date, last_date=last_date
        )
        return QueryData.csv_to_series(csv_input, period="M")

    @staticmethod
    def get_ror(
        symbol: str,
        first_date: str = "1913-01-01",
        last_date: str = "2100-01-01",
        period="M",
    ) -> pd.Series:
        """
        Requests api_methods.API for rate of return time series.
        """
        csv_input = api_methods.API.get_ror(
            symbol=symbol, first_date=first_date, last_date=last_date, period=period
        )
        return QueryData.csv_to_series(csv_input, period)

    @staticmethod
    def get_nav(
        symbol: str,
        first_date: str = "1913-01-01",
        last_date: str = "2100-01-01",
        period="M",
    ) -> pd.Series:
        """
        NAV time series for funds (works for PIF namespace only).
        """
        csv_input = api_methods.API.get_nav(
            symbol=symbol, first_date=first_date, last_date=last_date, period=period
        )
        return QueryData.csv_to_series(csv_input, period=period)

    @staticmethod
    def get_close(
        symbol: str,
        first_date: str = "1913-01-01",
        last_date: str = "2100-01-01",
        period="M",
    ) -> pd.Series:
        """
        Gets 'close' time series for a ticker.
        """
        csv_input = api_methods.API.get_close(
            symbol=symbol, first_date=first_date, last_date=last_date, period=period
        )
        return QueryData.csv_to_series(csv_input, period)

    @staticmethod
    def get_adj_close(
        symbol: str,
        first_date: str = "1913-01-01",
        last_date: str = "2100-01-01",
        period="M",
    ) -> pd.Series:
        """
        Gets 'adjusted close' time series for a ticker.
        """
        csv_input = api_methods.API.get_adjusted_close(
            symbol=symbol, first_date=first_date, last_date=last_date, period=period
        )
        return QueryData.csv_to_series(csv_input, period)

    @staticmethod
    def get_dividends(
        symbol: str, first_date: str = "1913-01-01", last_date: str = "2100-01-01",
    ) -> pd.Series:
        """
        Dividends time series daily data (dividend payment day should be considered).
        """
        if symbol.split(".", 1)[-1] not in namespaces.no_dividends_namespaces():
            csv_input = api_methods.API.get_dividends(
                symbol, first_date=first_date, last_date=last_date
            )
            ts = QueryData.csv_to_series(csv_input, period="D")
        else:
            # make empty time series when no dividends
            ts = pd.Series(dtype=float)
            ts.rename(symbol, inplace=True)
        return ts

    @staticmethod
    def get_live_price(symbol: str) -> float:
        price = api_methods.API.get_live_price(symbol)
        return float(price)
