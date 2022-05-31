from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd

from okama import settings
from okama.api import data_queries, namespaces
from okama.common.helpers import helpers


class MacroABC(ABC):
    def __init__(
        self,
        symbol: str,
        first_date: Union[str, pd.Timestamp] = "1800-01",
        last_date: Union[str, pd.Timestamp] = "2030-01",
    ):
        self.symbol: str = symbol
        self._check_namespace()
        self._get_symbol_data(symbol)
        self.values_monthly: pd.Series = data_queries.QueryData.get_macro_ts(symbol, first_date, last_date, period="M")
        self.first_date: pd.Timestamp = self.values_monthly.index[0].to_timestamp()
        self.last_date: pd.Timestamp = self.values_monthly.index[-1].to_timestamp()
        self.pl = settings.PeriodLength(
            self.values_monthly.shape[0] // settings._MONTHS_PER_YEAR,
            self.values_monthly.shape[0] % settings._MONTHS_PER_YEAR,
        )
        self._pl_txt = f"{self.pl.years} years, {self.pl.months} months"

    def __repr__(self):
        dic = {
            "symbol": self.symbol,
            "name": self.name,
            "country": self.country,
            "currency": self.currency,
            "type": self.type,
            "first date": self.first_date.strftime("%Y-%m"),
            "last date": self.last_date.strftime("%Y-%m"),
            "period length": self._pl_txt,
        }
        return repr(pd.Series(dic))

    def _check_namespace(self):
        pass

    def _get_symbol_data(self, symbol):
        x = data_queries.QueryData.get_symbol_info(symbol)
        self.ticker: str = x["code"]
        self.name: str = x["name"]
        self.country: str = x["country"]
        self.currency: str = x["currency"]
        self.type: str = x["type"]

    @abstractmethod
    def describe(self):
        pass


class Inflation(MacroABC):
    """
    Inflation related data and methods.

    Inflation symbols are in '.INFL' namespace.
    """

    def __init__(
            self,
            symbol: str = settings.default_macro_inflation,
            first_date: Union[str, pd.Timestamp] = "1800-01",
            last_date: Union[str, pd.Timestamp] = "2030-01",
    ):
        super().__init__(
            symbol,
            first_date=first_date,
            last_date=last_date,
        )

    def _check_namespace(self):
        namespace = self.symbol.split(".", 1)[-1]
        allowed_namespaces = ['INFL']
        if namespace not in allowed_namespaces:
            raise ValueError(f"{namespace} is not in allowed namespaces: {allowed_namespaces}")

    @property
    def cumulative_inflation(self) -> pd.Series:
        """
        Return cumulative inflation rate time series for a period from first_date to last_date.
        """
        if self.symbol.split(".", 1)[-1] != "INFL":
            raise ValueError("cumulative_inflation is defined for inflation only")
        return (self.values_monthly + 1.0).cumprod() - 1.0

    @property
    def annual_inflation_ts(self):
        return helpers.Frame.get_annual_return_ts_from_monthly(self.values_monthly)

    @property
    def purchasing_power_1000(self) -> helpers.Float:
        """
        Return purchasing power of 1000 (in a currency of inflation) after period from first_date to last_date.
        """
        return helpers.Float.get_purchasing_power(self.cumulative_inflation[-1])

    @property
    def rolling_inflation(self) -> pd.Series:
        """
        Return 12 months rolling inflation time series.
        """
        if self.symbol.split(".", 1)[-1] != "INFL":
            raise ValueError("cumulative_inflation is defined for inflation only")
        x = (self.values_monthly + 1.0).rolling(settings._MONTHS_PER_YEAR).apply(np.prod, raw=True) - 1.0
        x.dropna(inplace=True)
        return x

    def describe(self, years: Tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
        """
        Generate descriptive inflation statistics for a given list of tickers.
        Statistics includes:
        - YTD compound inflation
        - Annual inflation (geometric mean) for a given list of periods
        - max 12 months inflation for the periods
        - Annual inflation (geometric mean) for the whole history
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        df = self.values_monthly
        # YTD inflation properties
        year = pd.Timestamp.today().year
        ts = df[str(year) :]
        inflation = helpers.Frame.get_cumulative_return(ts)
        row1 = {self.name: inflation}
        row1.update(period="YTD", property="compound inflation")

        row2 = {self.name: helpers.Float.get_purchasing_power(inflation)}
        row2.update(period="YTD", property="1000 purchasing power")
        rows_df = pd.DataFrame.from_records([row1, row2], index=[0, 1])
        description = pd.concat([description, rows_df], ignore_index=True)

        # inflation properties for a given list of periods
        for i in years:
            dt = helpers.Date.subtract_years(dt0, i)
            if dt >= self.first_date:
                ts = df[dt:]
                # mean inflation
                inflation = helpers.Frame.get_cagr(ts)
                row1 = {self.name: inflation}

                # compound inflation
                comp_inflation = helpers.Frame.get_cumulative_return(ts)
                row2 = {self.name: comp_inflation}

                # max inflation
                max_inflation = self.rolling_inflation[dt:].nlargest(n=1)  # largest 12m inflation for selected period
                row3 = {self.name: max_inflation.iloc[0]}
                row3.update(period=max_inflation.index.values[0].strftime("%Y-%m"))

                # purchase power
                row4 = {self.name: helpers.Float.get_purchasing_power(comp_inflation)}
            else:
                row1 = {self.name: None}
                row2 = {self.name: None}
                row3 = {self.name: None}
                row3.update(period=f"{i} years")
                row4 = {self.name: None}
            row1.update(period=f"{i} years", property="annual inflation")

            row2.update(period=f"{i} years", property="compound inflation")

            row3.update(property="max 12m inflation")

            row4.update(period=f"{i} years", property="1000 purchasing power")

            df_rows = pd.DataFrame.from_records([row1, row2, row3, row4], index=[0, 1, 2, 3])
            description = pd.concat([description, df_rows], ignore_index=True)
        # Annual inflation for full period available
        ts = df
        full_inflation = helpers.Frame.get_cagr(ts)
        row = {self.name: full_inflation}
        row.update(period=self._pl_txt, property="annual inflation")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # compound inflation
        comp_inflation = helpers.Frame.get_cumulative_return(ts)
        row = {self.name: comp_inflation}
        row.update(period=self._pl_txt, property="compound inflation")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # max inflation for full period available
        max_inflation = self.rolling_inflation.nlargest(n=1)
        row = {self.name: max_inflation.iloc[0]}
        row.update(
            period=max_inflation.index.values[0].strftime("%Y-%m"),
            property="max 12m inflation",
        )
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # purchase power
        row = {self.name: helpers.Float.get_purchasing_power(comp_inflation)}
        row.update(period=self._pl_txt, property="1000 purchasing power")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        return helpers.Frame.change_columns_order(description, ["property", "period"], position="first")


class Rate(MacroABC):
    """
    Rates of central banks and banks.

    Rates symbols are in '.RATE' namespace.

    TODO: Add .values_daily property
    """

    def __init__(
            self,
            symbol: str = settings.default_macro_rate,
            first_date: Union[str, pd.Timestamp] = "1800-01",
            last_date: Union[str, pd.Timestamp] = "2030-01",
    ):
        super().__init__(
            symbol,
            first_date=first_date,
            last_date=last_date,
        )
        self.values_daily: pd.Series = data_queries.QueryData.get_macro_ts(symbol, first_date, last_date, period="D")

    def _check_namespace(self):
        namespace = self.symbol.split(".", 1)[-1]
        allowed_namespaces = ['RATE']
        if namespace not in allowed_namespaces:
            raise ValueError(f"{namespace} is not in allowed namespaces: {allowed_namespaces}")

    def describe(self, years: Tuple[int, ...] = (1, 5, 10)):
        # TODO: Make describe()
        pass


class Indicator(MacroABC):
    """
    Macroeconomic indicators and ratios.
    """
    def __init__(
            self,
            symbol: str = settings.default_macro_indicator,
            first_date: Union[str, pd.Timestamp] = "1800-01",
            last_date: Union[str, pd.Timestamp] = "2030-01",
    ):
        super().__init__(
            symbol,
            first_date=first_date,
            last_date=last_date,
        )

    def _check_namespace(self):
        """
        Allowed all macro namespaces except 'INFL' and 'RATES'.
        """
        namespace = self.symbol.split(".", 1)[-1]
        all_macro_namespaces = namespaces.get_macro_namespaces()
        restricted_namespaces = ['RATE', 'INFL']
        allowed_namespaces = [x for x in all_macro_namespaces if x not in restricted_namespaces]
        if namespace not in allowed_namespaces:
            raise ValueError(f"{namespace} is not in allowed namespaces: {allowed_namespaces}")

    def describe(self, years: Tuple[int, ...] = (1, 5, 10)):
        # TODO: Make describe()
        pass
