from abc import ABC
from typing import Union, Tuple

import numpy as np
import pandas as pd

import okama.common.validators
from okama import settings
from okama.api import data_queries, namespaces
from okama.common.helpers import helpers


class MacroABC(ABC):
    """
    Abstract class for all Macroeconomic parameters.

    Parameters
    ----------
    symbol: str
        Symbol (ticker) is unique series of letters with namespace after dot (EUR.INFL).

    first_date : str, default None
        First date of the values monthly time series.

    last_date : str, default None
        Last date of the values monthly time series.
    """

    def __init__(
        self,
        symbol: str,
        first_date: Union[str, pd.Timestamp, None] = None,
        last_date: Union[str, pd.Timestamp, None] = None,
    ):
        self.symbol: str = symbol
        self._check_namespace()
        self._get_symbol_data(symbol)
        self._first_date = first_date
        self._last_date = last_date
        self._values_monthly = self._get_values_monthly()
        self._set_first_last_dates()

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

    def _set_first_last_dates(self):
        self.first_date: pd.Timestamp = self.values_monthly.index[0].to_timestamp()
        self.last_date: pd.Timestamp = self.values_monthly.index[-1].to_timestamp()
        self.pl = settings.PeriodLength(
            self.values_monthly.shape[0] // settings._MONTHS_PER_YEAR,
            self.values_monthly.shape[0] % settings._MONTHS_PER_YEAR,
        )

    def _get_values_monthly(self) -> pd.Series:
        return data_queries.QueryData.get_macro_ts(self.symbol, self._first_date, self._last_date, period="M")

    def _get_symbol_data(self, symbol) -> None:
        x = data_queries.QueryData.get_symbol_info(symbol)
        self.ticker: str = x["code"]
        self.name: str = x["name"]
        self.country: str = x["country"]
        self.currency: str = x["currency"]
        self.type: str = x["type"]

    @property
    def values_monthly(self) -> pd.Series:
        """
        Return values time series historical monthly data.

        Returns
        -------
        Series
            Time series of values historical data (monthly).
        """
        return self._values_monthly

    def set_values_monthly(self, date: str, value: float):
        """
        Set monthly value for the past or future date.

        The date should be in month period format ("2023-12"). T
        The result stored only in the class instance. It can be used to analyze inflation with forecast
        or corrected data.
        """
        okama.common.validators.validate_real("value", value)
        self._values_monthly[pd.Period(date, freq="M")] = value
        self._set_first_last_dates()

    def describe(self, years: Tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
        """
        Generate descriptive statistics for YTD and given periods.
        Statistics includes:
         - arithmetic mean
         - median
         - max and min values

        Parameters
        ----------
        years : tuple of (int,), default (1, 5, 10)
            List of periods for the statistics.

        Returns
        -------
        DataFrame
            Table of descriptive statistics for a list of assets.
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        df = self.values_monthly
        # YTD properties
        year = dt0.year
        ts = df[str(year) :]
        row1 = {self.symbol: ts.mean()}
        row1.update(period="YTD", property="arithmetic mean")
        row2 = {self.symbol: ts.median()}
        row2.update(period="YTD", property="median value")
        # max value
        max_value = ts.nlargest(n=1)
        row3 = {self.symbol: max_value.iloc[0]}
        row3.update(period=max_value.index.values[0].strftime("%Y-%m"), property="max value")
        # min value
        min_value = ts.nsmallest(n=1)
        row4 = {self.symbol: min_value.iloc[0]}
        row4.update(period=min_value.index.values[0].strftime("%Y-%m"), property="min value")

        rows_df = pd.DataFrame.from_records([row1, row2, row3, row4], index=[0, 1, 2, 3])
        description = pd.concat([description, rows_df], ignore_index=True)
        # properties for a given list of periods
        for i in years:
            dt = helpers.Date.subtract_years(dt0, i)
            if dt >= self.first_date:
                ts = df[dt:]
                # arithmetic mean
                row1 = {self.symbol: ts.mean()}
                # median
                row2 = {self.symbol: ts.median()}
                # max value
                max_value = ts.nlargest(n=1)
                row3 = {self.symbol: max_value.iloc[0]}
                row3.update(period=max_value.index.values[0].strftime("%Y-%m"))
                # min value
                min_value = ts.nsmallest(n=1)
                row4 = {self.symbol: min_value.iloc[0]}
                row4.update(period=min_value.index.values[0].strftime("%Y-%m"))
            else:
                row1 = {self.symbol: None}
                row2 = {self.symbol: None}
                row3 = {self.symbol: None}
                row4 = {self.symbol: None}
                row3.update(period=f"{i} years")
                row4.update(period=f"{i} years")
            row1.update(period=f"{i} years", property="arithmetic mean")
            row2.update(period=f"{i} years", property="median")
            row3.update(property="max value")
            row4.update(property="min value")

            new_rows = pd.DataFrame.from_records([row1, row2, row3, row4], index=[0, 1, 2, 3])
            description = pd.concat([description, new_rows], ignore_index=True)
        # Full period
        # Arithmetic mean
        row0 = {self.symbol: df.mean()}
        row0.update(period=self._pl_txt, property="arithmetic mean")
        # Median
        row1 = {self.symbol: df.median()}
        row1.update(period=self._pl_txt, property="median")
        # max value
        max_value = df.nlargest(n=1)
        row2 = {self.symbol: max_value.iloc[0]}
        row2.update(
            period=max_value.index.values[0].strftime("%Y-%m"),
            property="max value",
        )
        # min value
        min_value = df.nsmallest(n=1)
        row3 = {self.symbol: min_value.iloc[0]}
        row3.update(period=min_value.index.values[0].strftime("%Y-%m"), property="min value")
        new_rows = pd.DataFrame.from_records([row0, row1, row2, row3], index=[0, 1, 2, 3])
        description = pd.concat([description, new_rows], ignore_index=True)
        return helpers.Frame.change_columns_order(description, ["property", "period"], position="first")


class Inflation(MacroABC):
    """
    Inflation related data and methods.

    Inflation symbols are in '.INFL' namespace.

    Parameters
    ----------
    symbol: str
        Inflation symbol is unique series of letters with namespace after dot (EUR.INFL).

    first_date : str, default None
        First date of the values time series (2020-01).

    last_date : str, default None
        Last date of the values time series (2022-03).
    """

    def __init__(
        self,
        symbol: str = settings.default_macro_inflation,
        first_date: Union[str, pd.Timestamp, None] = None,
        last_date: Union[str, pd.Timestamp, None] = None,
    ):
        super().__init__(
            symbol,
            first_date=first_date,
            last_date=last_date,
        )

    def _check_namespace(self):
        namespace = self.symbol.split(".", 1)[-1]
        allowed_namespaces = ["INFL"]
        if namespace not in allowed_namespaces:
            raise ValueError(f"{namespace} is not in allowed namespaces: {allowed_namespaces}")

    @property
    def cumulative_inflation(self) -> pd.Series:
        """
        Calculate cumulative inflation rate time series for the whole period.

        Returns
        -------
        Series
            Cumulative inflation rate.

        Examples
        --------
        >>> x = ok.Inflation('RUB.INFL', first_date='2020-01', last_date='2020-12')
        >>> x.cumulative_inflation
        date
        2020-01    0.004000
        2020-02    0.007313
        2020-03    0.012853
        2020-04    0.021260
        2020-05    0.024018
        2020-06    0.026270
        2020-07    0.029862
        2020-08    0.029450
        2020-09    0.028730
        2020-10    0.033153
        2020-11    0.040489
        2020-12    0.049125
        Freq: M, Name: RUB.INFL, dtype: float64
        """
        if self.symbol.split(".", 1)[-1] != "INFL":
            raise ValueError("cumulative_inflation is defined for inflation only")
        return (self.values_monthly + 1.0).cumprod() - 1.0

    @property
    def annual_inflation_ts(self):
        """
        Calculate annual inflation time series.

        Inflation is calculated for each calendar year.

        Returns
        -------
        Series
            Calendar annual Inflation time series.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> infl = ok.Inflation('EUR.INFL', first_date='2016-01', last_date='2021-12')
        >>> infl.annual_inflation_ts.plot(kind='bar')
        >>> plt.show()

        """
        return helpers.Frame.get_annual_return_ts_from_monthly(self.values_monthly)

    @property
    def purchasing_power_1000(self) -> float:
        """
        Calculate purchasing power of 1000 (in the currency of inflation) after period from first_date to last_date.

        Returns
        -------
        float
            The Purchasing power of 1000 currency units.

        Examples
        --------
        >>> x = ok.Inflation('RUB.INFL', first_date='2000-01', last_date='2020-12')
        >>> x.purchasing_power_1000
        145.8118461948026
        """
        return helpers.Float.get_purchasing_power(self.cumulative_inflation[-1])

    @property
    def rolling_inflation(self) -> pd.Series:
        """
        Calculate 12 months rolling inflation time series.

        Returns
        -------
        Series
            12 months rolling inflation time series.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> infl = ok.Inflation('ILS.INFL', first_date='1980-01', last_date='1989-12')
        >>> infl.rolling_inflation.plot()
        >>> plt.show()
        """
        if self.symbol.split(".", 1)[-1] != "INFL":
            raise ValueError("cumulative_inflation is defined for inflation only")
        if self.values_monthly.shape[0] < 12:
            raise ValueError("data history depth is less than rolling window size (12 months)")
        x = (self.values_monthly + 1.0).rolling(settings._MONTHS_PER_YEAR).apply(np.prod, raw=True) - 1.0
        x.dropna(inplace=True)
        return x

    def describe(self, years: Tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
        """
        Generate descriptive inflation statistics for YTD and a given list of periods.
        Statistics includes:
        - YTD compound inflation
        - Annual inflation (geometric mean) for a given list of periods
        - max 12 months inflation for the periods
        - Annual inflation (geometric mean) for the whole history

        Parameters
        ----------
        years : tuple of (int,), default (1, 5, 10)
            List of periods in years for the Inflation.

        Returns
        -------
        DataFrame
            Table of descriptive statistics for Inflation.

        Examples
        --------
        >>> infl = ok.Inflation('USD.INFL', last_date='2022-04')
        >>> infl.describe(years=(1, 15, 50))
                 property               period    USD.INFL
        0      compound inflation                  YTD    0.036987
        1   1000 purchasing power                  YTD  964.332475
        2        annual inflation              1 years    0.082611
        3      compound inflation              1 years    0.082611
        4       max 12m inflation              2022-03    0.085410
        5   1000 purchasing power              1 years  923.692547
        6        annual inflation             15 years    0.022632
        7      compound inflation             15 years    0.398916
        8       max 12m inflation              2022-03    0.085410
        9   1000 purchasing power             15 years  714.839226
        10       annual inflation             50 years    0.039595
        11     compound inflation             50 years    5.969612
        12      max 12m inflation              1980-03    0.147383
        13  1000 purchasing power             50 years  143.480004
        14       annual inflation  109 years, 3 months    0.031470
        15     compound inflation  109 years, 3 months   28.519646
        16      max 12m inflation              1920-06    0.236888
        17  1000 purchasing power  109 years, 3 months   33.875745
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        df = self.values_monthly
        # YTD inflation properties
        year = dt0.year
        ts = df[str(year) :]
        inflation = helpers.Frame.get_cumulative_return(ts)
        row1 = {self.symbol: inflation}
        row1.update(period="YTD", property="compound inflation")

        row2 = {self.symbol: helpers.Float.get_purchasing_power(inflation)}
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
                row1 = {self.symbol: inflation}

                # compound inflation
                comp_inflation = helpers.Frame.get_cumulative_return(ts)
                row2 = {self.symbol: comp_inflation}

                # max inflation
                max_inflation = self.rolling_inflation[dt:].nlargest(n=1)  # largest 12m inflation for selected period
                row3 = {self.symbol: max_inflation.iloc[0]}
                row3.update(period=max_inflation.index.values[0].strftime("%Y-%m"))

                # purchase power
                row4 = {self.symbol: helpers.Float.get_purchasing_power(comp_inflation)}
            else:
                row1 = {self.symbol: None}
                row2 = {self.symbol: None}
                row3 = {self.symbol: None}
                row3.update(period=f"{i} years")
                row4 = {self.symbol: None}
            row1.update(period=f"{i} years", property="annual inflation")

            row2.update(period=f"{i} years", property="compound inflation")

            row3.update(property="max 12m inflation")

            row4.update(period=f"{i} years", property="1000 purchasing power")

            df_rows = pd.DataFrame.from_records([row1, row2, row3, row4], index=[0, 1, 2, 3])
            description = pd.concat([description, df_rows], ignore_index=True)
        # Annual inflation for full period available
        ts = df
        full_inflation = helpers.Frame.get_cagr(ts)
        row = {self.symbol: full_inflation}
        row.update(period=self._pl_txt, property="annual inflation")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # compound inflation
        comp_inflation = helpers.Frame.get_cumulative_return(ts)
        row = {self.symbol: comp_inflation}
        row.update(period=self._pl_txt, property="compound inflation")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # max inflation for full period available
        max_inflation = self.rolling_inflation.nlargest(n=1)
        row = {self.symbol: max_inflation.iloc[0]}
        row.update(
            period=max_inflation.index.values[0].strftime("%Y-%m"),
            property="max 12m inflation",
        )
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # purchase power
        row = {self.symbol: helpers.Float.get_purchasing_power(comp_inflation)}
        row.update(period=self._pl_txt, property="1000 purchasing power")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        return helpers.Frame.change_columns_order(description, ["property", "period"], position="first")


class Rate(MacroABC):
    """
    Rates of central banks and banks.

    Rates symbols are in '.RATE' namespace.

    Parameters
    ----------
    symbol: str
        Symbol is unique series of letters with namespace after dot (RUB_CBR.RATE).

    first_date : str, default None
        First date of the values time series.

    last_date : str, default None
        Last date of the values time series.
    """

    def __init__(
        self,
        symbol: str = settings.default_macro_rate,
        first_date: Union[str, pd.Timestamp, None] = None,
        last_date: Union[str, pd.Timestamp, None] = None,
    ):
        super().__init__(
            symbol,
            first_date=first_date,
            last_date=last_date,
        )

    def _check_namespace(self):
        namespace = self.symbol.split(".", 1)[-1]
        allowed_namespaces = ["RATE"]
        if namespace not in allowed_namespaces:
            raise ValueError(f"{namespace} is not in allowed namespaces: {allowed_namespaces}")

    @property
    def values_daily(self) -> pd.Series:
        """
        Return values time series historical daily data.

        Returns
        -------
        Series
            Time series of values historical data (daily).
        """
        return data_queries.QueryData.get_macro_ts(self.symbol, self._first_date, self._last_date, period="D")


class Indicator(MacroABC):
    """
    Macroeconomic indicators and ratios.

    Parameters
    ----------
    symbol: str
        Symbol is unique series of letters with namespace after dot (USA_CAPE10.RATIO).

    first_date : str, default None
        First date of the values time series (2020-01).

    last_date : str, default None
        Last date of the values time series (2022-03).
    """

    def __init__(
        self,
        symbol: str = settings.default_macro_indicator,
        first_date: Union[str, pd.Timestamp, None] = None,
        last_date: Union[str, pd.Timestamp, None] = None,
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
        restricted_namespaces = ["RATE", "INFL"]
        allowed_namespaces = [x for x in all_macro_namespaces if x not in restricted_namespaces]
        if namespace not in allowed_namespaces:
            raise ValueError(f"{namespace} is not in allowed namespaces: {allowed_namespaces}")
