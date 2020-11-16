from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from okama.data import QueryData
from okama.helpers import Float, Frame, Date
from okama.settings import default_macro

_MONTHS_PER_YEAR = 12


class MacroABC(ABC):
    def __init__(self, symbol: str = default_macro, first_date: str = '1800-01', last_date: str = '2030-01'):
        self.symbol: str = symbol
        self._get_symbol_data(symbol)
        self.values_ts: pd.Series = QueryData.get_macro_ts(symbol, first_date, last_date)
        self.first_date: pd.Timestamp = self.values_ts.index[0].to_timestamp()
        self.last_date: pd.Timestamp = self.values_ts.index[-1].to_timestamp()
        self.period_length: float = round((self.last_date - self.first_date) / np.timedelta64(365, 'D'), ndigits=1)

    def __repr__(self):
        dic = {
            'symbol': self.symbol,
            'name': self.name,
            'country': self.country,
            'currency': self.currency,
            'type': self.type,
            'first date': self.first_date.strftime("%Y-%m"),
            'last date': self.last_date.strftime("%Y-%m"),
            'period length (Y)': "{:.2f}".format(self.period_length)
        }
        return repr(pd.Series(dic))

    def _get_symbol_data(self, symbol):
        x = QueryData.get_symbol_info(symbol)
        self.ticker: str = x['code']
        self.name: str = x['name']
        self.country: str = x['country']
        self.currency: str = x['currency']
        self.type: str = x['type']

    @abstractmethod
    def describe(self):
        pass


class Inflation(MacroABC):
    """
   Inflation related data and methods.
    """

    @property
    def cumulative_inflation(self) -> pd.Series:
        """
        Return cumulative inflation rate time series for a period from first_date to last_date.
        """
        if self.symbol.split(".", 1)[-1] == 'INFL':
            return (self.values_ts + 1.).cumprod() - 1.
        else:
            raise Exception('cumulative_inflation is defined for inflation only')

    @property
    def annual_inflation_ts(self):
        return Frame.get_annual_return_ts_from_monthly(self.values_ts)

    @property
    def purchasing_power_1000(self) -> Float:
        """
        Return purchasing power of 1000 (in a currency of inflation) after period from first_date to last_date.
        """
        return 1000. / (1. + self.cumulative_inflation[-1])

    @property
    def rolling_inflation(self) -> pd.Series:
        """
        Return 12 months rolling inflation time series.
        """
        if self.symbol.split(".", 1)[-1] == 'INFL':
            x = (self.values_ts + 1.).rolling(_MONTHS_PER_YEAR).apply(np.prod, raw=True) - 1.
            x.dropna(inplace=True)
            return x
        else:
            raise Exception('cumulative_inflation is defined for inflation only')

    def describe(self, years=[1, 5, 10]) -> pd.DataFrame:
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
        df = self.values_ts
        # YTD inflation properties
        year = pd.Timestamp.today().year
        ts = df[str(year):]
        inflation = Frame.get_compound_return(ts)
        row1 = {self.name: inflation}
        row1.update({'period': 'YTD'})
        row1.update({'property': 'compound inflation'})

        row2 = {self.name: 1000. / (1. + inflation)}
        row2.update({'period': 'YTD'})
        row2.update({'property': '1000 purchasing power'})

        description = description.append(row1, ignore_index=True)
        description = description.append(row2, ignore_index=True)

        # inflation properties for a given list of periods
        for i in years:
            dt = Date.subtract_years(dt0, i)
            if dt >= self.first_date:
                ts = df[dt:]
                # mean inflation
                inflation = Frame.get_cagr(ts)
                row1 = {self.name: inflation}

                # compound inflation
                comp_inflation = Frame.get_compound_return(ts)
                row2 = {self.name: comp_inflation}

                # max inflation
                max_inflation = self.rolling_inflation[dt:].nlargest(n=1)  # largest 12m inflation for selected period
                row3 = {self.name: max_inflation.iloc[0]}
                row3.update({'period': max_inflation.index.values[0].strftime('%Y-%m')})

                # purchase power
                row4 = {self.name: 1000. / (1. + comp_inflation)}
            else:
                row1 = {self.name: None}
                row2 = {self.name: None}
                row3 = {self.name: None}
                row3.update({'period': f'{i} years'})
                row4 = {self.name: None}
            row1.update({'period': f'{i} years'})
            row1.update({'property': 'annual inflation'})

            row2.update({'period': f'{i} years'})
            row2.update({'property': 'compound inflation'})

            row3.update({'property': 'max 12m inflation'})

            row4.update({'period': f'{i} years'})
            row4.update({'property': '1000 purchasing power'})

            description = description.append(row1, ignore_index=True)
            description = description.append(row2, ignore_index=True)
            description = description.append(row3, ignore_index=True)
            description = description.append(row4, ignore_index=True)
        # Annual inflation for full period available
        ts = df
        full_inflation = Frame.get_cagr(ts)
        row = {self.name: full_inflation}
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': 'annual inflation'})
        description = description.append(row, ignore_index=True)
        # compound inflation
        comp_inflation = Frame.get_compound_return(ts)
        row = {self.name: comp_inflation}
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': 'compound inflation'})
        description = description.append(row, ignore_index=True)
        # max inflation for full period available
        max_inflation = self.rolling_inflation.nlargest(n=1)
        row = {self.name: max_inflation.iloc[0]}
        row.update({'period': max_inflation.index.values[0].strftime('%Y-%m')})
        row.update({'property': 'max 12m inflation'})
        description = description.append(row, ignore_index=True)
        # purchase power
        row = {self.name: 1000. / (1. + comp_inflation)}
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': '1000 purchasing power'})
        description = description.append(row, ignore_index=True)

        description = Frame.change_columns_order(description, ['property', 'period'], position='first')
        return description


class Rate(MacroABC):
    """
    Rates of central banks and banks.
    """

    @property
    def okid(self) -> pd.Series:
        return Frame.get_okid_index(self.values_ts, self.symbol)

    def describe(self, years=[1, 5, 10]):
        # TODO: Make describe for OKID indexes
        pass
