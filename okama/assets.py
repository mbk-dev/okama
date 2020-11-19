from typing import Union, Optional, List, Tuple, Dict

import pandas as pd
import numpy as np

from .macro import Inflation
from .helpers import Float, Frame, Rebalance, Date
from .settings import default_ticker, assets_namespaces
from .data import QueryData


class Asset:
    """
    An asset, that could be used in a list of assets or in portfolio.
    """

    def __init__(self, symbol: str = default_ticker):
        self.symbol: str = symbol
        self._check_namespace()
        self._get_symbol_data(symbol)
        self.ror: pd.Series = QueryData.get_ror(symbol)
        self.first_date: pd.Timestamp = self.ror.index[0].to_timestamp()
        self.last_date: pd.Timestamp = self.ror.index[-1].to_timestamp()
        self.period_length: float = round((self.last_date - self.first_date) / np.timedelta64(365, 'D'), ndigits=1)

    def __repr__(self):
        dic = {
            'symbol': self.symbol,
            'name': self.name,
            'country': self.country,
            'exchange': self.exchange,
            'currency': self.currency,
            'type': self.type,
            'first date': self.first_date.strftime("%Y-%m"),
            'last date': self.last_date.strftime("%Y-%m"),
            'period length (Y)': "{:.2f}".format(self.period_length)
        }
        return repr(pd.Series(dic))

    def _check_namespace(self):
        namespace = self.symbol.split('.', 1)[-1]
        if namespace not in assets_namespaces:
            raise Exception(f'{namespace} is not in allowed assets namespaces: {assets_namespaces}')

    def _get_symbol_data(self, symbol) -> None:
        x = QueryData.get_symbol_info(symbol)
        self.ticker: str = x['code']
        self.name: str = x['name']
        self.country: str = x['country']
        self.exchange: str = x['exchange']
        self.currency: str = x['currency']
        self.type: str = x['type']
        self.inflation: str = f'{self.currency}.INFL'

    @property
    def price(self) -> float:
        """
        Live price of an asset.
        """
        return QueryData.get_live_price(self.symbol)

    @property
    def dividends(self) -> pd.Series:
        """
        Dividends time series daily data.
        Not defined for namespaces: 'PIF', 'INFL', 'INDX', 'FX', 'COMM'
        """
        div = QueryData.get_dividends(self.symbol)
        if div.empty:
            # Zero time series for assets where dividend yield is not defined.
            index = pd.date_range(start=self.first_date, end=self.last_date, freq='MS', closed=None)
            period = index.to_period('D')
            div = pd.Series(data=0, index=period)
            div.rename(self.symbol, inplace=True)
        return div

    @property
    def nav_ts(self) -> pd.Series:
        """
        NAV time series (monthly) for mutual funds when available in data.
        """
        if self.exchange == 'PIF':
            s = QueryData.get_nav(self.symbol)
            return s
        return np.nan


class AssetList:
    """
    The list of assets implementation.
    """
    def __init__(self,
                 symbols: Optional[List[str]] = None, *,
                 first_date: Optional[str] = None,
                 last_date: Optional[str] = None,
                 curr: str = 'USD',
                 inflation: bool = True):
        self.__symbols = symbols
        self.__tickers: List[str] = [x.split(".", 1)[0] for x in self.symbols]
        self.__currency: Asset = Asset(symbol=f'{curr}.FX')
        self.__make_asset_list(self.symbols)
        if inflation:
            self.inflation: str = f'{curr}.INFL'
            self._inflation_instance: Inflation = Inflation(self.inflation, self.first_date, self.last_date)
            self.inflation_ts: pd.Series = self._inflation_instance.values_ts
            self.inflation_first_date: pd.Timestamp = self._inflation_instance.first_date
            self.inflation_last_date: pd.Timestamp = self._inflation_instance.last_date
            self.first_date: pd.Timestamp = max(self.first_date, self.inflation_first_date)
            self.last_date: pd.Timestamp = min(self.last_date, self.inflation_last_date)
            # Add inflation to the date range dict
            self.assets_first_dates.update({self.inflation: self.inflation_first_date})
            self.assets_last_dates.update({self.inflation: self.inflation_last_date})
        if first_date:
            self.first_date: pd.Timestamp = max(self.first_date, pd.to_datetime(first_date))
        self.ror = self.ror[self.first_date:]
        if last_date:
            self.last_date: pd.Timestamp = min(self.last_date, pd.to_datetime(last_date))
        self.ror: pd.DataFrame = self.ror[self.first_date: self.last_date]
        self.period_length: float = round((self.last_date - self.first_date) / np.timedelta64(365, 'D'), ndigits=1)
        self._dividend_yield: pd.DataFrame = pd.DataFrame(dtype=float)
        self._dividends_ts: pd.DataFrame = pd.DataFrame(dtype=float)

    def __repr__(self):
        dic = {
            'symbols': self.symbols,
            'currency': self.currency.ticker,
            'first date': self.first_date.strftime("%Y-%m"),
            'last_date': self.last_date.strftime("%Y-%m"),
            'period length': self.period_length,
            'inflation': self.inflation if hasattr(self, 'inflation') else 'None',
        }
        return repr(pd.Series(dic))

    def __len__(self):
        return len(self.symbols)

    def __make_asset_list(self, ls: list) -> None:
        """
        Makes an asset list from a list of symbols. Returns DataFrame of returns (monthly) as an attribute.
        """
        first_dates: Dict[str, pd.Timestamp] = {}
        last_dates: Dict[str, pd.Timestamp] = {}
        names: Dict[str, str] = {}
        currencies: Dict[str, str] = {}
        for i, x in enumerate(ls):
            asset = Asset(x)
            if i == 0:
                if asset.currency == self.currency.name:
                    df = asset.ror
                else:
                    df = self._set_currency(returns=asset.ror, asset_currency=asset.currency)
            else:
                if asset.currency == self.currency.name:
                    new = asset.ror
                else:
                    new = self._set_currency(returns=asset.ror, asset_currency=asset.currency)
                df = pd.concat([df, new], axis=1, join='inner', copy='false')
            currencies.update({asset.symbol: asset.currency})
            names.update({asset.symbol: asset.name})
            first_dates.update({asset.symbol: asset.first_date})
            last_dates.update({asset.symbol: asset.last_date})
        # Add currency to the date range dict
        first_dates.update({self.currency.name: self.currency.first_date})
        last_dates.update({self.currency.name: self.currency.last_date})

        first_dates_sorted: list = sorted(first_dates.items(), key=lambda x: x[1])
        last_dates_sorted: list = sorted(last_dates.items(), key=lambda x: x[1])
        self.first_date: pd.Timestamp = first_dates_sorted[-1][1]
        self.last_date: pd.Timestamp = last_dates_sorted[0][1]
        self.newest_asset: str = first_dates_sorted[-1][0]
        self.eldest_asset: str = first_dates_sorted[0][0]
        self.names: Dict[str, str] = names
        currencies.update({'asset list': self.currency.currency})
        self.currencies: Dict[str, str] = currencies
        self.assets_first_dates: Dict[str, pd.Timestamp] = dict(first_dates_sorted)
        self.assets_last_dates: Dict[str, pd.Timestamp] = dict(last_dates_sorted)
        if isinstance(df, pd.Series):  # required to convert Series to DataFrame for single asset list
            df = df.to_frame()
        self.ror: pd.DataFrame = df

    def _set_currency(self, returns: pd.Series, asset_currency: str) -> pd.Series:
        """
        Set return to a certain currency. Input is a pd.Series of mean returns and a currency symbol.
        """
        currency = Asset(symbol=f'{asset_currency}{self.currency.name}.FX')
        asset_mult = returns + 1.
        currency_mult = currency.ror + 1.
        # join dataframes to have the same Time Series Index
        df = pd.concat([asset_mult, currency_mult], axis=1, join='inner', copy='false')
        currency_mult = df.iloc[:, -1]
        asset_mult = df.iloc[:, 0]
        x = asset_mult * currency_mult - 1.
        x.rename(returns.name, inplace=True)
        return x

    @property
    def symbols(self):
        if not self.__symbols:
            symbols = [default_ticker]
        else:
            symbols = self.__symbols
        if not isinstance(symbols, list):
            raise ValueError('Symbols should be a list of string values.')
        return symbols

    @property
    def tickers(self):
        return self.__tickers

    @property
    def currency(self):
        return self.__currency

    @property
    def wealth_indexes(self) -> pd.DataFrame:
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        return Frame.get_wealth_indexes(df)

    @property
    def risk_monthly(self) -> pd.Series:
        """
        Takes assets returns DataFrame and calculates monthly risks (std) for each asset.
        """
        return self.ror.std()

    @property
    def risk_annual(self) -> pd.Series:
        """
        Takes assets returns DataFrame and calculates annulized risks (std) for each asset.
        """
        risk = self.ror.std()
        mean_return = self.ror.mean()
        return Float.annualize_risk(risk, mean_return)

    @property
    def semideviation(self) -> pd.Series:
        return Frame.get_semideviation(self.ror)

    def get_var_historic(self, level: int = 5) -> pd.Series:
        return Frame.get_var_historic(self.ror, level)

    def get_cvar_historic(self, level: int = 5) -> pd.Series:
        return Frame.get_cvar_historic(self.ror, level)

    @property
    def drawdowns(self) -> pd.DataFrame:
        return Frame.get_drawdowns(self.ror)

    def get_cagr(self, period: Union[str, int, None] = None) -> pd.Series:
        """
        Calculates Compound Annual Growth Rate for a given period:
        None: full time
        'YTD': Year To Date compound rate of return (formally not a CAGR)
        Integer: several years
        """
        if hasattr(self, 'inflation'):
            df: pd.DataFrame = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        dt0 = self.last_date

        if not period:
            cagr = Frame.get_cagr(df)
        elif period == 'YTD':
            year = dt0.year
            cagr = (df[str(year):] + 1.).prod() - 1.
        elif isinstance(period, int):
            dt = Date.subtract_years(dt0, period)
            if dt >= self.first_date:
                cagr = Frame.get_cagr(df[dt:])
            else:
                row = {x: None for x in df.columns}
                cagr = pd.Series(row)
        else:
            raise ValueError(f'{period} is not a valid value for period')
        return cagr

    @property
    def annual_return_ts(self) -> pd.DataFrame:
        return Frame.get_annual_return_ts_from_monthly(self.ror)

    def describe(self, years: tuple = (1, 5, 10), tickers: bool = True) -> pd.DataFrame:
        """
        Generate descriptive statistics for a given list of tickers.
        Statistics includes:
        - YTD compound return
        - CAGR for a given list of periods
        - Dividend yield - yield for last 12 months (LTM)
        - risk (std) for a full period
        - CVAR for a full period
        - max drawdowns (and dates) for a full period
        - inception date - first date available for each asset
        - last asset date - available for each asset date
        - last data data - common for all assets data (may be set by last_date manually)
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        # YTD return
        ytd_return = self.get_cagr(period='YTD')
        row = ytd_return.to_dict()
        row.update({'period': 'YTD'})
        row.update({'property': 'Compound return'})
        description = description.append(row, ignore_index=True)
        # CAGR for a list of periods
        for i in years:
            dt = Date.subtract_years(dt0, i)
            if dt >= self.first_date:
                row = self.get_cagr(period=i).to_dict()
            else:
                row = {x: None for x in df.columns}
            row.update({'period': f'{i} years'})
            row.update({'property': 'CAGR'})
            description = description.append(row, ignore_index=True)
        # CAGR for full period
        row = self.get_cagr(period=None).to_dict()
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': 'CAGR'})
        description = description.append(row, ignore_index=True)
        # Dividend Yield
        row = self.dividend_yield.iloc[-1].to_dict()
        row.update({'period': 'LTM'})
        row.update({'property': 'Dividend yield'})
        description = description.append(row, ignore_index=True)
        # risk for full period
        row = self.risk_annual.to_dict()
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': 'Risk'})
        description = description.append(row, ignore_index=True)
        # CVAR
        row = self.get_cvar_historic().to_dict()
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': 'CVAR'})
        description = description.append(row, ignore_index=True)
        # max drawdowns
        row = self.drawdowns.min().to_dict()
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': 'Max drawdowns'})
        description = description.append(row, ignore_index=True)
        # max drawdowns dates
        row = self.drawdowns.idxmin().to_dict()
        row.update({'period': f'{self.period_length} years'})
        row.update({'property': 'Max drawdowns dates'})
        description = description.append(row, ignore_index=True)
        # inception dates
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_first_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update({'period': None})
        row.update({'property': 'Inception date'})
        if hasattr(self, 'inflation'):
            row.update({self.inflation: self.inflation_first_date.strftime("%Y-%m")})
        description = description.append(row, ignore_index=True)
        # last asset date
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_last_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update({'period': None})
        row.update({'property': 'Last asset date'})
        if hasattr(self, 'inflation'):
            row.update({self.inflation: self.inflation_last_date.strftime("%Y-%m")})
        description = description.append(row, ignore_index=True)
        # last data date
        row = {x: self.last_date.strftime("%Y-%m") for x in df.columns}
        row.update({'period': None})
        row.update({'property': 'Common last data date'})
        description = description.append(row, ignore_index=True)
        # rename columns
        if hasattr(self, 'inflation'):
            description.rename(columns={self.inflation: 'inflation'}, inplace=True)
            description = Frame.change_columns_order(description, ['inflation'], position='last')
        description = Frame.change_columns_order(description, ['property', 'period'], position='first')
        if not tickers:
            for ti in self.symbols:
                # short_ticker = ti.split(".", 1)[0]
                description.rename(columns={ti: self.names[ti]}, inplace=True)
        return description

    @property
    def mean_return(self) -> pd.Series:
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        mean: pd.Series = df.mean()
        return Float.annualize_return(mean)

    @property
    def real_mean_return(self) -> pd.Series:
        """
        Calculates real mean return (arithmetic mean).
        """
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            raise Exception('Real Return is not defined. Set inflation=True to calculate.')
        infl_mean = Float.annualize_return(self.inflation_ts.values.mean())
        ror_mean = Float.annualize_return(df.loc[:, self.symbols].mean())
        return (1. + ror_mean) / (1. + infl_mean) - 1.

    def _get_asset_dividends(self, tick, remove_forecast=True) -> pd.Series:
        first_period = pd.Period(self.first_date, freq='M')
        first_day = first_period.to_timestamp(how='Start')
        last_period = pd.Period(self.last_date, freq='M')
        last_day = last_period.to_timestamp(how='End')
        s = Asset(tick).dividends[first_day: last_day]  # limit divs by first_day and last_day
        if remove_forecast:
            s = s[:pd.Period.now(freq='D')]
        # Create time series with zeros to pad the empty spaces in dividends time series
        index = pd.date_range(start=first_day, end=last_day, freq='D')
        period = index.to_period('D')
        pad_s = pd.Series(data=0, index=period)
        z = s.add(pad_s, fill_value=0)
        return z

    def _get_dividends(self, remove_forecast=True) -> pd.DataFrame:
        if self._dividends_ts.empty:
            dic = {}
            for tick in self.symbols:
                s = self._get_asset_dividends(tick, remove_forecast=remove_forecast)
                dic.update({tick: s})
            self._dividends_ts = pd.DataFrame(dic)
        return self._dividends_ts

    @property
    def dividend_yield(self) -> pd.DataFrame:
        """
        Dividend yield (LTM) time series monthly.
        Calculates yield assuming original asset currency (not adjusting to AssetList currency).
        Forecast dividends are removed.
        """
        if self._dividend_yield.empty:
            frame = {}
            df = self._get_dividends(remove_forecast=True)
            for tick in self.symbols:
                # Get dividends time series
                div = df[tick]
                # Get close (not adjusted) values time series.
                # If the last_date month is current month live price of assets is used.
                if div.sum() != 0:
                    div_monthly = div.resample('M').sum()
                    price = QueryData.get_close(tick, period='M').loc[self.first_date: self.last_date]
                else:
                    # skipping prices if no dividends
                    div_yield = div.asfreq(freq='M')
                    frame.update({tick: div_yield})
                    continue
                if price.index[-1].month == pd.Timestamp.today().month:
                    price.loc[f'{pd.Timestamp.today().year}-{pd.Timestamp.today().month}'] = Asset(tick).price
                # Get dividend yield time series
                div_yield = pd.Series(dtype=float)
                div_monthly.index = div_monthly.index.to_timestamp()
                for date in price.index.to_timestamp(how='End'):
                    ltm_div = div_monthly[:date].last('12M').sum()
                    last_price = price.loc[:date].iloc[-1]
                    value = ltm_div / last_price
                    div_yield.at[date] = value
                div_yield.index = div_yield.index.to_period('M')
                # Currency adjusted yield
                # if self.currencies[tick] != self.currency.name:
                #     div_yield = self._set_currency(returns=div_yield, asset_currency=self.currencies[tick])
                frame.update({tick: div_yield})
            self._dividend_yield = pd.DataFrame(frame)
        return self._dividend_yield

    @property
    def dividends_annual(self) -> pd.DataFrame:
        """
        Time series of dividends for a calendar year.
        """
        df = self._get_dividends()
        df = df.resample('Y').sum()
        return df

    @property
    def dividend_growing_years(self) -> pd.DataFrame:
        """
        Returns the number of growing dividend years for each asset.
        """
        div_growth = self.dividends_annual.pct_change()[1:]
        df = pd.DataFrame()
        for name in div_growth:
            s = div_growth[name]
            s1 = s.where(s > 0).notnull().astype(int)
            s1_1 = s.where(s > 0).isnull().astype(int).cumsum()
            s2 = s1.groupby(s1_1).cumsum()
            df = pd.concat([df, s2], axis=1, copy='false')
        return df

    @property
    def dividend_paying_years(self) -> pd.DataFrame:
        """
        Returns the number of years of consecutive dividend payments.
        """
        div_annual = self.dividends_annual
        frame = pd.DataFrame()
        df = frame
        for name in div_annual:
            s = div_annual[name]
            s1 = s.where(s != 0).notnull().astype(int)
            s1_1 = s.where(s != 0).isnull().astype(int).cumsum()
            s2 = s1.groupby(s1_1).cumsum()
            df = pd.concat([df, s2], axis=1, copy='false')
        return df

    def get_dividend_mean_growth_rate(self, period=5) -> pd.Series:
        """
        Calculates geometric mean of dividends growth rate time series for a certain period.
        Period should be integer and not exceed the available data period_length.
        """
        if period <= self.period_length and isinstance(period, int):
            growth_ts = self.dividends_annual.pct_change().iloc[1:-1]  # Slice the last year for full dividends
            dt0 = self.last_date
            dt = Date.subtract_years(dt0, period)
            mean_growth_rate = ((growth_ts[dt:] + 1.).prod()) ** (1 / period) - 1.
        else:
            raise TypeError(f'{period} is not a valid value for period')
        return mean_growth_rate


class Portfolio:
    """
    Implementation of investment portfolio.
    Arguments are similar to AssetList (weights are added), but different behavior.
    """
    def __init__(self,
                 symbols: Optional[List[str]] = None, *,
                 first_date: Optional[str] = None,
                 last_date: Optional[str] = None,
                 curr: str = 'USD',
                 inflation: bool = True,
                 weights: Optional[List[float]] = None):
        self._list: AssetList = AssetList(symbols=symbols, first_date=first_date, last_date=last_date,
                                          curr=curr, inflation=inflation)
        self.currency: str = self._list.currency.name
        self._ror: pd.DataFrame = self._list.ror
        self.symbols: List[str] = self._list.symbols
        self.tickers: List[str] = [x.split(".", 1)[0] for x in self.symbols]
        self.names: Dict[str, str] = self._list.names
        self._weights = None
        self.weights = weights
        self.assets_weights = dict(zip(self.symbols, self.weights))
        self.assets_first_dates: Dict[str, pd.Timestamp] = self._list.assets_first_dates
        self.assets_last_dates: Dict[str, pd.Timestamp] = self._list.assets_last_dates
        self.first_date = self._list.first_date
        self.last_date = self._list.last_date
        self.period_length = self._list.period_length
        if inflation:
            self.inflation = self._list.inflation
            self.inflation_ts: pd.Series = self._list.inflation_ts

    def __repr__(self):
        dic = {
            'symbols': self.symbols,
            'weights': self.weights,
            'currency': self.currency,
            'first date': self.first_date.strftime("%Y-%m"),
            'last_date': self.last_date.strftime("%Y-%m"),
            'period length': self.period_length
        }
        return repr(pd.Series(dic))

    def __len__(self):
        return len(self.symbols)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: list):
        if weights is None:
            # Equally weighted portfolio
            n = len(self.symbols)  # number of assets
            weights = list(np.repeat(1/n, n))
            self._weights = weights
        else:
            Frame.weights_sum_is_one(weights)
            if len(weights) != len(self.symbols):
                raise Exception(f'Number of tickers ({len(self.symbols)}) should be equal '
                                f'to the weights number ({len(weights)})')
            self._weights = weights

    @property
    def returns_ts(self) -> pd.Series:
        s = Frame.get_portfolio_return_ts(self.weights, self._ror)
        s.rename('portfolio', inplace=True)
        return s

    @property
    def wealth_index(self) -> pd.DataFrame:
        if hasattr(self, 'inflation'):
            df = pd.concat([self.returns_ts, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.returns_ts
        df = Frame.get_wealth_indexes(df)
        if isinstance(df, pd.Series):  # return should always be DataFrame
            df = df.to_frame()
            df.rename({1: 'portfolio'}, axis='columns', inplace=True)
        return df

    @property
    def wealth_index_with_assets(self) -> pd.Series:
        if hasattr(self, 'inflation'):
            df = pd.concat([self.returns_ts, self._ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = pd.concat([self.returns_ts, self._ror], axis=1, join='inner', copy='false')
        return Frame.get_wealth_indexes(df)

    def get_rebalanced_portfolio_return_ts(self, period='Y') -> pd.Series:
        return Rebalance.rebalanced_portfolio_return_ts(self.weights, self._ror, period=period)

    @property
    def mean_return_monthly(self) -> float:
        return Frame.get_portfolio_mean_return(self.weights, self._ror)

    @property
    def mean_return_annual(self) -> float:
        return Float.annualize_return(self.mean_return_monthly)

    @property
    def cagr(self) -> Union[pd.Series, float]:
        if hasattr(self, 'inflation'):
            df = pd.concat([self.returns_ts, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.returns_ts
        return Frame.get_cagr(df)

    @property
    def annual_return_ts(self) -> pd.DataFrame:
        return Frame.get_annual_return_ts_from_monthly(self.returns_ts)

    @property
    def dividend_yield(self) -> pd.DataFrame:
        """
        Calculates dividend yield time series in all base currencies of portfolio assets.
        For every currency dividend yield is a weighted sum of the assets dividend yields.
        """
        div_yield_assets = self._list.dividend_yield
        currencies_dict = self._list.currencies
        if 'asset list' in currencies_dict:
            del currencies_dict['asset list']
        currencies_list = list(set(currencies_dict.values()))
        div_yield_df = pd.DataFrame(dtype=float)
        for currency in currencies_list:
            assets_with_the_same_currency = [x for x in currencies_dict if currencies_dict[x] == currency]
            df = div_yield_assets[assets_with_the_same_currency]
            weights = [self.assets_weights[k] for k in self.assets_weights if k in assets_with_the_same_currency]
            weighted_weights = np.asarray(weights) / np.asarray(weights).sum()
            div_yield_series = Frame.get_portfolio_return_ts(weighted_weights, df)
            div_yield_series.rename(currency, inplace=True)
            div_yield_df = pd.concat([div_yield_df, div_yield_series], axis=1)
        return div_yield_df

    @property
    def real_mean_return(self) -> float:
        if hasattr(self, 'inflation'):
            infl_mean = Float.annualize_return(self.inflation_ts.mean())
            ror_mean = Float.annualize_return(self.returns_ts.mean())
        else:
            raise Exception('Real Return is not defined. Set inflation=True to calculate.')
        return (1. + ror_mean) / (1. + infl_mean) - 1.

    @property
    def real_cagr(self) -> float:
        if hasattr(self, 'inflation'):
            infl_cagr = Frame.get_cagr(self.inflation_ts)
            ror_cagr = Frame.get_cagr(self.returns_ts)
        else:
            raise Exception('Real Return is not defined. Set inflation=True to calculate.')
        return (1. + ror_cagr) / (1. + infl_cagr) - 1.

    @property
    def risk_monthly(self) -> float:
        return Frame.get_portfolio_risk(self.weights, self._ror)

    @property
    def risk_annual(self) -> float:
        return Float.annualize_risk(self.risk_monthly, self.mean_return_monthly)

    @property
    def semideviation(self) -> float:
        return Frame.get_semideviation(self.returns_ts)

    def get_var_historic(self, level=5) -> float:
        rolling = self.returns_ts.rolling(12).apply(Frame.get_cagr)
        return Frame.get_var_historic(rolling, level)

    def get_cvar_historic(self, level=5) -> float:
        rolling = self.returns_ts.rolling(12).apply(Frame.get_cagr)
        return Frame.get_cvar_historic(rolling, level)

    @property
    def drawdowns(self) -> pd.Series:
        return Frame.get_drawdowns(self.returns_ts)

    def describe(self, years: Tuple[int] = (1, 5, 10)) -> pd.DataFrame:
        """
        Generate descriptive statistics for a given list of tickers.
        Statistics includes:
        - YTD compound return
        - CAGR for a given list of periods
        - risk (std) for a full period
        - CVAR for a full period
        - max drawdowns (and dates) for a full period
        TODO: add dividend yield
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        if hasattr(self, 'inflation'):
            df = pd.concat([self.returns_ts, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.returns_ts
        # YTD return
        year = pd.Timestamp.today().year
        ts = Rebalance.rebalanced_portfolio_return_ts(self.weights, self._ror[str(year):], period='N')
        value = Frame.get_compound_return(ts)
        if hasattr(self, 'inflation'):
            ts = df[str(year):].loc[:, self.inflation]
            inflation = Frame.get_compound_return(ts)
            row = {'portfolio': value, self.inflation: inflation}
        else:
            row = {'portfolio': value}
        row.update({'period': 'YTD'})
        row.update({'rebalancing': 'Not rebalanced'})
        row.update({'property': 'compound return'})
        description = description.append(row, ignore_index=True)
        # CAGR for a list of periods (rebalanced 1 month)
        for i in years:
            dt = Date.subtract_years(dt0, i)
            if dt >= self.first_date:
                ts = Rebalance.rebalanced_portfolio_return_ts(self.weights, self._ror[dt:], period='Y')
                value = Frame.get_cagr(ts)
                if hasattr(self, 'inflation'):
                    ts = df[dt:].loc[:, self.inflation]
                    inflation = Frame.get_cagr(ts)
                    row = {'portfolio': value, self.inflation: inflation}
                else:
                    row = {'portfolio': value}
            else:
                row = {x: None for x in df.columns}
            row.update({'period': f'{i} years'})
            row.update({'rebalancing': '1 year'})
            row.update({'property': 'CAGR'})
            description = description.append(row, ignore_index=True)
        # CAGR for full period (rebalanced 1 year)
        ts = Rebalance.rebalanced_portfolio_return_ts(self.weights, self._ror, period='Y')
        value = Frame.get_cagr(ts)
        if hasattr(self, 'inflation'):
            ts = df.loc[:, self.inflation]
            full_inflation = Frame.get_cagr(ts)  # full period inflation is required for following calc
            row = {'portfolio': value, self.inflation: full_inflation}
        else:
            row = {'portfolio': value}
        row.update({'period': f'{self.period_length} years'})
        row.update({'rebalancing': '1 year'})
        row.update({'property': 'CAGR'})
        description = description.append(row, ignore_index=True)
        # CAGR rebalanced 1 month
        value = self.cagr
        if hasattr(self, 'inflation'):
            row = value.to_dict()
            full_inflation = value.loc[self.inflation]  # full period inflation is required for following calc
        else:
            row = {'portfolio': value}
        row.update({'period': f'{self.period_length} years'})
        row.update({'rebalancing': '1 month'})
        row.update({'property': 'CAGR'})
        description = description.append(row, ignore_index=True)
        # CAGR not rebalanced
        value = Frame.get_cagr(self.get_rebalanced_portfolio_return_ts(period='N'))
        if hasattr(self, 'inflation'):
            row = {'portfolio': value, self.inflation: full_inflation}
        else:
            row = {'portfolio': value}
        row.update({'period': f'{self.period_length} years'})
        row.update({'rebalancing': 'Not rebalanced'})
        row.update({'property': 'CAGR'})
        description = description.append(row, ignore_index=True)
        # risk (rebalanced 1 month)
        row = {'portfolio': self.risk_annual}
        row.update({'period': f'{self.period_length} years'})
        row.update({'rebalancing': '1 month'})
        row.update({'property': 'Risk'})
        description = description.append(row, ignore_index=True)
        # CVAR (rebalanced 1 month)
        row = {'portfolio': self.get_cvar_historic()}
        row.update({'period': f'{self.period_length} years'})
        row.update({'rebalancing': '1 month'})
        row.update({'property': 'CVAR'})
        description = description.append(row, ignore_index=True)
        # max drawdowns (rebalanced 1 month)
        row = {'portfolio': self.drawdowns.min()}
        row.update({'period': f'{self.period_length} years'})
        row.update({'rebalancing': '1 month'})
        row.update({'property': 'Max drawdown'})
        description = description.append(row, ignore_index=True)
        # max drawdowns dates
        row = {'portfolio': self.drawdowns.idxmin()}
        row.update({'period': f'{self.period_length} years'})
        row.update({'rebalancing': '1 month'})
        row.update({'property': 'Max drawdown date'})
        description = description.append(row, ignore_index=True)
        if hasattr(self, 'inflation'):
            description.rename(columns={self.inflation: 'inflation'}, inplace=True)
        description = Frame.change_columns_order(description, ['property', 'rebalancing', 'period', 'portfolio'])
        return description

    @property
    def table(self) -> pd.DataFrame:
        """
        Returns security name - ticker - weight DataFrame table.
        """
        x = pd.DataFrame(data={'asset name': list(self.names.values()), 'ticker': list(self.names.keys())})
        x['weights'] = self.weights
        return x

    def get_rolling_return(self, years: int = 1) -> pd.Series:
        """
        Rolling portfolio rate of return time series.
        """
        rolling_return = (self.returns_ts + 1.).rolling(12 * years).apply(np.prod, raw=True) ** (1 / years) - 1.
        rolling_return.dropna(inplace=True)
        return rolling_return

    def forecast_from_history(self, percentiles: List[int] = [10, 50, 90]) -> pd.DataFrame:
        """
        Time series of future portfolio rate of returns for a given percentiles.
        Each percentile is calculated for a period range from 1 year to max forecast period
        from historic data rolling returns.
        Forecast max period is limited with half history of period length.
        """
        max_period = round(self.period_length / 2)
        if max_period < 1:
            raise Exception(f'Time series does not have enough history to forecast. '
                            f'Period length is {self.period_length:.2f} years. At least 2 years are required.')
        period_range = range(1, max_period + 1)
        returns_dict = dict()
        for percentile in percentiles:
            percentile_returns_list = [self.get_rolling_return(years).quantile(percentile / 100) for years in period_range]
            returns_dict.update({str(percentile): percentile_returns_list})
        df = pd.DataFrame(returns_dict, index=list(period_range))
        df.index.rename('years', inplace=True)
        return df

    def forecast_monte_carlo_norm_returns(self, years: int = 5, n: int = 100) -> pd.DataFrame:
        """
        Generates N random returns time series with normal distribution.
        Forecast period should not exceed 1/2 of portfolio history period length.
        """
        max_period_years = round(self.period_length / 2)
        if max_period_years < 1:
            raise ValueError(f'Time series does not have enough history to forecast.'
                             f'Period length is {self.period_length:.2f} years. At least 2 years are required.')
        if years > max_period_years:
            raise ValueError(f'Forecast period {years} years is not credible. '
                             f'It should not exceed 1/2 of portfolio history period length {self.period_length / 2} years')
        period_months = years * 12
        # make periods index where the shape is max_period
        start_period = self.last_date.to_period('M')
        end_period = self.last_date.to_period('M') + period_months - 1
        ts_index = pd.period_range(start_period, end_period, freq='M')
        # random returns
        random_returns = np.random.normal(self.mean_return_monthly, self.risk_monthly, (period_months, n))
        return_ts = pd.DataFrame(data=random_returns, index=ts_index)
        return return_ts

    def forecast_monte_carlo_norm_wealth_indexes(self, years: int = 5, n: int = 100) -> pd.DataFrame:
        """
        Generates N future wealth indexes with normally distributed monthly returns for a given period.
        """
        return_ts = self.forecast_monte_carlo_norm_returns(years=years, n=n)
        first_value = self.wealth_index['portfolio'].values[-1]
        forecast_wealth = Frame.get_wealth_indexes(return_ts, first_value)
        return forecast_wealth

    def forecast_monte_carlo_percentile_wealth_indexes(self,
                                                       years: int = 5,
                                                       percentiles: List[int] = [10, 50, 90],
                                                       today_value: Optional[int] = None,
                                                       n: int = 1000,
                                                       ) -> Dict[int, float]:
        """
        Calculates the final values of N forecasted wealth indexes with normal distribution assumption.
        Final values are taken for given percentiles.
        today_value - the value of portfolio today (before forecast period)
        """
        wealth_indexes = self.forecast_monte_carlo_norm_wealth_indexes(years=years, n=n)
        results = dict()
        for percentile in percentiles:
            value = wealth_indexes.iloc[-1, :].quantile(percentile / 100)
            results.update({percentile: value})
        if today_value:
            modifier = today_value / self.wealth_index['portfolio'].values[-1]
            results.update((x, y * modifier)for x, y in results.items())
        return results
